# Auto-download NLTK data when missing
import nltk
import ssl

# Fix SSL issues on Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data (only runs once)
required_nltk_data = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
for package in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package, quiet=False)

# Now continue with your normal imports
import streamlit as st
import json
from typing import List, Dict
import pandas as pd
import plotly.express as px
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class GeneralRAGEvaluator:
    def __init__(self, vectorstore, test_dataset_path="test_dataset.json"):
        self.vectorstore = vectorstore
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)['test_cases']
    
    def calculate_smarter_metrics(self, expected: str, actual: str, question: str = "") -> dict:
        """Dynamic evaluation that works for ANY document"""
        
        import re
        from collections import Counter
        
        # Clean texts
        expected_clean = expected.lower()
        actual_clean = actual.lower()
        
        # 1. Extract key phrases dynamically
        def extract_key_phrases(text):
            """Extract important n-grams from text"""
            words = text.split()
            phrases = set()
            
            # Single important words (length > 3, not common)
            common_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 
                           'have', 'are', 'was', 'were', 'has', 'been', 'will'}
            
            for word in words:
                if len(word) > 3 and word not in common_words:
                    phrases.add(word)
            
            # 2-word phrases
            for i in range(len(words)-1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:
                    phrases.add(phrase)
            
            # 3-word phrases (key for technical terms)
            for i in range(len(words)-2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 8:
                    phrases.add(phrase)
            
            return phrases
        
        expected_phrases = extract_key_phrases(expected_clean)
        
        if not expected_phrases:
            return {'factual_accuracy': 0.0, 'passed': False}
        
        # 2. Check phrase coverage
        found_phrases = [p for p in expected_phrases if p in actual_clean]
        coverage = len(found_phrases) / len(expected_phrases)
        
        # 3. Extract and compare dates
        date_pattern = r'\b(?:\d{4}|\w+ \d{4}|\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2},? \d{4})\b'
        expected_dates = set(re.findall(date_pattern, expected))
        actual_dates = set(re.findall(date_pattern, actual))
        date_score = len(expected_dates & actual_dates) / len(expected_dates) if expected_dates else 1.0
        
        # 4. Extract and compare numbers
        number_pattern = r'\b\d+(?:\.\d+)?(?:\s*[+-])?\s*(?:years?|months?|days?|users?|%|dollars?|\$)?\b'
        expected_numbers = set(re.findall(number_pattern, expected))
        actual_numbers = set(re.findall(number_pattern, actual))
        number_score = len(expected_numbers & actual_numbers) / len(expected_numbers) if expected_numbers else 1.0
        
        # 5. Check for hallucinations
        hallucination_phrases = ['no mention', 'not found', 'does not mention', 
                                 'not provided', 'not in the context', "don't know"]
        has_hallucination = any(phrase in actual_clean for phrase in hallucination_phrases)
        
        # Also check if answer is suspiciously short
        if len(actual.split()) < 5 and has_hallucination:
            has_hallucination = True
        
        # 6. Calculate final factual accuracy
        factual_accuracy = (coverage * 0.5 + date_score * 0.25 + number_score * 0.25)
        
        # 7. Adjust for hallucinations (big penalty)
        if has_hallucination:
            factual_accuracy *= 0.3
        
        # 8. Determine pass (adjustable based on question complexity)
        passed = factual_accuracy >= 0.5 and not has_hallucination
        
        return {
            'factual_accuracy': round(factual_accuracy, 3),
            'coverage': round(coverage, 3),
            'date_score': round(date_score, 3),
            'number_score': round(number_score, 3),
            'found_phrases': found_phrases[:5],
            'missing_phrases': list(expected_phrases - set(found_phrases))[:5],
            'has_hallucination': has_hallucination,
            'passed': passed
        }
    
    def get_retrieved_contexts(self, question: str, k: int = 3):
        """Get retrieved document chunks for analysis"""
        docs = self.vectorstore.similarity_search(question, k=k)
        return [doc.page_content for doc in docs]
    
    def answer_question_safe(self, question: str) -> str:
        """Safely answer question with error handling"""
        try:
            from rag_pipeline import answer_question
            return answer_question(self.vectorstore, question)
        except Exception as e:
            return f"[Error generating answer: {str(e)}]"
    
    def run_evaluation(self):
        """Run evaluation on all test cases"""
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            with st.spinner(f"Evaluating test case {i+1}/{len(self.test_cases)}: {test_case['question'][:50]}..."):
                # Get RAG answer
                rag_answer = self.answer_question_safe(test_case['question'])
                
                # Calculate metrics dynamically
                metrics = self.calculate_smarter_metrics(
                    test_case['expected_answer'], 
                    rag_answer,
                    test_case['question']
                )
                
                # Get retrieved contexts
                retrieved_contexts = self.get_retrieved_contexts(test_case['question'])
                
                results.append({
                    'question': test_case['question'],
                    'expected_answer': test_case['expected_answer'],
                    'rag_answer': rag_answer,
                    'metrics': metrics,
                    'retrieved_contexts': retrieved_contexts
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict]):
        """Calculate aggregate metrics"""
        if not results:
            return {}
        
        metrics = {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r['metrics']['passed']),
            'avg_factual_accuracy': np.mean([r['metrics']['factual_accuracy'] for r in results]),
            'avg_coverage': np.mean([r['metrics']['coverage'] for r in results]),
            'hallucination_rate': sum(1 for r in results if r['metrics']['has_hallucination']) / len(results),
            'failed_retrievals': 0  # Will be calculated separately if needed
        }
        
        metrics['pass_rate'] = metrics['passed_tests'] / metrics['total_tests']
        
        return metrics

def create_evaluation_dashboard():
    st.title("🔍 RAG System Evaluation Harness")
    
    # Initialize session state
    if 'eval_results' not in st.session_state:
        st.session_state['eval_results'] = None
    if 'eval_metrics' not in st.session_state:
        st.session_state['eval_metrics'] = None
    
    # Sidebar controls
    st.sidebar.header("📊 Evaluation Controls")
    
    # Test dataset management
    st.sidebar.subheader("Test Dataset")
    if st.sidebar.button("📝 View/Create Test Cases"):
        st.session_state['show_test_cases'] = True
    
    # Run evaluation button
    if st.sidebar.button("🚀 Run Full Evaluation", type="primary"):
        with st.spinner("Loading vector store and running evaluation..."):
            try:
                from rag_pipeline import load_vector_store
                vectorstore = load_vector_store()
                
                # Initialize evaluator
                evaluator = GeneralRAGEvaluator(vectorstore)
                
                # Run evaluation
                results = evaluator.run_evaluation()
                metrics = evaluator.calculate_metrics(results)
                
                # Store in session state
                st.session_state['eval_results'] = results
                st.session_state['eval_metrics'] = metrics
                
                st.sidebar.success("✅ Evaluation complete!")
            except FileNotFoundError:
                st.sidebar.error("❌ Vector database not found. Please upload a PDF document in the Support Assistant tab first.")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Display metrics if available
    if st.session_state['eval_metrics']:
        metrics = st.session_state['eval_metrics']
        
        # Key metrics at the top
        st.subheader("📈 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", metrics['total_tests'])
        with col2:
            st.metric("Pass Rate", f"{metrics['pass_rate']:.1%}")
        with col3:
            st.metric("Avg Factual Accuracy", f"{metrics['avg_factual_accuracy']:.2f}")
        with col4:
            st.metric("Hallucination Rate", f"{metrics['hallucination_rate']:.1%}")
        
        # Detailed metrics
        st.subheader("📊 Detailed Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Coverage", f"{metrics['avg_coverage']:.2f}")
        with col2:
            st.metric("Passed Tests", f"{metrics['passed_tests']}/{metrics['total_tests']}")
        
        # Visualization
        st.subheader("📊 Performance Visualization")
        
        # Create bar chart for pass/fail
        fig = px.bar(
            x=['Passed', 'Failed'],
            y=[metrics['passed_tests'], metrics['total_tests'] - metrics['passed_tests']],
            title="Test Results Overview",
            color=['Passed', 'Failed'],
            color_discrete_map={'Passed': '#2ecc71', 'Failed': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Test Results")
        
        results_df = pd.DataFrame([
            {
                'Question': r['question'][:80] + '...' if len(r['question']) > 80 else r['question'],
                'Factual Accuracy': f"{r['metrics']['factual_accuracy']:.2f}",
                'Coverage': f"{r['metrics']['coverage']:.2f}",
                'Hallucination': "⚠️ Yes" if r['metrics']['has_hallucination'] else "✅ No",
                'Pass/Fail': "✅ Pass" if r['metrics']['passed'] else "❌ Fail"
            }
            for r in st.session_state['eval_results']
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Failed cases analysis
        st.subheader("❌ Failed Test Cases Analysis")
        failed_cases = [r for r in st.session_state['eval_results'] if not r['metrics']['passed']]
        
        if failed_cases:
            st.warning(f"Found {len(failed_cases)} failed test cases")
            
            for i, failed in enumerate(failed_cases[:3]):  # Show top 3 failures
                with st.expander(f"Failure #{i+1}: {failed['question']}"):
                    st.markdown("**❌ Expected Answer:**")
                    st.info(failed['expected_answer'])
                    
                    st.markdown("**🤖 RAG Answer:**")
                    st.warning(failed['rag_answer'])
                    
                    st.markdown("**📊 Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Factual Accuracy", f"{failed['metrics']['factual_accuracy']:.2f}")
                    with col2:
                        st.metric("Coverage", f"{failed['metrics']['coverage']:.2f}")
                    with col3:
                        st.metric("Hallucination", "⚠️ Yes" if failed['metrics']['has_hallucination'] else "✅ No")
                    
                    if failed['metrics']['missing_phrases']:
                        st.markdown("**Missing key phrases:**")
                        st.write(", ".join(failed['metrics']['missing_phrases'][:3]))
                    
                    st.markdown("**📚 Retrieved Contexts:**")
                    for j, ctx in enumerate(failed['retrieved_contexts'][:2]):
                        with st.expander(f"Context {j+1}"):
                            st.text(ctx[:300] + "..." if len(ctx) > 300 else ctx)
        else:
            st.success("🎉 Amazing! All test cases passed! Your RAG system is performing perfectly!")
        
        # Regression testing
        st.subheader("🔄 Regression Testing")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save as Baseline"):
                baseline = {
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat(),
                    'results_summary': {
                        'total_tests': metrics['total_tests'],
                        'pass_rate': metrics['pass_rate'],
                        'avg_factual_accuracy': metrics['avg_factual_accuracy']
                    }
                }
                with open('baseline_results.json', 'w') as f:
                    json.dump(baseline, f, indent=2)
                st.success("Baseline saved!")
        
        with col2:
            if st.button("📊 Compare with Baseline"):
                try:
                    with open('baseline_results.json', 'r') as f:
                        baseline = json.load(f)
                    
                    st.markdown("### Baseline Comparison")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_rate = metrics['pass_rate']
                        baseline_rate = baseline['metrics']['pass_rate']
                        delta = current_rate - baseline_rate
                        st.metric("Pass Rate", 
                                 f"{current_rate:.1%}",
                                 delta=f"{delta:+.1%}")
                    
                    with col2:
                        current_acc = metrics['avg_factual_accuracy']
                        baseline_acc = baseline['metrics']['avg_factual_accuracy']
                        delta_acc = current_acc - baseline_acc
                        st.metric("Factual Accuracy",
                                 f"{current_acc:.2f}",
                                 delta=f"{delta_acc:+.2f}")
                    
                    with col3:
                        current_hall = metrics['hallucination_rate']
                        baseline_hall = baseline['metrics']['hallucination_rate']
                        delta_hall = current_hall - baseline_hall
                        st.metric("Hallucination Rate",
                                 f"{current_hall:.1%}",
                                 delta=f"{delta_hall:+.1%}")
                    
                    if delta < 0 or delta_acc < 0:
                        st.warning("⚠️ Performance regression detected! Review your recent changes.")
                    elif delta > 0 or delta_acc > 0:
                        st.success("✅ Performance improved!")
                    else:
                        st.info("Performance unchanged from baseline")
                        
                except FileNotFoundError:
                    st.info("No baseline found. Save current results first.")
    
    else:
        # Show placeholder when no evaluation has been run
        st.info("👈 Click 'Run Full Evaluation' in the sidebar to start testing your RAG system")
        
        st.markdown("""
        ### What this evaluation harness does:
        
        1. **Tests your RAG system** with predefined question-answer pairs
        2. **Calculates smart metrics**:
           - Factual accuracy (how correct the information is)
           - Key phrase coverage (important terms present)
           - Date and number matching
           - Hallucination detection
        3. **Identifies failures** and shows you exactly where your system breaks
        4. **Tracks regressions** - detect if changes make your system worse
        
        ### Next Steps:
        1. First upload a PDF in the "Support Assistant" tab
        2. Create test cases in `test_dataset.json`
        3. Click "Run Full Evaluation" to see results
        """)

def show_test_cases_creator():
    """Helper function to create/edit test cases"""
    st.subheader("📝 Test Case Manager")
    
    # Load existing test cases
    try:
        with open('test_dataset.json', 'r') as f:
            test_data = json.load(f)
        test_cases = test_data['test_cases']
    except:
        test_cases = []
    
    # Display existing test cases
    if test_cases:
        st.write(f"Found {len(test_cases)} test cases")
        
        for i, test in enumerate(test_cases):
            with st.expander(f"Test Case {i+1}: {test['question'][:50]}..."):
                new_question = st.text_area("Question", test['question'], key=f"q_{i}")
                new_answer = st.text_area("Expected Answer", test['expected_answer'], key=f"a_{i}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Update", key=f"update_{i}"):
                        test_cases[i]['question'] = new_question
                        test_cases[i]['expected_answer'] = new_answer
                        with open('test_dataset.json', 'w') as f:
                            json.dump({'test_cases': test_cases}, f, indent=2)
                        st.success("Updated!")
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_{i}"):
                        test_cases.pop(i)
                        with open('test_dataset.json', 'w') as f:
                            json.dump({'test_cases': test_cases}, f, indent=2)
                        st.success("Deleted!")
                        st.rerun()
    
    # Add new test case
    st.subheader("Add New Test Case")
    with st.form("new_test_case"):
        new_question = st.text_input("Question")
        new_answer = st.text_area("Expected Answer")
        submitted = st.form_submit_button("Add Test Case")
        
        if submitted and new_question and new_answer:
            test_cases.append({
                'question': new_question,
                'expected_answer': new_answer,
                'context_docs': ["custom.pdf"]
            })
            
            # Save to file
            with open('test_dataset.json', 'w') as f:
                json.dump({'test_cases': test_cases}, f, indent=2)
            
            st.success("Test case added!")
            st.rerun()

if __name__ == "__main__":
    create_evaluation_dashboard()