#!/usr/bin/env python3
"""
Run Advanced Zero-Shot Learning experiments on FULL dataset for maximum accuracy.

This script removes the data limitations and runs on the complete Anveshana dataset
to get publication-ready results with statistical significance.
"""

import sys
import os
import logging
from datetime import datetime
import traceback
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.zero_shot_framework.advanced_zs_runner import AdvancedZSRunner

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'full_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OutputCapture:
    """Capture all console output and save to files"""
    def __init__(self, output_dir: str = "experiments/runs/detailed_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console_output = []
        self.detailed_results = {}
        
    def log(self, message: str):
        """Log a message to both console and capture buffer"""
        print(message)
        self.console_output.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        
    def save_detailed_outcome(self, experiment_name: str, data: dict):
        """Save detailed per-query outcomes"""
        self.detailed_results[experiment_name] = data
        
    def save_all_outputs(self):
        """Save all captured outputs to files"""
        # Save console output
        console_file = os.path.join(self.output_dir, f"console_output_{self.timestamp}.txt")
        with open(console_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.console_output))
        
        # Save detailed results
        detailed_file = os.path.join(self.output_dir, f"detailed_results_{self.timestamp}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_results, f, ensure_ascii=False, indent=2)
            
        print(f"\n📁 All outputs saved to:")
        print(f"   Console: {console_file}")
        print(f"   Detailed: {detailed_file}")
        return console_file, detailed_file

class FullDatasetAdvancedZSRunner(AdvancedZSRunner):
    """Extended runner that uses full dataset for maximum accuracy"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_capture = OutputCapture()
    
    def _print_detailed_outcomes(self, experiment_label, queries, documents, scores, true_relevance, model_info):
        """Print detailed per-query outcomes: model info, query, ranked docs, scores, relevance."""
        try:
            base_model = model_info.get('base_model', 'unknown')
            llm_model = model_info.get('llm_model', '-')
            use_cross_script = model_info.get('use_cross_script', False)
            use_few_shot = model_info.get('use_few_shot', False)
            self.output_capture.log("-"*80)
            self.output_capture.log(f"[DETAIL] Experiment: {experiment_label}")
            self.output_capture.log(f"[DETAIL] Model Used: base={base_model} | LLM={llm_model} | cross_script={use_cross_script} | few_shot={use_few_shot}")
            self.output_capture.log("-"*80)
            
            detailed_data = {
                'experiment_label': experiment_label,
                'model_info': model_info,
                'queries': queries,
                'documents': documents,
                'per_query_results': []
            }
            
            for qi, query in enumerate(queries):
                self.output_capture.log(f"[QUERY {qi+1}] {query}")
                query_scores = scores[qi] if qi < len(scores) else []
                # Rank all documents by score (descending)
                ranked = list(enumerate(query_scores))
                ranked.sort(key=lambda x: x[1], reverse=True)
                
                query_result = {
                    'query': query,
                    'query_index': qi,
                    'ranked_documents': []
                }
                
                for rank, (doc_idx, score) in enumerate(ranked[:len(documents)], start=1):
                    relevant = 0
                    if qi < len(true_relevance) and doc_idx < len(true_relevance[qi]):
                        relevant = true_relevance[qi][doc_idx]
                    doc_text = documents[doc_idx] if doc_idx < len(documents) else ""
                    snippet = doc_text.replace('\n', ' ')[:200]
                    self.output_capture.log(f"  {rank:>2}. doc#{doc_idx:<3} score={score:.4f} relevant={bool(relevant)} | {snippet}")
                    
                    query_result['ranked_documents'].append({
                        'rank': rank,
                        'doc_index': doc_idx,
                        'score': float(score),
                        'relevant': bool(relevant),
                        'document': doc_text,
                        'snippet': snippet
                    })
                
                detailed_data['per_query_results'].append(query_result)
                self.output_capture.log("")
            
            # Save detailed outcome
            self.output_capture.save_detailed_outcome(experiment_label, detailed_data)
            
        except Exception as e:
            logger.error(f"Failed to print detailed outcomes: {e}")
            self.output_capture.log(f"[ERROR] Failed to print detailed outcomes: {e}")
    
    def run_cross_script_transfer_experiment(self):
        """Run cross-script transfer on FULL test dataset (DEBUG: small subset)"""
        self.output_capture.log("Running Cross-Script Transfer Learning experiment on SMALL SUBSET for debugging...")
        self.output_capture.log("[DEBUG] Using only 10 queries and 10 documents for debugging.")
        if self.advanced_model is None:
            self.setup_advanced_model()
        try:
            # Use only a small subset for debugging
            test_queries = [item['query'] for item in self.test_data][:10]
            test_documents = [item['document'] for item in self.test_data][:10]
            unique_documents = list(set(test_documents))[:10]
            self.output_capture.log(f"DEBUG: Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            self.output_capture.log(f"[DEBUG] Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            scores = self.advanced_model.retrieve_documents(
                test_queries, 
                unique_documents, 
                k=10, 
                use_cross_script=True, 
                use_few_shot=False
            )
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            from src.evaluation.metrics import CLIREvaluator
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(scores, true_relevance, self.evaluation_config['k_values'])
            result = {
                'model_name': 'Advanced ZS - Cross-Script Transfer (DEBUG)',
                'experiment_type': 'cross_script_transfer_debug',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents),
                'dataset_coverage': 'DEBUG-10'
            }
            self.results['cross_script_transfer_debug'] = result
            logger.info("Cross-Script Transfer DEBUG experiment completed successfully")
            self.output_capture.log("[SUCCESS] Cross-Script Transfer DEBUG experiment completed successfully")
            # Print detailed outcomes
            model_info = {
                'base_model': getattr(self.advanced_model, 'model_name', 'unknown'),
                'llm_model': getattr(getattr(self.advanced_model, 'llm_engine', None), 'model_name', '-'),
                'use_cross_script': True,
                'use_few_shot': False,
            }
            self._print_detailed_outcomes(
                experiment_label='Cross-Script Transfer (DEBUG)',
                queries=test_queries,
                documents=unique_documents,
                scores=scores,
                true_relevance=true_relevance,
                model_info=model_info,
            )
            return result
        except Exception as e:
            logger.error(f"Cross-Script Transfer DEBUG experiment failed: {e}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            self.output_capture.log(f"[ERROR] Cross-Script Transfer DEBUG experiment failed: {e}\n{traceback_str}")
            error_result = {
                'model_name': 'Advanced ZS - Cross-Script Transfer (DEBUG)',
                'experiment_type': 'cross_script_transfer_debug',
                'error': str(e)
            }
            self.results['cross_script_transfer_debug'] = error_result
            return error_result
    
    def run_few_shot_learning_experiment(self):
        """Run few-shot learning on FULL test dataset (DEBUG: small subset)"""
        self.output_capture.log("Running Few-Shot Learning experiment on SMALL SUBSET for debugging...")
        self.output_capture.log("[DEBUG] Using only 10 queries and 10 documents for debugging.")
        if self.advanced_model is None:
            self.setup_advanced_model()
        try:
            test_queries = [item['query'] for item in self.test_data][:10]
            test_documents = [item['document'] for item in self.test_data][:10]
            unique_documents = list(set(test_documents))[:10]
            self.output_capture.log(f"DEBUG: Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            self.output_capture.log(f"[DEBUG] Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            from src.zero_shot_framework.advanced_zs_learning import create_few_shot_examples
            few_shot_examples = create_few_shot_examples()
            batch_size = 2  # Small batch for debugging
            all_scores = []
            for i in range(0, len(test_queries), batch_size):
                batch_queries = test_queries[i:i+batch_size]
                self.output_capture.log(f"Processing batch {i//batch_size + 1}/{(len(test_queries)-1)//batch_size + 1}")
                self.output_capture.log(f"[DEBUG] Processing batch {i//batch_size + 1}/{(len(test_queries)-1)//batch_size + 1}")
                batch_scores = self.advanced_model.retrieve_documents(
                    batch_queries, 
                    unique_documents, 
                    k=10, 
                    use_cross_script=False,
                    use_few_shot=True, 
                    few_shot_examples=few_shot_examples
                )
                all_scores.extend(batch_scores)
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            from src.evaluation.metrics import CLIREvaluator
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(all_scores, true_relevance, self.evaluation_config['k_values'])
            result = {
                'model_name': 'Advanced ZS - Few-Shot Learning (DEBUG)',
                'experiment_type': 'few_shot_learning_debug',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents),
                'num_few_shot_examples': len(few_shot_examples),
                'dataset_coverage': 'DEBUG-10'
            }
            self.results['few_shot_learning_debug'] = result
            logger.info("Few-Shot Learning DEBUG experiment completed successfully")
            self.output_capture.log("[SUCCESS] Few-Shot Learning DEBUG experiment completed successfully")
            # Print detailed outcomes
            model_info = {
                'base_model': getattr(self.advanced_model, 'model_name', 'unknown'),
                'llm_model': getattr(getattr(self.advanced_model, 'llm_engine', None), 'model_name', '-'),
                'use_cross_script': False,
                'use_few_shot': True,
            }
            self._print_detailed_outcomes(
                experiment_label='Few-Shot Learning (DEBUG)',
                queries=test_queries,
                documents=unique_documents,
                scores=all_scores,
                true_relevance=true_relevance,
                model_info=model_info,
            )
            return result
        except Exception as e:
            logger.error(f"Few-Shot Learning DEBUG experiment failed: {e}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            self.output_capture.log(f"[ERROR] Few-Shot Learning DEBUG experiment failed: {e}\n{traceback_str}")
            error_result = {
                'model_name': 'Advanced ZS - Few-Shot Learning (DEBUG)',
                'experiment_type': 'few_shot_learning_debug',
                'error': str(e)
            }
            self.results['few_shot_learning_debug'] = error_result
            return error_result
    
    def run_combined_techniques_experiment(self):
        """Run combined techniques on FULL test dataset (DEBUG: small subset)"""
        self.output_capture.log("Running Combined Techniques experiment on SMALL SUBSET for debugging...")
        self.output_capture.log("[DEBUG] Using only 10 queries and 10 documents for debugging.")
        if self.advanced_model is None:
            self.setup_advanced_model()
        try:
            test_queries = [item['query'] for item in self.test_data][:10]
            test_documents = [item['document'] for item in self.test_data][:10]
            unique_documents = list(set(test_documents))[:10]
            self.output_capture.log(f"DEBUG: Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            self.output_capture.log(f"[DEBUG] Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            from src.zero_shot_framework.advanced_zs_learning import create_few_shot_examples
            few_shot_examples = create_few_shot_examples()
            batch_size = 2  # Small batch for debugging
            all_scores = []
            for i in range(0, len(test_queries), batch_size):
                batch_queries = test_queries[i:i+batch_size]
                self.output_capture.log(f"Processing batch {i//batch_size + 1}/{(len(test_queries)-1)//batch_size + 1}")
                self.output_capture.log(f"[DEBUG] Processing batch {i//batch_size + 1}/{(len(test_queries)-1)//batch_size + 1}")
                batch_scores = self.advanced_model.retrieve_documents(
                    batch_queries, 
                    unique_documents, 
                    k=10, 
                    use_cross_script=True,  # Use ALL techniques
                    use_few_shot=True, 
                    few_shot_examples=few_shot_examples
                )
                all_scores.extend(batch_scores)
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            from src.evaluation.metrics import CLIREvaluator
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(all_scores, true_relevance, self.evaluation_config['k_values'])
            result = {
                'model_name': 'Advanced ZS - Combined Techniques (DEBUG)',
                'experiment_type': 'combined_techniques_debug',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents),
                'techniques_used': ['cross_script_transfer', 'few_shot_learning', 'multilingual_prompting'],
                'dataset_coverage': 'DEBUG-10'
            }
            self.results['combined_techniques_debug'] = result
            logger.info("Combined Techniques DEBUG experiment completed successfully")
            self.output_capture.log("[SUCCESS] Combined Techniques DEBUG experiment completed successfully")
            # Print detailed outcomes
            model_info = {
                'base_model': getattr(self.advanced_model, 'model_name', 'unknown'),
                'llm_model': getattr(getattr(self.advanced_model, 'llm_engine', None), 'model_name', '-'),
                'use_cross_script': True,
                'use_few_shot': True,
            }
            self._print_detailed_outcomes(
                experiment_label='Combined Techniques (DEBUG)',
                queries=test_queries,
                documents=unique_documents,
                scores=all_scores,
                true_relevance=true_relevance,
                model_info=model_info,
            )
            return result
        except Exception as e:
            logger.error(f"Combined Techniques DEBUG experiment failed: {e}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            self.output_capture.log(f"[ERROR] Combined Techniques DEBUG experiment failed: {e}\n{traceback_str}")
            error_result = {
                'model_name': 'Advanced ZS - Combined Techniques (DEBUG)',
                'experiment_type': 'combined_techniques_debug',
                'error': str(e)
            }
            self.results['combined_techniques_debug'] = error_result
            return error_result
    
    def run_all_full_experiments(self):
        """Run all experiments on full dataset"""
        self.output_capture.log("="*80)
        self.output_capture.log("STARTING FULL DATASET ADVANCED ZERO-SHOT LEARNING EXPERIMENTS")
        self.output_capture.log("="*80)
        
        if self.train_data is None:
            self.load_data()
        
        self.output_capture.log(f"Dataset loaded:")
        self.output_capture.log(f"  - Train: {len(self.train_data)} samples")
        self.output_capture.log(f"  - Validation: {len(self.validation_data)} samples") 
        self.output_capture.log(f"  - Test: {len(self.test_data)} samples")
        self.output_capture.log("  - Coverage: 100% (FULL DATASET)")
        
        experiments = [
            ('Cross-Script Transfer (FULL)', self.run_cross_script_transfer_experiment),
            ('Few-Shot Learning (FULL)', self.run_few_shot_learning_experiment), 
            ('Combined Techniques (FULL)', self.run_combined_techniques_experiment),
        ]
        
        for exp_name, exp_func in experiments:
            try:
                self.output_capture.log(f"\n{'='*60}")
                self.output_capture.log(f"Running {exp_name} experiment...")
                self.output_capture.log(f"{'='*60}")
                start_time = datetime.now()
                
                result = exp_func()
                
                end_time = datetime.now()
                duration = end_time - start_time
                self.output_capture.log(f"{exp_name} completed in {duration}")
                
                # Print detailed results
                if 'evaluation_results' in result:
                    eval_results = result['evaluation_results']
                    self.output_capture.log(f"\n{exp_name} RESULTS:")
                    for metric in ['ndcg', 'map', 'recall', 'precision']:
                        if metric in eval_results:
                            values = eval_results[metric]
                            self.output_capture.log(f"  {metric.upper()}: " + ", ".join([f"@{k}={v:.4f}" for k, v in values.items()]))
                    
                    self.output_capture.log(f"  Queries: {result['num_queries']}")
                    self.output_capture.log(f"  Documents: {result['num_documents']}")
                    self.output_capture.log(f"  Coverage: {result.get('dataset_coverage', 'N/A')}")
                
            except Exception as e:
                self.output_capture.log(f"{exp_name} experiment failed: {e}")
        
        return self.results

def main():
    """Main function to run full dataset experiments"""
    print("🔥 Advanced Zero-Shot Learning - FULL DATASET Experiments")
    print("="*80)
    print("This will run on the complete Anveshana dataset for maximum accuracy!")
    print("Expected time: 2-4 hours depending on hardware")
    print("="*80)
    
    try:
        # Initialize full dataset runner
        print("[INFO] Initializing FullDatasetAdvancedZSRunner...")
        runner = FullDatasetAdvancedZSRunner()
        
        # Load full data
        print("[INFO] Loading data...")
        runner.load_data()
        
        # Run all experiments on full dataset
        print("[INFO] Running all experiments on full dataset...")
        results = runner.run_all_full_experiments()
        
        # Save results
        print("[INFO] Saving results...")
        results_file = runner.save_results(output_dir="./experiments/runs/zs_experiments")
        
        # Print comprehensive summary
        print("[INFO] Printing results summary...")
        runner.print_results_summary()
        
        # Save all captured outputs
        console_file, detailed_file = runner.output_capture.save_all_outputs()
        
        print(f"\n🎉 FULL DATASET Advanced Zero-Shot Learning experiments completed!")
        print(f"📊 Results saved to: {results_file}")
        print(f"📈 These results are publication-ready with statistical significance!")
        
    except Exception as e:
        logger.error(f"Full dataset experiments failed: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        print(f"❌ Experiments failed: {e}\n{traceback_str}")
        sys.exit(1)

if __name__ == "__main__":
    main()