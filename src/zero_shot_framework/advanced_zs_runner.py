"""
Advanced Zero-Shot Learning Runner for Sanskrit-English CLIR

This runner implements and tests the advanced zero-shot learning techniques:
1. Cross-Script Transfer Learning
2. Multilingual Prompting with LLMs
3. In-Context Learning
4. Parameter-Efficient Fine-Tuning (LoRA)
"""

import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
from datetime import datetime

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.utils.config_loader import ConfigLoader
from src.utils.data_loader import AnveshanaDataLoader
from src.evaluation.metrics import CLIREvaluator
from .advanced_zs_learning import (
    AdvancedZeroShotModel, 
    create_few_shot_examples,
    CrossScriptConfig,
    LoRAConfig
)

logger = logging.getLogger(__name__)

class AdvancedZSRunner:
    """Runner for Advanced Zero-Shot Learning experiments"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_full_config()
        self.zs_config = self.config['models']['zero_shot']
        self.evaluation_config = self.config['evaluation']
        
        # Data
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
        # Results storage
        self.results = {}
        
        # Advanced model
        self.advanced_model = None
        
        logger.info("Advanced ZS Runner initialized")
    
    def load_data(self) -> None:
        """Load and preprocess the dataset"""
        logger.info("Loading and preprocessing dataset...")
        
        data_loader = AnveshanaDataLoader(self.config['dataset'])
        data_loader.load_dataset()
        data_loader.preprocess_dataset()
        data_loader.create_splits()
        data_loader.create_negative_samples(data_loader.train_data)
        
        self.train_data = data_loader.train_data
        self.validation_data = data_loader.validation_data
        self.test_data = data_loader.test_data
        
        logger.info("Data loaded successfully:")
        logger.info(f"  - Train: {len(self.train_data)} samples")
        logger.info(f"  - Validation: {len(self.validation_data)} samples")
        logger.info(f"  - Test: {len(self.test_data)} samples")
    
    def setup_advanced_model(self, model_name: str = 'xlm-roberta-base') -> None:
        """Setup the advanced zero-shot model"""
        logger.info(f"Setting up advanced zero-shot model: {model_name}")
        
        # Configuration for advanced model
        advanced_config = {
            'model_name': model_name,
            'source_languages': ['hindi', 'marathi', 'nepali'],
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'llm_model': 'facebook/mbart-large-50-many-to-many-mmt'  # Smaller LLM for testing
        }
        
        # Try to use a smaller model if the main one fails
        try:
            self.advanced_model = AdvancedZeroShotModel(advanced_config)
            logger.info("Advanced model setup completed")
        except Exception as e:
            logger.warning(f"Failed to setup {model_name}: {e}")
            # Fallback to smaller model
            advanced_config['model_name'] = 'distilbert-base-multilingual-cased'
            advanced_config['llm_model'] = 'google/mt5-small'
            self.advanced_model = AdvancedZeroShotModel(advanced_config)
            logger.info("Advanced model setup completed with fallback models")
    
    def run_cross_script_transfer_experiment(self) -> Dict[str, Any]:
        """Run cross-script transfer learning experiment"""
        logger.info("Running Cross-Script Transfer Learning experiment...")
        
        if self.advanced_model is None:
            self.setup_advanced_model()
        
        try:
            # Prepare test data
            test_queries = [item['query'] for item in self.test_data]
            test_documents = [item['document'] for item in self.test_data]
            
            # Limit for testing (full dataset can be memory intensive)
            max_test_size = min(50, len(test_queries))
            test_queries = test_queries[:max_test_size]
            test_documents = test_documents[:max_test_size]
            
            # Get unique documents
            unique_documents = list(set(test_documents))
            logger.info(f"Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            
            # Run retrieval with cross-script transfer
            scores = self.advanced_model.retrieve_documents(
                test_queries, 
                unique_documents, 
                k=10, 
                use_cross_script=True, 
                use_few_shot=False
            )
            
            # Prepare for evaluation
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            
            # Evaluate
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(scores, true_relevance, self.evaluation_config['k_values'])
            
            result = {
                'model_name': 'Advanced ZS - Cross-Script Transfer',
                'experiment_type': 'cross_script_transfer',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents)
            }
            
            self.results['cross_script_transfer'] = result
            logger.info("Cross-Script Transfer experiment completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Cross-Script Transfer experiment failed: {e}")
            error_result = {
                'model_name': 'Advanced ZS - Cross-Script Transfer',
                'experiment_type': 'cross_script_transfer',
                'error': str(e)
            }
            self.results['cross_script_transfer'] = error_result
            return error_result
    
    def run_few_shot_learning_experiment(self) -> Dict[str, Any]:
        """Run few-shot learning with in-context examples"""
        logger.info("Running Few-Shot Learning experiment...")
        
        if self.advanced_model is None:
            self.setup_advanced_model()
        
        try:
            # Prepare test data
            test_queries = [item['query'] for item in self.test_data]
            test_documents = [item['document'] for item in self.test_data]
            
            # Limit for testing
            max_test_size = min(30, len(test_queries))  # Smaller for LLM processing
            test_queries = test_queries[:max_test_size]
            test_documents = test_documents[:max_test_size]
            
            # Get unique documents
            unique_documents = list(set(test_documents))
            logger.info(f"Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            
            # Create few-shot examples
            few_shot_examples = create_few_shot_examples()
            
            # Run retrieval with few-shot learning
            scores = self.advanced_model.retrieve_documents(
                test_queries, 
                unique_documents, 
                k=10, 
                use_cross_script=False,  # Focus on few-shot
                use_few_shot=True, 
                few_shot_examples=few_shot_examples
            )
            
            # Prepare for evaluation
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            
            # Evaluate
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(scores, true_relevance, self.evaluation_config['k_values'])
            
            result = {
                'model_name': 'Advanced ZS - Few-Shot Learning',
                'experiment_type': 'few_shot_learning',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents),
                'num_few_shot_examples': len(few_shot_examples)
            }
            
            self.results['few_shot_learning'] = result
            logger.info("Few-Shot Learning experiment completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Few-Shot Learning experiment failed: {e}")
            error_result = {
                'model_name': 'Advanced ZS - Few-Shot Learning',
                'experiment_type': 'few_shot_learning',
                'error': str(e)
            }
            self.results['few_shot_learning'] = error_result
            return error_result
    
    def run_combined_techniques_experiment(self) -> Dict[str, Any]:
        """Run experiment combining all advanced techniques"""
        logger.info("Running Combined Techniques experiment...")
        
        if self.advanced_model is None:
            self.setup_advanced_model()
        
        try:
            # Prepare test data
            test_queries = [item['query'] for item in self.test_data]
            test_documents = [item['document'] for item in self.test_data]
            
            # Limit for testing
            max_test_size = min(25, len(test_queries))  # Conservative for combined approach
            test_queries = test_queries[:max_test_size]
            test_documents = test_documents[:max_test_size]
            
            # Get unique documents
            unique_documents = list(set(test_documents))
            logger.info(f"Testing with {len(test_queries)} queries and {len(unique_documents)} documents")
            
            # Create few-shot examples
            few_shot_examples = create_few_shot_examples()
            
            # Run retrieval with all techniques combined
            scores = self.advanced_model.retrieve_documents(
                test_queries, 
                unique_documents, 
                k=10, 
                use_cross_script=True, 
                use_few_shot=True, 
                few_shot_examples=few_shot_examples
            )
            
            # Prepare for evaluation
            true_relevance = self._prepare_relevance_labels(test_queries, test_documents, unique_documents)
            
            # Evaluate
            evaluator = CLIREvaluator(k_values=self.evaluation_config['k_values'])
            evaluation_results = evaluator.evaluate_batch(scores, true_relevance, self.evaluation_config['k_values'])
            
            result = {
                'model_name': 'Advanced ZS - Combined Techniques',
                'experiment_type': 'combined_techniques',
                'evaluation_results': evaluation_results,
                'num_queries': len(test_queries),
                'num_documents': len(unique_documents),
                'techniques_used': ['cross_script_transfer', 'few_shot_learning', 'multilingual_prompting']
            }
            
            self.results['combined_techniques'] = result
            logger.info("Combined Techniques experiment completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Combined Techniques experiment failed: {e}")
            error_result = {
                'model_name': 'Advanced ZS - Combined Techniques',
                'experiment_type': 'combined_techniques',
                'error': str(e)
            }
            self.results['combined_techniques'] = error_result
            return error_result
    
    def run_parameter_efficient_fine_tuning_demo(self) -> Dict[str, Any]:
        """Demonstrate parameter-efficient fine-tuning (LoRA)"""
        logger.info("Running Parameter-Efficient Fine-Tuning demo...")
        
        if self.advanced_model is None:
            self.setup_advanced_model()
        
        try:
            # Prepare training data (small subset for demo)
            train_queries = [item['query'] for item in self.train_data[:100]]
            train_documents = [item['document'] for item in self.train_data[:100]]
            train_labels = list(range(len(train_queries)))  # Simplified labeling
            
            # Create output directory
            output_dir = "./experiments/runs/advanced_zs_lora"
            os.makedirs(output_dir, exist_ok=True)
            
            # Note: Full fine-tuning is commented out to avoid long training times in demo
            # Uncomment the following line to actually perform fine-tuning:
            # self.advanced_model.parameter_efficient_fine_tune(train_queries, train_documents, train_labels, output_dir)
            
            result = {
                'model_name': 'Advanced ZS - LoRA Fine-tuning (Demo)',
                'experiment_type': 'parameter_efficient_fine_tuning',
                'status': 'demo_completed',
                'note': 'Full fine-tuning skipped in demo - uncomment line 259 to enable',
                'output_dir': output_dir,
                'training_samples': len(train_queries)
            }
            
            self.results['parameter_efficient_fine_tuning'] = result
            logger.info("Parameter-Efficient Fine-Tuning demo completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter-Efficient Fine-Tuning demo failed: {e}")
            error_result = {
                'model_name': 'Advanced ZS - LoRA Fine-tuning',
                'experiment_type': 'parameter_efficient_fine_tuning',
                'error': str(e)
            }
            self.results['parameter_efficient_fine_tuning'] = error_result
            return error_result
    
    def _prepare_relevance_labels(self, queries: List[str], documents: List[str], unique_documents: List[str]) -> List[List[int]]:
        """Prepare relevance labels for evaluation"""
        relevance_labels = []
        
        for query in queries:
            query_labels = []
            for unique_doc in unique_documents:
                # Simple relevance: 1 if document appears with this query in test data, 0 otherwise
                relevant = 0
                for i, test_query in enumerate([item['query'] for item in self.test_data]):
                    if test_query == query and self.test_data[i]['document'] == unique_doc:
                        relevant = 1
                        break
                query_labels.append(relevant)
            relevance_labels.append(query_labels)
        
        return relevance_labels
    
    def run_all_advanced_experiments(self) -> Dict[str, Any]:
        """Run all advanced zero-shot learning experiments"""
        logger.info("Starting all advanced zero-shot learning experiments...")
        
        if self.train_data is None:
            self.load_data()
        
        experiments = [
            ('Cross-Script Transfer', self.run_cross_script_transfer_experiment),
            ('Few-Shot Learning', self.run_few_shot_learning_experiment),
            ('Combined Techniques', self.run_combined_techniques_experiment),
            ('Parameter-Efficient Fine-Tuning', self.run_parameter_efficient_fine_tuning_demo)
        ]
        
        for exp_name, exp_func in experiments:
            try:
                logger.info(f"Running {exp_name} experiment...")
                result = exp_func()
                logger.info(f"{exp_name} experiment completed")
                
                # Print summary
                if 'evaluation_results' in result:
                    ndcg_10 = result['evaluation_results'].get('ndcg', {}).get(10, 'N/A')
                    map_10 = result['evaluation_results'].get('map', {}).get(10, 'N/A')
                    logger.info(f"{exp_name} Results - NDCG@10: {ndcg_10}, MAP@10: {map_10}")
                
            except Exception as e:
                logger.error(f"{exp_name} experiment failed: {e}")
        
        return self.results
    
    def save_results(self, output_dir: str = "./experiments/runs/advanced_zs_experiments") -> str:
        """Save experiment results"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'advanced_zs_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        return results_file
    
    def print_results_summary(self):
        """Print a summary of all results"""
        print("\n" + "="*80)
        print("ADVANCED ZERO-SHOT LEARNING RESULTS SUMMARY")
        print("="*80)
        
        for exp_name, result in self.results.items():
            print(f"\n{exp_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if 'error' in result:
                print(f"Status: FAILED")
                print(f"Error: {result['error']}")
            elif 'evaluation_results' in result:
                print(f"Status: SUCCESS")
                print(f"Model: {result['model_name']}")
                
                eval_results = result['evaluation_results']
                for metric in ['ndcg', 'map', 'recall', 'precision']:
                    if metric in eval_results:
                        values = eval_results[metric]
                        print(f"{metric.upper()}: " + ", ".join([f"@{k}={v:.4f}" for k, v in values.items()]))
                
                if 'num_queries' in result:
                    print(f"Queries: {result['num_queries']}, Documents: {result['num_documents']}")
            else:
                print(f"Status: {result.get('status', 'COMPLETED')}")
                if 'note' in result:
                    print(f"Note: {result['note']}")
        
        print("\n" + "="*80)

def main():
    """Main function to run advanced zero-shot learning experiments"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize runner
        runner = AdvancedZSRunner()
        
        # Load data
        runner.load_data()
        
        # Run all experiments
        results = runner.run_all_advanced_experiments()
        
        # Save results
        results_file = runner.save_results()
        
        # Print summary
        runner.print_results_summary()
        
        print(f"\nAdvanced Zero-Shot Learning experiments completed!")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Advanced ZS experiments failed: {e}")
        print(f"Experiments failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()