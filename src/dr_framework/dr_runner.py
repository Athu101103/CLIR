"""
DR Runner for orchestrating Direct Retrieve experiments.
"""
import logging
import time
from typing import Dict, Any, Optional
import os

from src.utils.config_loader import ConfigLoader
from src.utils.data_loader import AnveshanaDataLoader
from .mdpr_dr import MDPRDRModel
from .xlm_roberta_dr import XLMRoBERTaDRModel
from .multilingual_e5_dr import MultilingualE5DRModel

logger = logging.getLogger(__name__)


class DRRunner:
    """Runner for Direct Retrieve (DR) framework experiments."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize DR runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_full_config()
        
        # Get DR-specific configurations
        self.dr_config = self.config['models']['dr']
        self.evaluation_config = self.config['evaluation']
        
        # Data
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
        # Results storage
        self.results = {}
        
        logger.info("DR Runner initialized")
    
    def load_data(self) -> None:
        """Load and preprocess the dataset."""
        logger.info("Loading and preprocessing dataset...")
        
        # Initialize data loader
        data_loader = AnveshanaDataLoader(self.config['dataset'])
        
        # Load and preprocess data
        data_loader.load_dataset()
        data_loader.preprocess_dataset()
        data_loader.create_splits()
        data_loader.create_negative_samples(data_loader.train_data)
        
        # Store data
        self.train_data = data_loader.train_data
        self.validation_data = data_loader.validation_data
        self.test_data = data_loader.test_data
        
        logger.info("Data loaded successfully:")
        logger.info(f"  - Train: {len(self.train_data)} samples")
        logger.info(f"  - Validation: {len(self.validation_data)} samples")
        logger.info(f"  - Test: {len(self.test_data)} samples")
    
    def run_mdpr_experiment(self) -> Dict[str, Any]:
        """
        Run mDPR DR experiment.
        
        Returns:
            Experiment results
        """
        logger.info("Starting mDPR DR experiment")
        
        # Initialize model
        mdpr_config = self.dr_config['mdpr']
        model = MDPRDRModel(mdpr_config)
        
        # Train model
        train_metrics = model.train(self.train_data, self.validation_data)
        
        # Prepare test data
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        
        # Get all unique documents for evaluation
        all_documents = list(set(test_documents))
        
        # Create true relevance matrix
        true_relevance = []
        for i, item in enumerate(self.test_data):
            relevance = [0] * len(all_documents)
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        
        # Evaluate
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        
        # Save results
        self.results['mdpr_dr'] = {
            'model_name': 'mDPR DR',
            'train_metrics': train_metrics,
            'evaluation_results': results
        }
        
        logger.info("mDPR DR experiment completed")
        
        return self.results['mdpr_dr']
    
    def run_xlm_roberta_experiment(self) -> Dict[str, Any]:
        """
        Run XLM-RoBERTa DR experiment.
        
        Returns:
            Experiment results
        """
        logger.info("Starting XLM-RoBERTa DR experiment")
        
        # Initialize model
        xlm_config = self.dr_config['xlm_roberta']
        model = XLMRoBERTaDRModel(xlm_config)
        
        # Train model
        train_metrics = model.train(self.train_data, self.validation_data)
        
        # Prepare test data
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        
        # Get all unique documents for evaluation
        all_documents = list(set(test_documents))
        
        # Create true relevance matrix
        true_relevance = []
        for i, item in enumerate(self.test_data):
            relevance = [0] * len(all_documents)
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        
        # Evaluate
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        
        # Save results
        self.results['xlm_roberta_dr'] = {
            'model_name': 'XLM-RoBERTa DR',
            'train_metrics': train_metrics,
            'evaluation_results': results
        }
        
        logger.info("XLM-RoBERTa DR experiment completed")
        
        return self.results['xlm_roberta_dr']
    
    def run_e5_experiment(self) -> Dict[str, Any]:
        """Run Multilingual-E5 DR experiment."""
        logger.info("Starting Multilingual-E5 DR experiment")
        e5_config = self.dr_config['multilingual_e5']
        model = MultilingualE5DRModel(e5_config)
        train_metrics = model.train(self.train_data, self.validation_data)
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        all_documents = list(set(test_documents))
        true_relevance = []
        for item in self.test_data:
            relevance = [0] * len(all_documents)
            try:
                idx = all_documents.index(item['document'])
                relevance[idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        self.results['e5_dr'] = {
            'model_name': 'Multilingual-E5 DR',
            'train_metrics': train_metrics,
            'evaluation_results': results,
        }
        logger.info("Multilingual-E5 DR experiment completed")
        return self.results['e5_dr']

    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all DR experiments.
        
        Returns:
            All experiment results
        """
        logger.info("Running all DR experiments")
        
        # Run mDPR experiment
        self.run_mdpr_experiment()
        
        # Run XLM-RoBERTa experiment
        self.run_xlm_roberta_experiment()
        
        # Run Multilingual-E5 experiment
        self.run_e5_experiment()
        
        # TODO: Add other models (Multilingual-E5, GPT-2)
        
        logger.info("All DR experiments completed")
        
        return self.results
    
    def save_results(self, output_dir: str = 'results') -> None:
        """
        Save experiment results.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to file
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'dr_results_{timestamp}.json')
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'model_name': result['model_name'],
                'train_metrics': result['train_metrics'],
                'evaluation_results': result['evaluation_results']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def print_results(self) -> None:
        """Print experiment results."""
        print("\n" + "="*80)
        print("DIRECT RETRIEVE (DR) EXPERIMENT RESULTS")
        print("="*80)
        
        for model_name, result in self.results.items():
            print(f"\n{result['model_name']} Results:")
            print("-" * 40)
            
            eval_results = result['evaluation_results']
            for metric in ['ndcg', 'map', 'recall', 'precision']:
                print(f"{metric.upper()}:", end=" ")
                for k in [1, 3, 5, 10]:
                    value = eval_results[metric][k]
                    print(f"@{k}={value:.4f}", end=" ")
                print()
            
            print(f"\nTraining Info:")
            train_metrics = result['train_metrics']
            for key, value in train_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*80)
    
    def compare_with_paper(self) -> None:
        """Compare results with paper results."""
        print(f"\nComparison with Paper Results (Table 3):")
        print("-" * 50)
        
        paper_results = {
            'mDPR DR': {'ndcg@1': 0.0295, 'ndcg@10': 0.0295},
            'XLM-RoBERTa DR': {'ndcg@1': 0.0295, 'ndcg@10': 0.0295},
            'Multilingual-E5 DR': {'ndcg@1': 0.0295, 'ndcg@10': 0.0295},
            'GPT-2 DR': {'ndcg@1': 0.0295, 'ndcg@10': 0.0295}
        }
        
        for model_name, result in self.results.items():
            if model_name in paper_results:
                paper_ndcg1 = paper_results[model_name]['ndcg@1']
                our_ndcg1 = result['evaluation_results']['ndcg'][1]
                
                print(f"{result['model_name']}:")
                print(f"  Paper NDCG@1: {paper_ndcg1:.4f} ({paper_ndcg1*100:.2f}%)")
                print(f"  Our NDCG@1:   {our_ndcg1:.4f} ({our_ndcg1*100:.2f}%)")
                print(f"  Difference:   {our_ndcg1 - paper_ndcg1:+.4f}")
                print()
