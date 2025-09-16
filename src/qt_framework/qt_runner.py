"""
Main runner for Query Translation (QT) framework experiments.
"""
import logging
import os
import json
from typing import Dict, Any, List
from datetime import datetime

from .bm25_qt import BM25QTModel
from .xlm_roberta_qt import XLMQTModel
from src.utils.config_loader import ConfigLoader
from src.utils.data_loader import AnveshanaDataLoader
from src.evaluation.metrics import CLIREvaluator

logger = logging.getLogger(__name__)


class QTRunner:
    """Main runner for QT framework experiments."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize QT runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_full_config()
        
        # Get specific configs
        self.qt_config = self.config_loader.get_model_config('qt')
        self.dataset_config = self.config_loader.get_dataset_config()
        self.translation_config = self.config_loader.get_translation_config()
        self.evaluation_config = self.config_loader.get_evaluation_config()
        self.logging_config = self.config_loader.get_logging_config()
        
        # Initialize data loader
        self.data_loader = AnveshanaDataLoader(self.dataset_config)
        
        # Results storage
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
        logger.info("QT Runner initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.logging_config.get('log_level', 'INFO'))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self):
        """Load and preprocess dataset."""
        logger.info("Loading and preprocessing dataset...")
        
        # Load dataset
        self.data_loader.load_dataset()
        
        # Preprocess
        self.data_loader.preprocess_dataset()
        
        # Create splits
        self.data_loader.create_splits()
        
        # Get data
        self.train_data = self.data_loader.get_train_data()
        self.validation_data = self.data_loader.get_validation_data()
        self.test_data = self.data_loader.get_test_data()
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  - Train: {len(self.train_data)} samples")
        logger.info(f"  - Validation: {len(self.validation_data)} samples")
        logger.info(f"  - Test: {len(self.test_data)} samples")
    
    def run_bm25_experiment(self) -> Dict[str, Any]:
        """Run BM25 experiment."""
        logger.info("Starting BM25 QT experiment")
        
        # Initialize model
        bm25_config = self.qt_config.get('bm25', {})
        model = BM25QTModel(bm25_config, self.translation_config)
        
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
            # Find the index of the relevant document
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                # If document not found, mark first as relevant (fallback)
                relevance[0] = 1
            true_relevance.append(relevance)
        
        # Evaluate
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        
        # Save results
        self.results['bm25_qt'] = {
            'model_name': 'BM25 QT',
            'train_metrics': train_metrics,
            'evaluation_results': results
        }
        
        logger.info("BM25 QT experiment completed")
        return self.results['bm25_qt']
    
    def run_xlm_roberta_experiment(self) -> Dict[str, Any]:
        """Run XLM-RoBERTa experiment."""
        logger.info("Starting XLM-RoBERTa QT experiment")
        
        # Initialize model
        xlm_config = self.qt_config.get('xlm_roberta', {})
        model = XLMQTModel(xlm_config, self.translation_config)
        
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
        self.results['xlm_roberta_qt'] = {
            'model_name': 'XLM-RoBERTa QT',
            'train_metrics': train_metrics,
            'evaluation_results': results
        }
        
        logger.info("XLM-RoBERTa QT experiment completed")
        return self.results['xlm_roberta_qt']
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all QT experiments."""
        logger.info("Starting all QT experiments")
        
        # Load data first
        self.load_data()
        
        # Run experiments
        bm25_results = self.run_bm25_experiment()
        xlm_results = self.run_xlm_roberta_experiment()
        
        # Create summary
        summary = self._create_summary()
        
        # Save results
        self._save_results()
        
        logger.info("All QT experiments completed")
        return summary
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary of all results."""
        summary = {
            'framework': 'Query Translation (QT)',
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, results in self.results.items():
            summary['models'][model_name] = {
                'model_name': results['model_name'],
                'evaluation_results': results['evaluation_results']
            }
        
        return summary
    
    def _save_results(self):
        """Save results to file."""
        results_dir = self.logging_config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(results_dir, 'qt_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(results_dir, 'qt_summary.json')
        summary = self._create_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_dir}")
    
    def print_results(self):
        """Print results in a readable format."""
        print("\n" + "="*60)
        print("QUERY TRANSLATION (QT) FRAMEWORK RESULTS")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{results['model_name']}:")
            print("-" * 40)
            
            eval_results = results['evaluation_results']
            for metric in ['ndcg', 'map', 'recall', 'precision']:
                print(f"{metric.upper()}:", end=" ")
                for k in [1, 3, 5, 10]:
                    value = eval_results[metric][k]
                    print(f"@{k}={value:.4f}", end=" ")
                print()
        
        print("\n" + "="*60)


def main():
    """Main function to run QT experiments."""
    runner = QTRunner()
    summary = runner.run_all_experiments()
    runner.print_results()
    
    return summary


if __name__ == "__main__":
    main()
