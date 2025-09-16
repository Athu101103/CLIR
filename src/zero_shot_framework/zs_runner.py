"""
ZS Runner for orchestrating Zero-shot experiments.
"""
import logging
import os
from typing import Dict, Any

# Disable TensorFlow/Flax to avoid DLL issues
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.utils.config_loader import ConfigLoader
from src.utils.data_loader import AnveshanaDataLoader
from .xlm_roberta_zs import XLMRoBERTaZSModel
from .contriever_zs import ContrieverZSModel
from .e5_zs import MultilingualE5ZSModel

# Optional ColBERT import
try:
    import torch
    _ENABLE_COLBERT = (torch.cuda.is_available() and os.getenv('ENABLE_COLBERT') == '1')
    if _ENABLE_COLBERT:
        from .colbert_zs import ColBERTZSModel
    else:
        ColBERTZSModel = None
except Exception:
    ColBERTZSModel = None

logger = logging.getLogger(__name__)

class ZSRunner:
    """Runner for Zero-shot (ZS) framework experiments."""
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_full_config()
        self.zs_config = self.config['models']['zero_shot']
        self.evaluation_config = self.config['evaluation']
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.results = {}
        logger.info("ZS Runner initialized")
    def load_data(self) -> None:
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
    def run_xlm_roberta_experiment(self) -> Dict[str, Any]:
        logger.info("Running Zero-shot XLM-RoBERTa experiment")
        config = self.zs_config['xlm_roberta']
        model = XLMRoBERTaZSModel(config)
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        all_documents = list(set(test_documents))
        true_relevance = []
        for item in self.test_data:
            relevance = [0] * len(all_documents)
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        self.results['xlm_roberta_zs'] = {
            'model_name': 'XLM-RoBERTa ZS',
            'evaluation_results': results,
            'info': {'model': config['model_name']}
        }
        logger.info("Zero-shot XLM-RoBERTa experiment completed")
        return self.results['xlm_roberta_zs']
    def run_contriever_experiment(self) -> Dict[str, Any]:
        logger.info("Running Zero-shot Contriever experiment")
        config = self.zs_config['contriever']
        model = ContrieverZSModel(config)
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        all_documents = list(set(test_documents))
        true_relevance = []
        for item in self.test_data:
            relevance = [0] * len(all_documents)
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        self.results['contriever_zs'] = {
            'model_name': 'Contriever ZS',
            'evaluation_results': results,
            'info': {'model': config['model_name']}
        }
        logger.info("Zero-shot Contriever experiment completed")
        return self.results['contriever_zs']
    def run_colbert_experiment(self) -> Dict[str, Any]:
        if ColBERTZSModel is None:
            logger.warning("ColBERT not available (requires CUDA and ENABLE_COLBERT=1)")
            return {}
        logger.info("Running Zero-shot ColBERT experiment")
        config = self.zs_config['colbert']
        model = ColBERTZSModel(config)
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]
        all_documents = list(set(test_documents))
        true_relevance = []
        for item in self.test_data:
            relevance = [0] * len(all_documents)
            try:
                relevant_idx = all_documents.index(item['document'])
                relevance[relevant_idx] = 1
            except ValueError:
                relevance[0] = 1
            true_relevance.append(relevance)
        k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        results = model.evaluate(test_queries, all_documents, true_relevance, k_values)
        self.results['colbert_zs'] = {
            'model_name': 'ColBERT ZS',
            'evaluation_results': results,
            'info': {'model': config['model_name']}
        }
        logger.info("Zero-shot ColBERT experiment completed")
        return self.results['colbert_zs']
    def run_e5_experiment(self) -> Dict[str, Any]:
        logger.info("Running Zero-shot Multilingual-E5 experiment")
        config = self.zs_config['multilingual_e5']
        model = MultilingualE5ZSModel(config)
        test_queries = [item['query'] for item in self.test_data]
        test_documents = [item['document'] for item in self.test_data]

        # Get all unique documents for evaluation
        all_documents = list(set(test_documents))

        # Create true relevance matrix
        true_relevance = []
        for item in self.test_data:
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
        self.results['e5_zs'] = {
            'model_name': 'Multilingual-E5 ZS',
            'evaluation_results': results,
            'info': {'model': config['model_name']}
        }

        logger.info("Zero-shot Multilingual-E5 experiment completed")

        return self.results['e5_zs']
    def run_all_experiments(self) -> Dict[str, Any]:
        logger.info("Running all Zero-shot experiments")
        self.run_xlm_roberta_experiment()
        self.run_contriever_experiment()
        # Run E5 before ColBERT to ensure tests complete even if ColBERT weights are unavailable
        self.run_e5_experiment()
        try:
            self.run_colbert_experiment()
        except Exception as e:
            logger.warning(f"ColBERT ZS skipped due to error: {e}")
        logger.info("All Zero-shot experiments completed")
        return self.results
    def print_results(self) -> None:
        print("\n" + "="*80)
        print("ZERO-SHOT (ZS) EXPERIMENT RESULTS")
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
            print(f"\nInfo:")
            for key, value in result.get('info', {}).items():
                print(f"  {key}: {value}")
        print("\n" + "="*80)
