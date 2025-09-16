"""
Base class for Zero-shot (ZS) framework models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

class BaseZSModel(ABC):
    """Base class for Zero-shot models."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
    
    @abstractmethod
    def retrieve_documents(self, queries: List[str], documents: List[str], k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using zero-shot model (no fine-tuning).
        Args:
            queries: List of English queries
            documents: List of Sanskrit documents
            k: Number of top documents to retrieve
        Returns:
            List of relevance scores for each query-document pair
        """
        pass
    
    def evaluate(self, test_queries: List[str], test_documents: List[str], true_relevance: List[List[int]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the model.
        Args:
            test_queries: Test queries (English)
            test_documents: Test documents (Sanskrit)
            true_relevance: True relevance labels
            k_values: List of k values for evaluation
        Returns:
            Evaluation metrics
        """
        from src.evaluation.metrics import CLIREvaluator
        evaluator = CLIREvaluator(k_values=k_values)
        all_predictions = []
        for query in test_queries:
            scores = self.retrieve_documents([query], test_documents, k=max(k_values))
            all_predictions.append(scores[0])
        results = evaluator.evaluate_batch(all_predictions, true_relevance, k_values)
        return results
