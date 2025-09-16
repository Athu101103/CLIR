"""
Base class for Query Translation (QT) framework models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


class BaseQTModel(ABC):
    """Base class for Query Translation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base QT model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
        self.is_trained = False
    
    @abstractmethod
    def translate_queries(self, queries: List[str]) -> List[str]:
        """
        Translate English queries to Sanskrit.
        
        Args:
            queries: List of English queries
            
        Returns:
            List of Sanskrit queries
        """
        pass
    
    @abstractmethod
    def retrieve_documents(self, 
                          translated_queries: List[str], 
                          documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using translated queries.
        
        Args:
            translated_queries: List of Sanskrit queries
            documents: List of Sanskrit documents
            k: Number of top documents to retrieve
            
        Returns:
            List of relevance scores for each query-document pair
        """
        pass
    
    @abstractmethod
    def train(self, train_data: Dataset, validation_data: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            
        Returns:
            Training metrics
        """
        pass
    
    def evaluate(self, 
                test_queries: List[str], 
                test_documents: List[str], 
                true_relevance: List[List[int]], 
                k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the model.
        
        Args:
            test_queries: Test queries
            test_documents: Test documents
            true_relevance: True relevance labels
            k_values: List of k values for evaluation
            
        Returns:
            Evaluation metrics
        """
        from src.evaluation.metrics import CLIREvaluator
        
        evaluator = CLIREvaluator(k_values=k_values)
        
        # Translate queries
        translated_queries = self.translate_queries(test_queries)
        
        # Get predictions for all queries
        all_predictions = []
        for query in translated_queries:
            scores = self.retrieve_documents([query], test_documents, k=max(k_values))
            all_predictions.append(scores[0])
        
        # Evaluate
        results = evaluator.evaluate_batch(all_predictions, true_relevance, k_values)
        
        return results
    
    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        logger.info(f"Saving model to {path}")
        # Implementation depends on the specific model
    
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        logger.info(f"Loading model from {path}")
        # Implementation depends on the specific model
