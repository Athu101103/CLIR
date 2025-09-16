"""
Base class for Document Translation (DT) framework models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


class BaseDTModel(ABC):
    """Base class for Document Translation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base DT model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
        self.is_trained = False
    
    @abstractmethod
    def translate_documents(self, documents: List[str]) -> List[str]:
        """
        Translate Sanskrit documents to English.
        
        Args:
            documents: List of Sanskrit documents
            
        Returns:
            List of English documents
        """
        pass
    
    @abstractmethod
    def retrieve_documents(self, 
                          queries: List[str], 
                          translated_documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using English queries and translated documents.
        
        Args:
            queries: List of English queries
            translated_documents: List of English documents (translated from Sanskrit)
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
            test_queries: Test queries (English)
            test_documents: Test documents (Sanskrit)
            true_relevance: True relevance labels
            k_values: List of k values for evaluation
            
        Returns:
            Evaluation metrics
        """
        from src.evaluation.metrics import CLIREvaluator
        
        evaluator = CLIREvaluator(k_values=k_values)
        
        # Translate documents
        translated_documents = self.translate_documents(test_documents)
        
        # Get predictions for all queries
        all_predictions = []
        for query in test_queries:
            scores = self.retrieve_documents([query], translated_documents, k=max(k_values))
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
