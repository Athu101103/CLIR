"""
BM25 implementation for Query Translation (QT) framework.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import Dataset
import pickle
import os

from .base_qt import BaseQTModel
from src.utils.translation_utils import TranslationManager

logger = logging.getLogger(__name__)


class BM25QTModel(BaseQTModel):
    """BM25 model for Query Translation framework."""
    
    def __init__(self, config: Dict[str, Any], translation_config: Dict[str, Any]):
        """
        Initialize BM25 QT model.
        
        Args:
            config: BM25 configuration
            translation_config: Translation configuration
        """
        super().__init__(config)
        
        # BM25 parameters
        self.k1 = config.get('k1', 1.2)
        self.b = config.get('b', 0.75)
        
        # Translation manager
        self.translation_manager = TranslationManager(translation_config)
        
        # BM25 index
        self.bm25_index = None
        self.documents = None
        
        logger.info(f"Initialized BM25 QT model with k1={self.k1}, b={self.b}")
    
    def translate_queries(self, queries: List[str]) -> List[str]:
        """
        Translate English queries to Sanskrit using Google Translate.
        
        Args:
            queries: List of English queries
            
        Returns:
            List of Sanskrit queries
        """
        logger.info(f"Translating {len(queries)} queries from English to Sanskrit")
        
        # Use translation manager to translate queries
        translated_queries = self.translation_manager.translate_queries_to_sanskrit(queries)
        
        logger.info(f"Translation completed. Sample: '{queries[0][:50]}...' -> '{translated_queries[0][:50]}...'")
        
        return translated_queries
    
    def build_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of Sanskrit documents
        """
        logger.info(f"Building BM25 index for {len(documents)} documents")
        
        # Tokenize documents
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization for Sanskrit (split by whitespace)
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        self.documents = documents
        
        logger.info("BM25 index built successfully")
    
    def retrieve_documents(self, 
                          translated_queries: List[str], 
                          documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using BM25 with translated queries.
        
        Args:
            translated_queries: List of Sanskrit queries
            documents: List of Sanskrit documents
            k: Number of top documents to retrieve
            
        Returns:
            List of relevance scores for each query-document pair
        """
        if self.bm25_index is None:
            self.build_index(documents)
        
        all_scores = []
        
        for query in translated_queries:
            # Tokenize query
            query_tokens = query.split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Normalize scores to [0, 1] range
            if len(scores) > 0:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    scores = [0.5] * len(scores)  # Default score if all scores are equal
            
            all_scores.append(scores)
        
        return all_scores
    
    def train(self, train_data: Dataset, validation_data: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Train BM25 model (build index from training documents).
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Training BM25 QT model")
        
        # Extract documents from training data
        documents = [item['document'] for item in train_data]
        
        # Build BM25 index
        self.build_index(documents)
        
        # Mark as trained
        self.is_trained = True
        
        logger.info(f"BM25 QT model trained with {len(documents)} documents")
        
        return {
            'model_name': self.model_name,
            'documents_indexed': len(documents),
            'k1': self.k1,
            'b': self.b
        }
    
    def save_model(self, path: str) -> None:
        """
        Save BM25 model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'bm25_index': self.bm25_index,
            'documents': self.documents,
            'k1': self.k1,
            'b': self.b,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"BM25 QT model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load BM25 model.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.bm25_index = model_data['bm25_index']
        self.documents = model_data['documents']
        self.k1 = model_data['k1']
        self.b = model_data['b']
        self.config = model_data['config']
        self.is_trained = True
        
        logger.info(f"BM25 QT model loaded from {path}")
    
    def get_top_k_documents(self, query: str, k: int = 10) -> List[tuple]:
        """
        Get top-k documents for a query.
        
        Args:
            query: Sanskrit query
            k: Number of top documents
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25_index is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Tokenize query
        query_tokens = query.split()
        
        # Get scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return (document, score) pairs
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], scores[idx]))
        
        return results
