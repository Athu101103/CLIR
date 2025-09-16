"""
BM25 implementation for Document Translation (DT) framework.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import Dataset
import pickle
import os

from .base_dt import BaseDTModel
from src.utils.translation_utils import TranslationManager

logger = logging.getLogger(__name__)


class BM25DTModel(BaseDTModel):
    """BM25 model for Document Translation framework."""
    
    def __init__(self, config: Dict[str, Any], translation_config: Dict[str, Any]):
        """
        Initialize BM25 DT model.
        
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
        
        logger.info(f"Initialized BM25 DT model with k1={self.k1}, b={self.b}")
    
    def translate_documents(self, documents: List[str]) -> List[str]:
        """
        Translate Sanskrit documents to English using Google Translate.
        
        Args:
            documents: List of Sanskrit documents
            
        Returns:
            List of English documents
        """
        logger.info(f"Translating {len(documents)} documents from Sanskrit to English")
        
        # Use translation manager to translate documents
        translated_documents = self.translation_manager.translate_documents_to_english(documents)
        
        logger.info(f"Translation completed. Sample: '{documents[0][:50]}...' -> '{translated_documents[0][:50]}...'")
        
        return translated_documents
    
    def build_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from English documents.
        
        Args:
            documents: List of English documents
        """
        logger.info(f"Building BM25 index for {len(documents)} documents")
        
        # Tokenize documents
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization for English (split by whitespace)
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        self.documents = documents
        
        logger.info("BM25 index built successfully")
    
    def retrieve_documents(self, 
                          queries: List[str], 
                          translated_documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using BM25 with English queries and translated documents.
        
        Args:
            queries: List of English queries
            translated_documents: List of English documents (translated from Sanskrit)
            k: Number of top documents to retrieve
            
        Returns:
            List of relevance scores for each query-document pair
        """
        if self.bm25_index is None:
            self.build_index(translated_documents)
        
        all_scores = []
        
        for query in queries:
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
        logger.info("Training BM25 DT model")
        
        # Extract documents from training data
        documents = [item['document'] for item in train_data]
        
        # Translate documents to English
        translated_documents = self.translate_documents(documents)
        
        # Build BM25 index
        self.build_index(translated_documents)
        
        # Mark as trained
        self.is_trained = True
        
        logger.info(f"BM25 DT model trained with {len(translated_documents)} documents")
        
        return {
            'model_name': self.model_name,
            'documents_indexed': len(translated_documents),
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
        
        logger.info(f"BM25 DT model saved to {path}")
    
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
        
        logger.info(f"BM25 DT model loaded from {path}")
    
    def get_top_k_documents(self, query: str, documents: List[str], k: int = 10) -> List[tuple]:
        """
        Get top-k documents for a query.
        
        Args:
            query: English query
            documents: List of English documents
            k: Number of top documents
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25_index is None:
            self.build_index(documents)
        
        # Tokenize query
        query_tokens = query.split()
        
        # Get scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return (document, score) pairs
        results = []
        for idx in top_indices:
            results.append((documents[idx], scores[idx]))
        
        return results
