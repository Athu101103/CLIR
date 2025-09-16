"""
Data loader utility for Anveshana CLIR dataset.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class AnveshanaDataLoader:
    """Load and preprocess Anveshana dataset."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_name = config.get('name', 'manojbalaji1/anveshana')
        self.train_split = config.get('train_split', 0.9)
        self.test_size = config.get('test_size', 340)
        self.negative_sampling_ratio = config.get('negative_sampling_ratio', 2)
        
        # Preprocessing configs
        self.sanskrit_config = config.get('sanskrit', {})
        self.english_config = config.get('english', {})
        
        self.dataset = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
    def load_dataset(self) -> None:
        """Load the Anveshana dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset['train'])}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_sanskrit_text(self, text: str) -> str:
        """
        Preprocess Sanskrit text according to paper specifications.
        
        Args:
            text: Raw Sanskrit text
            
        Returns:
            Preprocessed Sanskrit text
        """
        if not text:
            return text
            
        # Handle poetic structures (replace "——1.1.3——" with "——")
        if self.sanskrit_config.get('handle_poetic_structures', True):
            text = re.sub(r'——\d+\.\d+\.\d+——', '——', text)
        
        # Remove non-Devanagari characters if specified
        if self.sanskrit_config.get('remove_non_devanagari', True):
            # Keep Devanagari characters, spaces, and basic punctuation
            devanagari_pattern = r'[^\u0900-\u097F\s——।॥]'
            text = re.sub(devanagari_pattern, '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_english_text(self, text: str) -> str:
        """
        Preprocess English text with minimal processing.
        
        Args:
            text: Raw English text
            
        Returns:
            Preprocessed English text
        """
        if not text:
            return text
            
        if self.english_config.get('minimal_preprocessing', True):
            # Only basic cleaning
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_dataset(self) -> None:
        """Preprocess the entire dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Preprocessing dataset...")
        
        # Preprocess train split
        train_data = self.dataset['train'].map(
            lambda x: {
                'query': self.preprocess_english_text(x['query']),
                'document': self.preprocess_sanskrit_text(x['document']),
                'relevance': x.get('relevance', 1)  # Default to relevant
            }
        )
        
        # Preprocess test split
        test_data = self.dataset['test'].map(
            lambda x: {
                'query': self.preprocess_english_text(x['query']),
                'document': self.preprocess_sanskrit_text(x['document']),
                'relevance': x.get('relevance', 1)
            }
        )
        
        self.train_data = train_data
        self.test_data = test_data
        
        logger.info(f"Preprocessing completed. Train: {len(self.train_data)}, Test: {len(self.test_data)}")
    
    def create_splits(self) -> None:
        """Create train/validation splits from the training data."""
        if self.train_data is None:
            raise ValueError("Training data not available. Call preprocess_dataset() first.")
        
        # Calculate split sizes
        total_train = len(self.train_data)
        validation_size = int(total_train * (1 - self.train_split))
        train_size = total_train - validation_size
        
        # Create splits - fix the indexing issue
        train_indices = list(range(train_size))
        validation_indices = list(range(train_size, total_train))
        
        # Create new datasets with correct indices
        original_train_data = self.train_data
        self.train_data = original_train_data.select(train_indices)
        self.validation_data = original_train_data.select(validation_indices)
        
        logger.info(f"Created splits - Train: {len(self.train_data)}, Validation: {len(self.validation_data)}")
    
    def create_negative_samples(self, data: Dataset) -> Dataset:
        """
        Create negative samples for training.
        
        Args:
            data: Original dataset with positive samples
            
        Returns:
            Dataset with negative samples added
        """
        logger.info("Creating negative samples...")
        
        # Get all documents
        all_documents = [item['document'] for item in data]
        all_queries = [item['query'] for item in data]
        
        # Create BM25 index for negative sampling
        tokenized_docs = [doc.split() for doc in all_documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        negative_samples = []
        
        for i, query_item in enumerate(data):
            query = query_item['query']
            positive_doc = query_item['document']
            
            # Get BM25 scores for all documents
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)
            
            # Find documents with high scores but not the positive one
            candidate_indices = np.argsort(scores)[::-1]
            
            # Filter out the positive document
            negative_candidates = [idx for idx in candidate_indices if all_documents[idx] != positive_doc]
            
            # Create negative samples
            num_negative = self.negative_sampling_ratio
            for j in range(min(num_negative, len(negative_candidates))):
                neg_idx = negative_candidates[j]
                negative_samples.append({
                    'query': query,
                    'document': all_documents[neg_idx],
                    'relevance': 0  # Negative sample
                })
        
        # Combine positive and negative samples
        positive_data = [dict(item) for item in data]
        combined_data = positive_data + negative_samples
        
        logger.info(f"Created {len(negative_samples)} negative samples. Total: {len(combined_data)}")
        
        return Dataset.from_list(combined_data)
    
    def get_train_data(self) -> Dataset:
        """Get training data with negative samples."""
        if self.train_data is None:
            raise ValueError("Training data not available. Call preprocess_dataset() and create_splits() first.")
        
        return self.create_negative_samples(self.train_data)
    
    def get_validation_data(self) -> Dataset:
        """Get validation data."""
        if self.validation_data is None:
            raise ValueError("Validation data not available. Call create_splits() first.")
        
        return self.validation_data
    
    def get_test_data(self) -> Dataset:
        """Get test data."""
        if self.test_data is None:
            raise ValueError("Test data not available. Call preprocess_dataset() first.")
        
        return self.test_data
    
    def get_all_documents(self) -> List[str]:
        """Get all unique documents from the dataset."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dataset not processed. Call preprocess_dataset() first.")
        
        train_docs = set(item['document'] for item in self.train_data)
        test_docs = set(item['document'] for item in self.test_data)
        
        return list(train_docs.union(test_docs))
    
    def get_document_to_id_mapping(self) -> Dict[str, int]:
        """Create mapping from document text to document ID."""
        documents = self.get_all_documents()
        return {doc: idx for idx, doc in enumerate(documents)} 