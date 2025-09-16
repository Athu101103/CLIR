"""
Contriever implementation for Document Translation (DT) framework.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import Dataset
import pickle
import os
import math

from .base_dt import BaseDTModel
from src.utils.translation_utils import TranslationManager

logger = logging.getLogger(__name__)


class EnglishDataset(TorchDataset):
    """Dataset for English retrieval training."""
    
    def __init__(self, data: Dataset, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: Contriever tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query and document
        query_tokens = self.tokenizer(
            item['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        doc_tokens = self.tokenizer(
            item['document'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'doc_input_ids': doc_tokens['input_ids'].squeeze(0),
            'doc_attention_mask': doc_tokens['attention_mask'].squeeze(0),
            'relevance': torch.tensor(item['relevance'], dtype=torch.float)
        }


class ContrieverDTModel(BaseDTModel):
    """Contriever model for Document Translation framework."""
    
    def __init__(self, config: Dict[str, Any], translation_config: Dict[str, Any]):
        """
        Initialize Contriever DT model.
        
        Args:
            config: Contriever configuration
            translation_config: Translation configuration
        """
        super().__init__(config)
        
        # Model parameters
        self.model_name = config.get('model_name', 'mjwong/contriever-mnli')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.epochs = config.get('epochs', 3)
        self.training_method = config.get('training_method', 'DOT')  # 'DOT' or 'CONCAT'
        
        # Translation manager
        self.translation_manager = TranslationManager(translation_config)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized Contriever DT model: {self.model_name}")
        logger.info(f"Training method: {self.training_method}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self):
        """Load Contriever model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading Contriever model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("Contriever model loaded successfully")
    
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
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tensor of embeddings
        """
        self._load_model()
        self.model.eval()
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                
                # Normalize by embedding length as per paper
                embedding = embedding / math.sqrt(embedding.size(0))
                
                embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def _calculate_similarity(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity between query and document embeddings.
        
        Args:
            query_embeddings: Query embeddings
            doc_embeddings: Document embeddings
            
        Returns:
            Similarity scores
        """
        # Dot product as per paper specification
        similarity = torch.matmul(query_embeddings, doc_embeddings.T)
        
        # Apply sigmoid to get scores between 0 and 1
        scores = torch.sigmoid(similarity)
        
        return scores
    
    def retrieve_documents(self, 
                          queries: List[str], 
                          translated_documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using Contriever with English queries and translated documents.
        
        Args:
            queries: List of English queries
            translated_documents: List of English documents (translated from Sanskrit)
            k: Number of top documents to retrieve
            
        Returns:
            List of relevance scores for each query-document pair
        """
        logger.info(f"Retrieving documents for {len(queries)} queries")
        
        # Get embeddings
        query_embeddings = self._get_embeddings(queries)
        doc_embeddings = self._get_embeddings(translated_documents)
        
        # Calculate similarities
        similarity_scores = self._calculate_similarity(query_embeddings, doc_embeddings)
        
        # Convert to list of lists
        all_scores = []
        for i in range(len(queries)):
            scores = similarity_scores[i].cpu().numpy().tolist()
            all_scores.append(scores)
        
        return all_scores
    
    def train(self, train_data: Dataset, validation_data: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Train Contriever model for English retrieval.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Training Contriever DT model")
        
        self._load_model()
        
        # Translate documents in training data
        translated_train_data = []
        for item in train_data:
            translated_doc = self.translate_documents([item['document']])[0]
            translated_train_data.append({
                'query': item['query'],
                'document': translated_doc,
                'relevance': item['relevance']
            })
        
        # Create datasets
        train_dataset = EnglishDataset(Dataset.from_list(translated_train_data), self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if validation_data:
            translated_val_data = []
            for item in validation_data:
                translated_doc = self.translate_documents([item['document']])[0]
                translated_val_data.append({
                    'query': item['query'],
                    'document': translated_doc,
                    'relevance': item['relevance']
                })
            val_dataset = EnglishDataset(Dataset.from_list(translated_val_data), self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.model.train()
        training_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                doc_input_ids = batch['doc_input_ids'].to(self.device)
                doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                relevance = batch['relevance'].to(self.device)
                
                # Forward pass
                query_outputs = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
                doc_outputs = self.model(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
                
                # Get [CLS] embeddings
                query_embeddings = query_outputs.last_hidden_state[:, 0, :]
                doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]
                
                # Normalize by embedding length
                query_embeddings = query_embeddings / math.sqrt(query_embeddings.size(1))
                doc_embeddings = doc_embeddings / math.sqrt(doc_embeddings.size(1))
                
                # Calculate similarity (dot product)
                similarity = torch.sum(query_embeddings * doc_embeddings, dim=1)
                
                # Calculate loss
                loss = criterion(similarity, relevance)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            training_losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # Mark as trained
        self.is_trained = True
        
        logger.info("Contriever DT model training completed")
        
        return {
            'model_name': self.model_name,
            'training_method': self.training_method,
            'epochs': self.epochs,
            'final_loss': training_losses[-1],
            'training_losses': training_losses
        }
    
    def save_model(self, path: str) -> None:
        """
        Save Contriever model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and tokenizer
        model_path = os.path.join(path, 'model')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save additional info
        info_path = os.path.join(path, 'info.pkl')
        info = {
            'config': self.config,
            'model_name': self.model_name,
            'training_method': self.training_method,
            'max_length': self.max_length
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        
        logger.info(f"Contriever DT model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load Contriever model.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load model and tokenizer
        model_path = os.path.join(path, 'model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load additional info
        info_path = os.path.join(path, 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            self.config = info['config']
            self.model_name = info['model_name']
            self.training_method = info['training_method']
            self.max_length = info['max_length']
        
        self.is_trained = True
        
        logger.info(f"Contriever DT model loaded from {path}")
    
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
        # Get embeddings
        query_embedding = self._get_embeddings([query])
        doc_embeddings = self._get_embeddings(documents)
        
        # Calculate similarities
        similarity_scores = self._calculate_similarity(query_embedding, doc_embeddings)
        
        # Get top-k
        scores = similarity_scores[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return (document, score) pairs
        results = []
        for idx in top_indices:
            results.append((documents[idx], scores[idx]))
        
        return results
