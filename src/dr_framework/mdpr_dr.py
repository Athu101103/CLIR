"""
mDPR (Multi-Dimensional Passage Retrieval) implementation for Direct Retrieve (DR) framework.
"""
import logging
from typing import List, Dict, Any, Optional
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

from .base_dr import BaseDRModel

logger = logging.getLogger(__name__)


class CrossLingualDataset(TorchDataset):
    """Dataset for cross-lingual retrieval training."""
    
    def __init__(self, data: Dataset, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: mDPR tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize English query and Sanskrit document
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


class MDPRDRModel(BaseDRModel):
    """mDPR model for Direct Retrieve framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mDPR DR model.
        
        Args:
            config: mDPR configuration
        """
        super().__init__(config)
        
        # Model parameters
        self.model_name = config.get('model_name', 'facebook/dpr-question_encoder-multiset-base')
        self.max_length = int(config.get('max_length', 512))
        self.batch_size = int(config.get('batch_size', 16))
        self.learning_rate = float(config.get('learning_rate', 2e-5))
        self.epochs = int(config.get('epochs', 3))
        self.temperature = float(config.get('temperature', 0.1))  # Temperature for contrastive learning
        
        # Model components
        self.tokenizer = None
        self.query_encoder = None
        self.doc_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized mDPR DR model: {self.model_name}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self):
        """Load mDPR model and tokenizer."""
        if self.tokenizer is None:
            logger.info(f"Loading mDPR model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load separate encoders for query and document
            self.query_encoder = AutoModel.from_pretrained(self.model_name)
            self.doc_encoder = AutoModel.from_pretrained(self.model_name)
            
            self.query_encoder.to(self.device)
            self.doc_encoder.to(self.device)
            
            logger.info("mDPR model loaded successfully")
    
    def _get_embeddings(self, texts: List[str], encoder_type: str = 'query') -> torch.Tensor:
        """
        Get embeddings for texts using specified encoder.
        
        Args:
            texts: List of texts
            encoder_type: 'query' or 'document'
            
        Returns:
            Tensor of embeddings
        """
        self._load_model()
        
        if encoder_type == 'query':
            encoder = self.query_encoder
        else:
            encoder = self.doc_encoder
        
        encoder.eval()
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
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                
                # Normalize embeddings
                embedding = nn.functional.normalize(embedding, p=2, dim=0)
                
                embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def _calculate_similarity(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate cosine similarity between query and document embeddings.
        
        Args:
            query_embeddings: Query embeddings
            doc_embeddings: Document embeddings
            
        Returns:
            Similarity scores
        """
        # Cosine similarity
        similarity = torch.matmul(query_embeddings, doc_embeddings.T)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Apply sigmoid to get scores between 0 and 1
        scores = torch.sigmoid(similarity)
        
        return scores
    
    def retrieve_documents(self, 
                          queries: List[str], 
                          documents: List[str], 
                          k: int = 10) -> List[List[float]]:
        """
        Retrieve documents using mDPR with direct cross-lingual matching.
        
        Args:
            queries: List of English queries
            documents: List of Sanskrit documents
            k: Number of top documents to retrieve
            
        Returns:
            List of relevance scores for each query-document pair
        """
        logger.info(f"Retrieving documents for {len(queries)} queries")
        
        # Get embeddings
        query_embeddings = self._get_embeddings(queries, 'query')
        doc_embeddings = self._get_embeddings(documents, 'document')
        
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
        Train mDPR model for cross-lingual retrieval.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Training mDPR DR model")
        
        self._load_model()
        
        # Create datasets
        train_dataset = CrossLingualDataset(train_data, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if validation_data:
            val_dataset = CrossLingualDataset(validation_data, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW([
            {'params': self.query_encoder.parameters()},
            {'params': self.doc_encoder.parameters()}
        ], lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Loss function (contrastive learning)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.query_encoder.train()
        self.doc_encoder.train()
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
                query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
                doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
                
                # Get [CLS] embeddings
                query_embeddings = query_outputs.last_hidden_state[:, 0, :]
                doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]
                
                # Normalize embeddings
                query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=1)
                doc_embeddings = nn.functional.normalize(doc_embeddings, p=2, dim=1)
                
                # Calculate similarity matrix
                similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature
                
                # Create labels for contrastive learning
                labels = torch.arange(similarity_matrix.size(0)).to(self.device)
                
                # Calculate loss
                loss = criterion(similarity_matrix, labels)
                
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
        
        logger.info("mDPR DR model training completed")
        
        return {
            'model_name': self.model_name,
            'epochs': self.epochs,
            'final_loss': training_losses[-1],
            'training_losses': training_losses,
            'temperature': self.temperature
        }
    
    def save_model(self, path: str) -> None:
        """
        Save mDPR model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save encoders and tokenizer
        query_path = os.path.join(path, 'query_encoder')
        doc_path = os.path.join(path, 'doc_encoder')
        
        self.query_encoder.save_pretrained(query_path)
        self.doc_encoder.save_pretrained(doc_path)
        self.tokenizer.save_pretrained(path)
        
        # Save additional info
        info_path = os.path.join(path, 'info.pkl')
        info = {
            'config': self.config,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'temperature': self.temperature
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        
        logger.info(f"mDPR DR model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load mDPR model.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load tokenizer and encoders
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        query_path = os.path.join(path, 'query_encoder')
        doc_path = os.path.join(path, 'doc_encoder')
        
        self.query_encoder = AutoModel.from_pretrained(query_path)
        self.doc_encoder = AutoModel.from_pretrained(doc_path)
        
        self.query_encoder.to(self.device)
        self.doc_encoder.to(self.device)
        
        # Load additional info
        info_path = os.path.join(path, 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            self.config = info['config']
            self.model_name = info['model_name']
            self.max_length = info['max_length']
            self.temperature = info['temperature']
        
        self.is_trained = True
        
        logger.info(f"mDPR DR model loaded from {path}")
    
    def get_top_k_documents(self, query: str, documents: List[str], k: int = 10) -> List[tuple]:
        """
        Get top-k documents for a query.
        
        Args:
            query: English query
            documents: List of Sanskrit documents
            k: Number of top documents
            
        Returns:
            List of (document, score) tuples
        """
        # Get embeddings
        query_embedding = self._get_embeddings([query], 'query')
        doc_embeddings = self._get_embeddings(documents, 'document')
        
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
