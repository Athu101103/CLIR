"""
Multilingual-E5 implementation for Direct Retrieve (DR) framework.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import Dataset
import pickle
import os

from .base_dr import BaseDRModel

logger = logging.getLogger(__name__)


class CrossLingualE5Dataset(TorchDataset):
    def __init__(self, data: Dataset, tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # E5 recommends prompts
        query_text = f"query: {item['query']}"
        doc_text = f"passage: {item['document']}"
        q = self.tokenizer(query_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        d = self.tokenizer(doc_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'query_input_ids': q['input_ids'].squeeze(0),
            'query_attention_mask': q['attention_mask'].squeeze(0),
            'doc_input_ids': d['input_ids'].squeeze(0),
            'doc_attention_mask': d['attention_mask'].squeeze(0),
            'relevance': torch.tensor(item['relevance'], dtype=torch.float)
        }


class MultilingualE5DRModel(BaseDRModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'intfloat/multilingual-e5-base')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.epochs = config.get('epochs', 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        logger.info(f"Initialized Multilingual-E5 DR model: {self.model_name}")
        logger.info(f"Device: {self.device}")

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            logger.info("Multilingual-E5 model loaded")

    def _embed_texts(self, texts: List[str], is_query: bool) -> torch.Tensor:
        self._load_model()
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for t in texts:
                prefix = 'query: ' if is_query else 'passage: '
                tokens = self.tokenizer(prefix + t, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                emb = out.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(emb)
        return torch.stack(embeddings)

    def retrieve_documents(self, queries: List[str], documents: List[str], k: int = 10) -> List[List[float]]:
        logger.info(f"Retrieving with Multilingual-E5 for {len(queries)} queries")
        q_emb = self._embed_texts(queries, is_query=True)
        d_emb = self._embed_texts(documents, is_query=False)
        sim = torch.matmul(q_emb, d_emb.T)
        scores = torch.sigmoid(sim)
        return [scores[i].cpu().numpy().tolist() for i in range(len(queries))]

    def train(self, train_data: Dataset, validation_data: Optional[Dataset] = None) -> Dict[str, Any]:
        logger.info("Training Multilingual-E5 DR model")
        self._load_model()
        train_ds = CrossLingualE5Dataset(train_data, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                qi = batch['query_input_ids'].to(self.device)
                qa = batch['query_attention_mask'].to(self.device)
                di = batch['doc_input_ids'].to(self.device)
                da = batch['doc_attention_mask'].to(self.device)
                rel = batch['relevance'].to(self.device)
                q_out = self.model(input_ids=qi, attention_mask=qa)
                d_out = self.model(input_ids=di, attention_mask=da)
                q_emb = q_out.last_hidden_state[:, 0, :]
                d_emb = d_out.last_hidden_state[:, 0, :]
                sim = torch.sum(q_emb * d_emb, dim=1)
                loss = criterion(sim, rel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            avg = epoch_loss / max(1, len(train_loader))
            losses.append(avg)
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg:.4f}")
        self.is_trained = True
        return {
            'model_name': self.model_name,
            'epochs': self.epochs,
            'final_loss': float(losses[-1]) if losses else 0.0,
            'training_losses': losses,
        }

    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Multilingual-E5 DR model saved to {path}")

    def load_model(self, path: str) -> None:
        model_path = os.path.join(path, 'model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.is_trained = True
        logger.info(f"Multilingual-E5 DR model loaded from {path}")
