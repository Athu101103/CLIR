"""
Zero-shot ColBERT model for CLIR.
"""
import logging
from typing import List, Dict, Any
import torch
import numpy as np
from colbert.modeling.colbert import ColBERT
from colbert.modeling.checkpoint import Checkpoint
from .base_zs import BaseZSModel

logger = logging.getLogger(__name__)

class ColBERTZSModel(BaseZSModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'colbert-ir/colbertv2.0')
        self.max_length = config.get('max_length', 128)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.colbert = Checkpoint(self.model_name).colbert.to(self.device)
        logger.info(f"Loaded ColBERT zero-shot model: {self.model_name}")
    def _get_colbert_scores(self, queries: List[str], documents: List[str]) -> List[List[float]]:
        # Use ColBERT's late interaction scoring
        self.colbert.eval()
        with torch.no_grad():
            # Tokenize and encode queries and documents
            Q = self.colbert.query_tokenizer.batch_encode_plus(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            D = self.colbert.doc_tokenizer.batch_encode_plus(documents, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            Q = {k: v.to(self.device) for k, v in Q.items()}
            D = {k: v.to(self.device) for k, v in D.items()}
            # Get ColBERT embeddings
            Q_emb = self.colbert.query(*Q.values())
            D_emb = self.colbert.doc(*D.values())
            # Compute late interaction scores
            scores = torch.einsum('qmd,knd->qknm', Q_emb, D_emb).max(-1).values.sum(-1)
            # Normalize scores to [0,1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            return scores.cpu().numpy().tolist()
    def retrieve_documents(self, queries: List[str], documents: List[str], k: int = 10) -> List[List[float]]:
        logger.info(f"Zero-shot ColBERT: retrieving for {len(queries)} queries vs {len(documents)} docs")
        scores = self._get_colbert_scores(queries, documents)
        return scores
