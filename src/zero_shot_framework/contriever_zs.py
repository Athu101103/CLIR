"""
Zero-shot Contriever model for CLIR.
"""
import logging
from typing import List, Dict, Any
import torch
import numpy as np
import math
from transformers import AutoModel, AutoTokenizer
from .base_zs import BaseZSModel

logger = logging.getLogger(__name__)

class ContrieverZSModel(BaseZSModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'mjwong/contriever-mnli')
        self.max_length = config.get('max_length', 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        logger.info(f"Loaded Contriever zero-shot model: {self.model_name}")
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                embedding = embedding / math.sqrt(embedding.size(0))
                embeddings.append(embedding)
        return torch.stack(embeddings)
    def retrieve_documents(self, queries: List[str], documents: List[str], k: int = 10) -> List[List[float]]:
        logger.info(f"Zero-shot Contriever: retrieving for {len(queries)} queries vs {len(documents)} docs")
        query_emb = self._get_embeddings(queries)
        doc_emb = self._get_embeddings(documents)
        similarity = torch.matmul(query_emb, doc_emb.T)
        scores = torch.sigmoid(similarity)
        all_scores = [scores[i].cpu().numpy().tolist() for i in range(len(queries))]
        return all_scores
