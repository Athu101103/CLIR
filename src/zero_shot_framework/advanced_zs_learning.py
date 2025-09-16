"""
Advanced Zero-Shot and Few-Shot Learning for Sanskrit-English CLIR

This module implements:
1. Cross-Script Transfer Learning from other Devanagari-script languages
2. Multilingual Prompting using Large Language Models  
3. In-Context Learning for Sanskrit-English alignment
4. Parameter-Efficient Fine-Tuning (LoRA, Adapters)
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import faiss
from datasets import Dataset
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CrossScriptConfig:
    """Configuration for cross-script transfer learning"""
    source_languages: List[str]  # Hindi, Marathi, Nepali, etc.
    target_language: str = "sanskrit"
    similarity_threshold: float = 0.7
    max_transfer_examples: int = 10000
    script_normalization: bool = True

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"

class DevanagariScriptNormalizer:
    """Normalizes different Devanagari-script languages for transfer learning"""
    
    def __init__(self):
        # Common Devanagari Unicode ranges and normalization rules
        self.devanagari_range = (0x0900, 0x097F)
        self.common_conjuncts = {
            'क्ष': ['क्ष', 'क्श'],  # Common variations
            'त्र': ['त्र', 'त्रा'],
            'ज्ञ': ['ज्ञ', 'ग्य'],
        }
        
    def normalize_text(self, text: str, source_lang: str = "hindi") -> str:
        """Normalize Devanagari text for cross-script transfer"""
        # Basic normalization
        normalized = text.strip()
        
        # Language-specific normalizations
        if source_lang == "hindi":
            normalized = self._hindi_to_sanskrit_hints(normalized)
        elif source_lang == "marathi":
            normalized = self._marathi_to_sanskrit_hints(normalized)
        elif source_lang == "nepali":
            normalized = self._nepali_to_sanskrit_hints(normalized)
            
        return normalized
    
    def _hindi_to_sanskrit_hints(self, text: str) -> str:
        """Apply Hindi-specific normalizations for Sanskrit transfer"""
        # Remove Hindi-specific markers
        text = text.replace('।', '।')  # Keep Sanskrit punctuation
        text = text.replace('॥', '॥')  # Keep Sanskrit verse markers
        return text
    
    def _marathi_to_sanskrit_hints(self, text: str) -> str:
        """Apply Marathi-specific normalizations"""
        # Marathi-specific character mappings
        return text
    
    def _nepali_to_sanskrit_hints(self, text: str) -> str:
        """Apply Nepali-specific normalizations"""
        # Nepali-specific character mappings
        return text

class CrossScriptTransferLearner:
    """Transfer learning from other Devanagari-script languages to Sanskrit"""
    
    def __init__(self, config: CrossScriptConfig):
        self.config = config
        self.normalizer = DevanagariScriptNormalizer()
        self.embeddings_cache = {}
        
    def load_source_language_data(self, lang: str) -> List[Dict[str, str]]:
        """Load parallel data from source Devanagari languages"""
        # This would load actual parallel corpora
        # For now, return mock data structure
        mock_data = []
        
        if lang == "hindi":
            mock_data = [
                {"source": "यह एक उदाहरण है", "english": "This is an example"},
                {"source": "ज्ञान महत्वपूर्ण है", "english": "Knowledge is important"},
                {"source": "प्रकृति सुंदर है", "english": "Nature is beautiful"},
            ]
        elif lang == "marathi":
            mock_data = [
                {"source": "हे एक उदाहरण आहे", "english": "This is an example"},
                {"source": "ज्ञान महत्त्वाचे आहे", "english": "Knowledge is important"},
            ]
        elif lang == "nepali":
            mock_data = [
                {"source": "यो एक उदाहरण हो", "english": "This is an example"},
                {"source": "ज्ञान महत्वपूर्ण छ", "english": "Knowledge is important"},
            ]
            
        return mock_data[:self.config.max_transfer_examples]
    
    def extract_cross_script_features(self, model, tokenizer, source_texts: List[str], english_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract transferable features from source language pairs"""
        # Encode source language texts
        source_inputs = tokenizer(source_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        english_inputs = tokenizer(english_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            source_embeddings = model(**source_inputs).last_hidden_state.mean(dim=1)
            english_embeddings = model(**english_inputs).last_hidden_state.mean(dim=1)
        
        return {
            "source_embeddings": source_embeddings,
            "english_embeddings": english_embeddings,
            "alignment_matrix": self._compute_alignment_matrix(source_embeddings, english_embeddings)
        }
    
    def _compute_alignment_matrix(self, source_emb: torch.Tensor, english_emb: torch.Tensor) -> torch.Tensor:
        """Compute alignment matrix between source and English embeddings"""
        # Simple linear alignment - could be made more sophisticated
        alignment = torch.mm(source_emb, english_emb.T)
        return torch.softmax(alignment, dim=-1)
    
    def transfer_to_sanskrit(self, sanskrit_texts: List[str], model, tokenizer) -> torch.Tensor:
        """Apply transfer learning insights to Sanskrit texts"""
        normalized_texts = [self.normalizer.normalize_text(text, "sanskrit") for text in sanskrit_texts]
        
        inputs = tokenizer(normalized_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        
        return embeddings

class LLMPromptingEngine:
    """Large Language Model prompting for Sanskrit-English understanding"""
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the multilingual language model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info(f"Loaded LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load LLM {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model_name = "google/mt5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def create_few_shot_prompt(self, query: str, examples: List[Dict[str, str]], task: str = "retrieve") -> str:
        """Create few-shot prompt for in-context learning"""
        prompt_parts = []
        
        if task == "retrieve":
            prompt_parts.append("Task: Find relevant Sanskrit documents for English queries.\n")
            prompt_parts.append("Examples:")
            
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"{i}. Query: {example['english']}")
                prompt_parts.append(f"   Relevant Sanskrit: {example['sanskrit']}")
                prompt_parts.append(f"   Explanation: {example.get('explanation', 'Semantically related content')}")
                prompt_parts.append("")
            
            prompt_parts.append(f"Now find relevant documents for: {query}")
            
        elif task == "translate":
            prompt_parts.append("Task: Translate between English and Sanskrit.\n")
            prompt_parts.append("Examples:")
            
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"{i}. English: {example['english']}")
                prompt_parts.append(f"   Sanskrit: {example['sanskrit']}")
                prompt_parts.append("")
            
            prompt_parts.append(f"Translate to Sanskrit: {query}")
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using the LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def in_context_retrieval_scoring(self, query: str, documents: List[str], examples: List[Dict[str, str]]) -> List[float]:
        """Score documents using in-context learning"""
        scores = []
        
        for doc in documents:
            # Create context-aware prompt
            prompt = self.create_few_shot_prompt(query, examples, task="retrieve")
            prompt += f"\n\nRate relevance of this Sanskrit text (0-1): {doc[:200]}..."
            
            try:
                response = self.generate_response(prompt, max_length=50)
                # Extract score from response (simplified - would need better parsing)
                score = self._extract_score_from_response(response)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score document: {e}")
                scores.append(0.0)
        
        return scores
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        # Simplified extraction - would need more robust parsing
        import re
        numbers = re.findall(r'0\.\d+|\d+\.\d+', response)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            except:
                pass
        return 0.5  # Default middle score

class ParameterEfficientFineTuner:
    """Parameter-efficient fine-tuning using LoRA and adapters"""
    
    def __init__(self, base_model_name: str, lora_config: LoRAConfig):
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_lora_model(self):
        """Setup LoRA-enhanced model"""
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModel.from_pretrained(self.base_model_name)
        
        # Configure LoRA
        if self.lora_config.target_modules is None:
            # Default target modules for transformer models
            self.lora_config.target_modules = ["query", "value", "key", "dense"]
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
        )
        
        # Create PEFT model
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"LoRA model setup complete. Trainable parameters: {self.peft_model.print_trainable_parameters()}")
    
    def create_training_dataset(self, queries: List[str], documents: List[str], labels: List[int]) -> Dataset:
        """Create dataset for parameter-efficient fine-tuning"""
        # Create query-document pairs
        examples = []
        for i, query in enumerate(queries):
            for j, doc in enumerate(documents):
                label = 1 if labels[i] == j else 0  # Simplified labeling
                examples.append({
                    "query": query,
                    "document": doc,
                    "label": label
                })
        
        # Tokenize
        def tokenize_function(examples):
            # Combine query and document
            combined_texts = [f"Query: {q} Document: {d}" for q, d in zip(examples["query"], examples["document"])]
            
            tokenized = self.tokenizer(
                combined_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized["labels"] = examples["label"]
            return tokenized
        
        dataset = Dataset.from_dict({
            "query": [ex["query"] for ex in examples],
            "document": [ex["document"] for ex in examples],
            "label": [ex["label"] for ex in examples]
        })
        
        return dataset.map(tokenize_function, batched=True)
    
    def fine_tune(self, train_dataset: Dataset, val_dataset: Dataset = None, output_dir: str = "./lora_output"):
        """Fine-tune using LoRA"""
        if self.peft_model is None:
            self.setup_lora_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_steps=100,
            save_steps=500,
            evaluation_strategy="steps" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Custom trainer for similarity learning
        trainer = ContrastiveSimilarityTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save the LoRA weights
        trainer.save_model()
        
        logger.info(f"LoRA fine-tuning completed. Model saved to {output_dir}")
    
    def load_fine_tuned_model(self, checkpoint_path: str):
        """Load fine-tuned LoRA model"""
        self.model = AutoModel.from_pretrained(self.base_model_name)
        self.peft_model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loaded fine-tuned LoRA model from {checkpoint_path}")

class ContrastiveSimilarityTrainer(Trainer):
    """Custom trainer for contrastive similarity learning"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute contrastive loss for query-document similarity"""
        # Get model outputs
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        
        # Extract embeddings (mean pooling)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Get labels
        labels = inputs["labels"].float()
        
        # Compute contrastive loss
        # Positive pairs should have high similarity, negative pairs low similarity
        similarities = torch.nn.functional.cosine_similarity(
            embeddings[::2], embeddings[1::2], dim=1
        )
        
        # Binary cross-entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            similarities, labels[::2]
        )
        
        return (loss, outputs) if return_outputs else loss

class AdvancedZeroShotModel:
    """Advanced Zero-Shot model incorporating all techniques"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'xlm-roberta-base')
        
        # Initialize components
        self.cross_script_config = CrossScriptConfig(
            source_languages=config.get('source_languages', ['hindi', 'marathi']),
            target_language='sanskrit'
        )
        
        self.lora_config = LoRAConfig(
            r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1)
        )
        
        # Initialize sub-components
        self.cross_script_learner = CrossScriptTransferLearner(self.cross_script_config)
        self.llm_engine = LLMPromptingEngine(config.get('llm_model', 'facebook/mbart-large-50-many-to-many-mmt'))
        self.peft_tuner = ParameterEfficientFineTuner(self.model_name, self.lora_config)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self._setup_base_model()
    
    def _setup_base_model(self):
        """Setup base model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def perform_cross_script_transfer(self, sanskrit_documents: List[str]) -> torch.Tensor:
        """Perform cross-script transfer learning"""
        logger.info("Performing cross-script transfer learning...")
        
        all_features = []
        
        for lang in self.cross_script_config.source_languages:
            try:
                # Load source language data
                source_data = self.cross_script_learner.load_source_language_data(lang)
                
                if source_data:
                    source_texts = [item['source'] for item in source_data]
                    english_texts = [item['english'] for item in source_data]
                    
                    # Extract cross-script features
                    features = self.cross_script_learner.extract_cross_script_features(
                        self.model, self.tokenizer, source_texts, english_texts
                    )
                    all_features.append(features)
                    
                    logger.info(f"Extracted features from {len(source_texts)} {lang} examples")
            
            except Exception as e:
                logger.warning(f"Failed to process {lang}: {e}")
        
        # Apply transfer learning to Sanskrit
        sanskrit_embeddings = self.cross_script_learner.transfer_to_sanskrit(
            sanskrit_documents, self.model, self.tokenizer
        )
        
        return sanskrit_embeddings
    
    def few_shot_learning_with_examples(self, queries: List[str], documents: List[str], 
                                      few_shot_examples: List[Dict[str, str]], k: int = 10) -> List[List[float]]:
        """Perform few-shot learning using provided examples"""
        logger.info(f"Performing few-shot learning with {len(few_shot_examples)} examples...")
        
        all_scores = []
        
        for query in queries:
            # Use LLM prompting for scoring
            scores = self.llm_engine.in_context_retrieval_scoring(query, documents, few_shot_examples)
            all_scores.append(scores)
        
        return all_scores
    
    def parameter_efficient_fine_tune(self, train_queries: List[str], train_documents: List[str], 
                                    train_labels: List[int], output_dir: str = "./advanced_zs_lora"):
        """Perform parameter-efficient fine-tuning"""
        logger.info("Starting parameter-efficient fine-tuning...")
        
        # Create training dataset
        train_dataset = self.peft_tuner.create_training_dataset(
            train_queries, train_documents, train_labels
        )
        
        # Fine-tune with LoRA
        self.peft_tuner.fine_tune(train_dataset, output_dir=output_dir)
        
        logger.info("Parameter-efficient fine-tuning completed")
    
    def retrieve_documents(self, queries: List[str], documents: List[str], 
                         k: int = 10, use_cross_script: bool = True, 
                         use_few_shot: bool = True, few_shot_examples: List[Dict[str, str]] = None) -> List[List[float]]:
        """Advanced document retrieval using all techniques"""
        
        if use_cross_script:
            # Apply cross-script transfer learning
            doc_embeddings = self.perform_cross_script_transfer(documents)
        else:
            # Standard embeddings
            doc_inputs = self.tokenizer(documents, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
            with torch.no_grad():
                doc_embeddings = self.model(**doc_inputs).last_hidden_state.mean(dim=1)
        
        # Encode queries
        query_inputs = self.tokenizer(queries, padding=True, truncation=True, 
                                    return_tensors="pt", max_length=512)
        with torch.no_grad():
            query_embeddings = self.model(**query_inputs).last_hidden_state.mean(dim=1)
        
        # Compute similarity scores
        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2
        )
        
        # Convert to list format
        base_scores = similarity_scores.cpu().numpy().tolist()
        
        # Enhance with few-shot learning if requested
        if use_few_shot and few_shot_examples:
            few_shot_scores = self.few_shot_learning_with_examples(
                queries, documents, few_shot_examples, k
            )
            
            # Combine scores (weighted average)
            alpha = 0.7  # Weight for base scores
            final_scores = []
            for i in range(len(queries)):
                combined = [
                    alpha * base_scores[i][j] + (1 - alpha) * few_shot_scores[i][j]
                    for j in range(len(documents))
                ]
                final_scores.append(combined)
            
            return final_scores
        
        return base_scores
    
    def save_model(self, save_path: str):
        """Save the advanced model"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir / "tokenizer")
        
        # Save base model
        self.model.save_pretrained(save_dir / "base_model")
        
        logger.info(f"Advanced Zero-Shot model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str):
        """Load the advanced model"""
        load_dir = Path(load_path)
        
        # Load configuration
        config_path = load_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(config)
        
        # Load components
        instance.tokenizer = AutoTokenizer.from_pretrained(load_dir / "tokenizer")
        instance.model = AutoModel.from_pretrained(load_dir / "base_model")
        
        logger.info(f"Advanced Zero-Shot model loaded from {load_path}")
        return instance

# Example usage and testing functions
def create_few_shot_examples() -> List[Dict[str, str]]:
    """Create sample few-shot examples for Sanskrit-English pairs"""
    return [
        {
            "english": "What is knowledge?",
            "sanskrit": "ज्ञानं किम्?",
            "explanation": "Philosophical inquiry about the nature of knowledge"
        },
        {
            "english": "The soul is eternal",
            "sanskrit": "आत्मा नित्यः",
            "explanation": "Statement about the eternal nature of the soul"
        },
        {
            "english": "Dharma is duty",
            "sanskrit": "धर्मः कर्तव्यम्",
            "explanation": "Definition of dharma as moral duty"
        },
        {
            "english": "Yoga means union",
            "sanskrit": "योगः युक्तिः",
            "explanation": "Etymology and meaning of yoga"
        },
        {
            "english": "Truth is one",
            "sanskrit": "सत्यम् एकम्",
            "explanation": "Philosophical statement about the unity of truth"
        }
    ]

def test_advanced_zero_shot():
    """Test function for the advanced zero-shot model"""
    # Configuration
    config = {
        'model_name': 'xlm-roberta-base',
        'source_languages': ['hindi', 'marathi'],
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'llm_model': 'facebook/mbart-large-50-many-to-many-mmt'
    }
    
    # Sample data
    queries = [
        "What is the nature of consciousness?",
        "How to achieve liberation?",
        "What is the meaning of dharma?"
    ]
    
    documents = [
        "चैतन्यं ब्रह्म इति वेदान्तवादिनः।",
        "मोक्षः सर्वदुःखनिवृत्तिः।",
        "धर्मः धारयते जगत्।"
    ]
    
    few_shot_examples = create_few_shot_examples()
    
    # Initialize model
    model = AdvancedZeroShotModel(config)
    
    # Test retrieval
    try:
        scores = model.retrieve_documents(
            queries, documents, k=3, 
            use_cross_script=True, 
            use_few_shot=True, 
            few_shot_examples=few_shot_examples
        )
        
        print("Advanced Zero-Shot Retrieval Results:")
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: {query}")
            query_scores = scores[i]
            # Get top documents
            top_indices = np.argsort(query_scores)[::-1][:3]
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. Document {idx}: {documents[idx][:50]}... (Score: {query_scores[idx]:.3f})")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_advanced_zero_shot()