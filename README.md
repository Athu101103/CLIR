# Anveshana CLIR Implementation

This repository implements the Cross-Lingual Information Retrieval (CLIR) system described in the research paper "Anveshana: A New Benchmark Dataset for Cross-Lingual Information Retrieval On English Queries and Sanskrit Documents".

## Project Overview

The implementation follows the paper's methodology with three main frameworks:
- **Query Translation (QT)**: Translate English queries to Sanskrit for monolingual retrieval
- **Document Translation (DT)**: Translate Sanskrit documents to English for monolingual retrieval  
- **Direct Retrieve (DR)**: Use shared embedding spaces without translation
- **Zero-shot**: Evaluate pre-trained models without fine-tuning

## Current Progress

### ✅ Completed (Step 1)
- [x] Project structure and directory setup
- [x] Configuration management system
- [x] Data loading and preprocessing utilities
- [x] Translation utilities (Google Translate API integration)
- [x] Evaluation metrics (NDCG, MAP, Recall, Precision)
- [x] Basic test suite

### 🔄 In Progress
- [ ] Query Translation (QT) framework implementation
- [ ] Document Translation (DT) framework implementation
- [ ] Direct Retrieve (DR) framework implementation
- [ ] Zero-shot framework implementation

### 📋 Planned
- [ ] Model training and fine-tuning
- [ ] Comprehensive evaluation
- [ ] Results visualization
- [ ] Performance optimization

## Project Structure

```
anveshana-clir/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── test_setup.py              # Setup verification script
├── README.md                  # This file
├── src/                       # Source code
│   ├── utils/                 # Utility modules
│   │   ├── config_loader.py   # Configuration management
│   │   ├── data_loader.py     # Dataset loading and preprocessing
│   │   └── translation_utils.py # Translation utilities
│   ├── evaluation/            # Evaluation modules
│   │   └── metrics.py         # CLIR evaluation metrics
│   ├── qt_framework/          # Query Translation framework
│   ├── dt_framework/          # Document Translation framework
│   ├── dr_framework/          # Direct Retrieve framework
│   └── zero_shot_framework/   # Zero-shot framework
├── data/                      # Data directory
│   ├── raw/                   # Raw dataset
│   ├── processed/             # Preprocessed data
│   ├── translations/          # Translated documents/queries
│   └── embeddings/            # Generated embeddings
├── experiments/               # Experiment management
│   ├── configs/               # Experiment configurations
│   ├── runs/                  # Experiment results
│   └── logs/                  # Experiment logs
└── results/                   # Final results and visualizations
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python test_setup.py
```

### 3. Configuration
Edit `config.yaml` to customize:
- Model parameters
- Training settings
- Evaluation metrics
- Translation settings

## Key Features Implemented

### Data Loading and Preprocessing
- **Sanskrit Text Processing**: Handles poetic structures and Devanagari character cleaning
- **English Text Processing**: Minimal preprocessing to preserve intent
- **Negative Sampling**: 2:1 ratio of negative to positive samples using BM25
- **Dataset Splits**: Train/validation/test splits as per paper specifications

### Translation System
- **Google Translate Integration**: Batch translation with caching
- **Language Detection**: Automatic language identification
- **Error Handling**: Retry mechanisms and fallback options

### Evaluation Metrics
- **NDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision  
- **Recall**: Proportion of relevant documents retrieved
- **Precision**: Proportion of retrieved documents that are relevant
- **Multi-k Evaluation**: k=1, 3, 5, 10 as specified in the paper

## Models to be Implemented

### Query Translation (QT)
- BM25 for Sanskrit retrieval
- XLM-RoBERTa-base fine-tuned for Sanskrit

### Document Translation (DT)
- BM25 for English retrieval
- Contriever (mjwong/contriever-mnli) with CONCAT/DOT methods
- ColBERT fine-tuned with DOT method
- GPT-2 for retrieval
- REPLUG LSR with Mistral-7B supervision

### Direct Retrieve (DR)
- mDPR (Multi-Dimensional Passage Retrieval)
- XLM-RoBERTa-base with DOT/CONCAT configurations
- Multilingual-E5-base
- GPT-2 with multiple approaches

### Zero-shot
- ColBERT (monolingual-Contriever eng)
- XLM-RoBERTa-base
- Contriever (monolingual-ColBERT eng)
- Multilingual-E5-base

## Expected Results

Based on the paper, we expect to achieve:
- **DT Framework**: Best performance with BM25 achieving ~62.46% NDCG@10
- **Zero-shot**: Good performance with monolingual-Contriever achieving ~30.48% NDCG@10
- **QT Framework**: Lower performance due to Sanskrit translation challenges
- **DR Framework**: Moderate performance due to cross-lingual alignment challenges

## Next Steps

1. **Implement QT Framework**: Start with BM25 and XLM-RoBERTa models
2. **Implement DT Framework**: Focus on high-performing models like BM25 and Contriever
3. **Implement DR Framework**: Work on shared embedding space approaches
4. **Implement Zero-shot**: Evaluate pre-trained models
5. **Comprehensive Evaluation**: Compare all frameworks and models
6. **Results Analysis**: Generate tables and visualizations matching the paper

## Contributing

This is a step-by-step implementation. Each framework will be implemented and tested before moving to the next. Please evaluate the code at each step and provide feedback.

## License

This implementation follows the research paper's methodology and uses the Anveshana dataset which is publicly available at: https://huggingface.co/datasets/manojbalaji1/anveshana 