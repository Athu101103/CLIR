# Advanced Zero-Shot Learning for Sanskrit-English CLIR

This repository implements the advanced zero-shot learning framework described in our research paper "Advanced Zero-Shot Learning for Sanskrit-English Cross-Lingual Information Retrieval". Our novel approach combines cross-script transfer learning, multilingual prompting, and parameter-efficient fine-tuning to achieve state-of-the-art performance on Sanskrit-English CLIR tasks.

## Research Paper

📄 **Paper**: "Advanced Zero-Shot Learning for Sanskrit-English Cross-Lingual Information Retrieval"  
🏛️ **Institution**: PSG College of Technology, Department of Applied Mathematics and Computational Sciences  
📊 **Results**: 80.2% improvement over baseline methods, NDCG@10 of 0.5847  

## Project Overview

Our framework addresses the unique challenges of Sanskrit-English CLIR through three innovative components:

1. **Cross-Script Transfer Learning**: Leverages linguistic similarities between Sanskrit and related Devanagari languages (Hindi, Marathi, Nepali)
2. **Multilingual Prompting**: Uses large language models with carefully crafted few-shot examples for contextual understanding
3. **Parameter-Efficient Fine-Tuning**: Employs LoRA (Low-Rank Adaptation) to reduce computational overhead by 80%

### Key Innovations
- **Cross-script normalization** for Devanagari languages
- **Semantic alignment learning** across multiple source languages  
- **Few-shot prompting** with philosophical Sanskrit-English examples
- **Contrastive learning** with temperature-scaled loss
- **Multi-component score fusion** for optimal retrieval

## Performance Results

Our advanced zero-shot framework achieves significant improvements across all metrics:

| Method | NDCG@10 | MAP@10 | Recall@10 | Precision@10 |
|--------|---------|--------|-----------|--------------|
| Baseline (XLM-RoBERTa) | 0.3245 | 0.2876 | 0.4521 | 0.1834 |
| Cross-Script Transfer | 0.4799 | 0.3226 | 0.6234 | 0.2145 |
| Few-Shot Learning | 0.5063 | 0.3608 | 0.5987 | 0.2298 |
| **Combined Techniques** | **0.5847** | **0.4123** | **0.7234** | **0.2567** |

### Key Achievements
- ✅ **80.2% improvement** over baseline methods
- ✅ **State-of-the-art performance** on Anveshana dataset
- ✅ **80% reduction** in computational overhead
- ✅ **99.2% parameter reduction** through LoRA
- ✅ **Robust cross-lingual understanding** for philosophical concepts

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

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Athu101103/CLIR.git
cd CLIR

# Create virtual environment
python -m venv anveshana_env
source anveshana_env/bin/activate  # On Windows: anveshana_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_setup.py
```

### 3. Quick Test Run
```bash
# Test the advanced zero-shot framework
python quick_qt_test.py
```

## How to Run the Project

### Option 1: Run Individual Frameworks

#### Zero-Shot Framework (Recommended)
```bash
# Run advanced zero-shot experiments
python examples/advanced_zs_example.py

# Run full zero-shot experiments with all models
python run_full_advanced_experiments.py
```

#### Query Translation Framework
```bash
# Run QT experiments
python examples/unified_query_runner.py --framework qt --model bm25
python examples/unified_query_runner.py --framework qt --model xlm_roberta
```

#### Document Translation Framework  
```bash
# Run DT experiments
python examples/unified_query_runner.py --framework dt --model bm25
python examples/unified_query_runner.py --framework dt --model contriever
```

#### Direct Retrieve Framework
```bash
# Run DR experiments
python examples/unified_query_runner.py --framework dr --model mdpr
python examples/unified_query_runner.py --framework dr --model xlm_roberta
```

### Option 2: Run All Experiments
```bash
# Run comprehensive experiments across all frameworks
python experiments/run_all_experiments.py
```

### Option 3: Custom Configuration
```bash
# Edit config.yaml for custom settings
# Then run with specific parameters
python examples/unified_query_runner.py --config custom_config.yaml
```

## Expected Outputs

### 1. Experiment Results
Running experiments generates detailed results in `experiments/runs/`:
```
experiments/runs/zs_experiments/advanced_zs_results_YYYYMMDD_HHMMSS.json
```

### 2. Performance Metrics
Each run outputs comprehensive metrics:
- **NDCG@k** (k=1,3,5,10): Ranking quality assessment
- **MAP@k**: Mean Average Precision across relevant documents  
- **Recall@k**: Proportion of relevant documents retrieved
- **Precision@k**: Proportion of retrieved documents that are relevant

### 3. Model Comparisons
Results include comparisons across:
- Baseline XLM-RoBERTa
- Cross-script transfer learning
- Few-shot learning with LLMs
- Combined advanced techniques

### 4. Sample Output Format
```json
{
  "experiment_info": {
    "timestamp": "20250820_235706",
    "model": "advanced_zs_learning",
    "dataset": "anveshana"
  },
  "results": {
    "ndcg_at_1": 0.2567,
    "ndcg_at_3": 0.4234,
    "ndcg_at_5": 0.5123,
    "ndcg_at_10": 0.5847,
    "map_at_10": 0.4123,
    "recall_at_10": 0.7234,
    "precision_at_10": 0.2567
  }
}
```

## Key Technical Features

### Advanced Zero-Shot Learning
- **Cross-Script Transfer**: Devanagari script normalization and semantic alignment
- **Multilingual Prompting**: Few-shot learning with philosophical examples
- **Parameter-Efficient Fine-Tuning**: LoRA with 99.2% parameter reduction

### Data Processing
- **Sanskrit Text Processing**: Handles complex morphological structures and poetic forms
- **Cross-Lingual Alignment**: Multi-source transfer from Hindi, Marathi, Nepali
- **Negative Sampling**: Intelligent sampling using BM25 for training

### Model Architecture
- **Base Model**: XLM-RoBERTa-base (278M parameters)
- **LLM Integration**: mBART-large-50 for multilingual prompting
- **Optimization**: Contrastive learning with temperature scaling (τ=0.07)

## Implemented Models & Frameworks

### Advanced Zero-Shot Learning (Primary Focus)
- **Cross-Script Transfer**: Multi-source alignment from Hindi/Marathi/Nepali
- **Few-Shot Prompting**: Contextual learning with philosophical examples
- **LoRA Fine-Tuning**: Parameter-efficient adaptation (rank=16, α=32)
- **Score Fusion**: Weighted combination of multiple similarity measures

### Query Translation (QT) Framework
- **BM25**: Traditional term-based retrieval for Sanskrit
- **XLM-RoBERTa**: Cross-lingual transformer fine-tuned for Sanskrit queries

### Document Translation (DT) Framework  
- **BM25**: English retrieval on translated documents
- **Contriever**: Dense retrieval with CONCAT/DOT similarity
- **ColBERT**: Late interaction model with DOT scoring

### Direct Retrieve (DR) Framework
- **mDPR**: Multi-lingual Dense Passage Retrieval
- **XLM-RoBERTa**: Cross-lingual embeddings with DOT/CONCAT
- **Multilingual-E5**: Multilingual sentence embeddings

### Zero-Shot Baselines
- **ColBERT**: Monolingual English model evaluation
- **Contriever**: English-trained dense retrieval
- **XLM-RoBERTa**: Pre-trained multilingual understanding

## Configuration

### Main Configuration (`config.yaml`)
```yaml
# Model settings
model:
  base_model: "xlm-roberta-base"
  lora_rank: 16
  lora_alpha: 32
  temperature: 0.07

# Training parameters  
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500

# Cross-script transfer
transfer:
  source_languages: ["hindi", "marathi", "nepali"]
  alignment_weight: 0.3
  
# Few-shot learning
few_shot:
  num_examples: 5
  example_selection: "diverse"
```

### Advanced Configuration Options
- **Cross-Script Settings**: Configure source languages and alignment weights
- **LoRA Parameters**: Adjust rank, alpha, and target modules
- **Few-Shot Examples**: Customize philosophical example selection
- **Evaluation Metrics**: Set k values and metric preferences

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.yaml
   training:
     batch_size: 8  # or smaller
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade transformers torch datasets
   ```

3. **Data Loading Issues**
   ```bash
   # Verify dataset access
   python -c "from datasets import load_dataset; print(load_dataset('manojbalaji1/anveshana'))"
   ```

4. **Model Download Issues**
   ```bash
   # Pre-download models
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('xlm-roberta-base')"
   ```

## Research Impact & Applications

### Digital Humanities
- **Cultural Preservation**: Democratizes access to Sanskrit philosophical texts
- **Cross-Cultural Understanding**: Bridges ancient wisdom with modern accessibility
- **Educational Tools**: Enables Sanskrit learning through English queries

### Technical Contributions
- **Ancient Language NLP**: Novel approaches for low-resource historical languages
- **Cross-Script Learning**: Transferable techniques for related writing systems
- **Parameter Efficiency**: Practical deployment strategies for resource-constrained environments

### Future Extensions
- **Multi-Modal Integration**: Manuscript image processing and OCR
- **Question Answering**: Direct answers from Sanskrit philosophical texts
- **Temporal Analysis**: Historical concept evolution tracking
- **Multi-Language Support**: Extension to other ancient languages (Latin, Greek, Arabic)

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{atharsh2025advanced,
  title={Advanced Zero-Shot Learning for Sanskrit-English Cross-Lingual Information Retrieval},
  author={Atharsh K and Kuricheti Raja Mukhesh},
  booktitle={Proceedings of LLNCS},
  year={2025},
  organization={PSG College of Technology}
}
```

## Dataset

This implementation uses the **Anveshana dataset**:
- **Source**: https://huggingface.co/datasets/manojbalaji1/anveshana
- **Size**: 3,400 Sanskrit-English query-document pairs
- **Domain**: Srimadbhagavatam philosophical texts
- **Splits**: 7,332 training, 814 validation, 1,021 test samples

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request with detailed description

### Areas for Contribution
- Additional ancient language support
- Improved few-shot example selection
- Enhanced cross-script normalization
- Performance optimization techniques

## License

This project is licensed under the MIT License. The Anveshana dataset is publicly available under its respective license terms.

## Acknowledgments

- **PSG College of Technology** for institutional support
- **Anveshana Dataset** creators for providing the benchmark
- **Hugging Face** for model hosting and datasets platform
- **Sanskrit NLP Community** for foundational research
