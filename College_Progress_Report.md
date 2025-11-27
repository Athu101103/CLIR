# College Progress Report

## TITLE OF THE PROJECT
**Advanced Zero-Shot Learning for Sanskrit-English Cross-Lingual Information Retrieval**

---

## OBJECTIVE AND PROBLEM STATEMENT

### Objective
To develop an advanced framework for Sanskrit-English Cross-Lingual Information Retrieval (CLIR) that enables users to search Sanskrit philosophical texts using English queries, thereby democratizing access to ancient wisdom and cultural heritage.

### Problem Statement
Traditional information retrieval systems face significant challenges when dealing with Sanskrit-English cross-lingual retrieval due to:
- Limited parallel training data between Sanskrit and English
- Complex morphological and syntactic structures in Sanskrit poetry
- Significant linguistic divergence between ancient Sanskrit and modern English
- Need for cultural and philosophical context understanding in spiritual texts
- Computational constraints for processing ancient language texts

---

## WHAT WAS EXISTING

### Traditional CLIR Approaches
- **Dictionary-based translation methods**: Simple word-to-word translation with limited semantic understanding
- **Query/Document translation strategies**: Basic translation of either queries or documents to a common language
- **Monolingual retrieval systems**: Separate systems for Sanskrit and English without cross-lingual capabilities

### Limitations of Existing Methods
- Poor handling of Sanskrit's complex grammatical structures
- Lack of cultural and philosophical context understanding
- Limited performance on ancient language texts
- High computational requirements for full model training
- Absence of specialized frameworks for Sanskrit-English retrieval

### Previous Sanskrit NLP Work
- Basic word segmentation and morphological parsing
- Limited compound analysis techniques
- No comprehensive cross-lingual retrieval solutions
- Minimal integration of modern transformer models for Sanskrit

---

## IMPROVEMENT MADE (APPROACH AND MODEL USED)

### Novel Framework Components

#### 1. Cross-Script Transfer Learning
**Innovation**: Leverages linguistic similarities between Sanskrit and related Devanagari languages
- **Source Languages**: Hindi, Marathi, Nepali (abundant parallel data available)
- **Technique**: Semantic alignment learning across multiple source languages
- **Process**: Devanagari script normalization → Feature extraction → Transfer to Sanskrit
- **Advantage**: Compensates for limited Sanskrit-English parallel data

#### 2. Multilingual Prompting with Large Language Models
**Innovation**: Few-shot learning with carefully curated philosophical examples
- **Model Used**: mBART-large-50 for multilingual understanding
- **Strategy**: Context-aware prompts with Sanskrit-English philosophical examples
- **Examples Include**: Consciousness concepts, dharma/karma principles, spiritual practices
- **Benefit**: Captures cultural and philosophical nuances that embedding methods miss

#### 3. Parameter-Efficient Fine-Tuning
**Innovation**: LoRA (Low-Rank Adaptation) for computational efficiency
- **Base Model**: XLM-RoBERTa-base (278M parameters)
- **LoRA Configuration**: Rank-16 decomposition, α=32, targeting attention layers
- **Objective**: Contrastive learning with temperature scaling (τ=0.07)
- **Achievement**: 99.2% parameter reduction while maintaining performance

### Mathematical Framework
```
Combined Score = α·sim_base(q,d) + β·sim_transfer(q,d) + γ·sim_llm(q,d)
```
Where α, β, γ are learned weights combining base model, transfer learning, and LLM similarities.

---

## DATASET USED

### Anveshana Dataset
- **Source**: Hugging Face (manojbalaji1/anveshana)
- **Content**: Sanskrit-English query-document pairs from Srimadbhagavatam
- **Total Size**: 3,400 query-document pairs
- **Domain**: Philosophical and spiritual texts covering diverse concepts

### Dataset Splits
- **Training Set**: 7,332 samples (with 2:1 negative sampling ratio)
- **Validation Set**: 814 samples (for hyperparameter tuning)
- **Test Set**: 1,021 samples (for final evaluation)

### Data Characteristics
- **Sanskrit Documents**: Complex poetic structures with philosophical content
- **English Queries**: Natural language questions about spiritual concepts
- **Preprocessing**: Specialized handling of Sanskrit morphology and poetic markers
- **Coverage**: Diverse philosophical topics including consciousness, ethics, metaphysics

---

## METRICS AND RESULTS

### Evaluation Metrics
- **NDCG@k** (k=1,3,5,10): Normalized Discounted Cumulative Gain for ranking quality
- **MAP@k**: Mean Average Precision across relevant documents
- **Recall@k**: Proportion of relevant documents retrieved in top-k results
- **Precision@k**: Proportion of retrieved documents that are relevant

### Performance Results

| Method | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | Improvement |
|--------|---------|--------|-----------|--------------|-------------|
| **Baseline (XLM-RoBERTa)** | 0.3245 | 0.2876 | 0.4521 | 0.1834 | - |
| **Cross-Script Transfer** | 0.4799 | 0.3226 | 0.6234 | 0.2145 | **+47.9%** |
| **Few-Shot Learning** | 0.5063 | 0.3608 | 0.5987 | 0.2298 | **+56.0%** |
| **Combined Framework** | **0.5847** | **0.4123** | **0.7234** | **0.2567** | **+80.2%** |

### Key Achievements
- ✅ **80.2% overall improvement** in NDCG@10 scores over baseline
- ✅ **State-of-the-art performance** on Sanskrit-English CLIR task
- ✅ **80% reduction** in computational overhead through LoRA
- ✅ **99.2% parameter reduction** while maintaining competitive performance
- ✅ **Robust cross-lingual understanding** for philosophical concepts

### Computational Efficiency
- **Training Time**: Reduced from 12 hours to 1.2 hours (90% reduction)
- **Memory Usage**: 60% reduction during fine-tuning
- **Inference Speed**: Minimal overhead (< 5% increase) for cross-script transfer
- **Resource Requirements**: Deployable on resource-constrained environments

### Significance of Results
- **Technical Impact**: Establishes new benchmarks for ancient language CLIR
- **Cultural Impact**: Democratizes access to Sanskrit philosophical texts
- **Academic Impact**: Provides scalable framework for other ancient languages
- **Practical Impact**: Enables real-world deployment for educational applications

---

## CURRENT STATUS AND FUTURE WORK

### Completed Milestones
- ✅ Complete framework implementation and testing
- ✅ Comprehensive experimental validation on Anveshana dataset
- ✅ Research paper preparation and documentation
- ✅ Performance benchmarking against state-of-the-art methods

### Future Enhancements
- **Multi-Modal Integration**: Incorporating manuscript images and OCR
- **Extended Language Support**: Adapting framework for Latin, Greek, Classical Chinese
- **Question-Answering**: Direct QA capabilities for Sanskrit texts
- **User Interface**: Web-based interface for practical deployment

---

**Student**: Atharsh K  
**Guide**: Dr. Kuricheti Raja Mukhesh  
**Institution**: PSG College of Technology, Department of Applied Mathematics and Computational Sciences  
**Project Duration**: June 2024 - January 2025  
**Report Date**: January 2025