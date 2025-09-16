"""
Unified runner to execute a set of queries across multiple CLIR models and compute metrics.

- Supports Zero-Shot models and (optionally) Advanced Zero-Shot model
- Computes NDCG, MAP, Recall, Precision using CLIREvaluator
- Outputs a structured JSON with per-model results and per-query top-k

SAFE DEFAULTS (to avoid OOM / GPU issues):
- ColBERT is DISABLED unless CUDA is available AND ENABLE_COLBERT=1
- Advanced model is DISABLED by default. Enable when resources allow.

Usage:
  python examples/unified_query_runner.py

Customize `QUERIES`, `DOCUMENTS`, and `RELEVANCE_INDICES` in main(),
or adapt this module to load your dataset programmatically.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch

# Ensure `src` is importable when running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.evaluation.metrics import CLIREvaluator

# Zero-shot models
from src.zero_shot_framework.xlm_roberta_zs import XLMRoBERTaZSModel
from src.zero_shot_framework.contriever_zs import ContrieverZSModel
from src.zero_shot_framework.e5_zs import MultilingualE5ZSModel

# Advanced model
from src.zero_shot_framework.advanced_zs_learning import (
    AdvancedZeroShotModel,
    create_few_shot_examples,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_true_relevance_from_indices(
    num_queries: int,
    num_documents: int,
    relevance_indices_per_query: List[List[int]],
) -> List[List[int]]:
    """Build a binary relevance matrix [num_queries x num_documents] from indices per query.

    relevance_indices_per_query: list where each entry is a list of document indices (0-based)
    considered relevant for that query.
    """
    true_relevance: List[List[int]] = []
    for q in range(num_queries):
        indices = set(relevance_indices_per_query[q]) if q < len(relevance_indices_per_query) else set()
        row = [1 if d in indices else 0 for d in range(num_documents)]
        true_relevance.append(row)
    return true_relevance


def get_top_k_for_query(scores: List[float], documents: List[str], k: int) -> List[Dict[str, Any]]:
    """Return the top-k documents with scores for a single query."""
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top = indexed[:k]
    return [
        {
            "doc_index": idx,
            "document": documents[idx],
            "score": float(score),
        }
        for idx, score in top
    ]


def run_all_models(
    queries: List[str],
    documents: List[str],
    true_relevance: List[List[int]],
    k_values: List[int] = [1, 3, 5, 10],
    include_advanced: bool = False,
    use_advanced_cross_script: bool = False,
    use_advanced_few_shot: bool = False,
) -> Dict[str, Any]:
    """Run queries across all available models and compute metrics.

    Returns a dict with per-model metrics and per-query top-k.
    """
    results: Dict[str, Any] = {}
    evaluator = CLIREvaluator(k_values=k_values)
    max_k = max(k_values)

    # Construct model registry
    model_specs: List[Dict[str, Any]] = [
        {
            "name": "XLMRoBERTaZSModel",
            "factory": lambda: XLMRoBERTaZSModel({"model_name": "xlm-roberta-base"}),
            "runner": lambda model: model.retrieve_documents(queries, documents, k=max_k),
        },
        {
            "name": "ContrieverZSModel",
            "factory": lambda: ContrieverZSModel({"model_name": "mjwong/contriever-mnli"}),
            "runner": lambda model: model.retrieve_documents(queries, documents, k=max_k),
        },
        {
            "name": "MultilingualE5ZSModel",
            "factory": lambda: MultilingualE5ZSModel({"model_name": "intfloat/multilingual-e5-base"}),
            "runner": lambda model: model.retrieve_documents(queries, documents, k=max_k),
        },
    ]

    # ColBERT may not be installed; try/catch at construction time
    # Enable ColBERT only if CUDA is available and explicitly requested
    enable_colbert = torch.cuda.is_available() and os.getenv("ENABLE_COLBERT") == "1"
    if enable_colbert:
        def _mk_colbert():
            # Lazy import to avoid hard dependency when disabled
            from src.zero_shot_framework.colbert_zs import ColBERTZSModel  # type: ignore
            return ColBERTZSModel({"model_name": "colbert-ir/colbertv2.0"})

        model_specs.append(
            {
                "name": "ColBERTZSModel",
                "factory": _mk_colbert,
                "runner": lambda model: model.retrieve_documents(queries, documents, k=max_k),
            }
        )
    else:
        logger.info("ColBERT disabled (requires CUDA and ENABLE_COLBERT=1)")

    if include_advanced:
        def _mk_advanced():
            # Use smaller models to reduce memory footprint
            config = {
                "model_name": "distilbert-base-multilingual-cased",
                "source_languages": ["hindi", "marathi", "nepali"],
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "llm_model": "google/mt5-small",
            }
            return AdvancedZeroShotModel(config)

        def _run_advanced(model):
            few_shot_examples = create_few_shot_examples() if use_advanced_few_shot else None
            return model.retrieve_documents(
                queries,
                documents,
                k=max_k,
                use_cross_script=use_advanced_cross_script,
                use_few_shot=use_advanced_few_shot,
                few_shot_examples=few_shot_examples,
            )

        model_specs.append(
            {
                "name": "AdvancedZeroShotModel",
                "factory": _mk_advanced,
                "runner": _run_advanced,
            }
        )

    for spec in model_specs:
        model_name = spec["name"]
        logger.info(f"Running model: {model_name}")
        model_result: Dict[str, Any] = {"model": model_name}
        try:
            model = spec["factory"]()
            scores: List[List[float]] = spec["runner"](model)
            # Evaluate
            metrics = evaluator.evaluate_batch(scores, true_relevance, k_values)
            # Per-query top-k (use max_k)
            per_query_topk = []
            for qi, query in enumerate(queries):
                topk = get_top_k_for_query(scores[qi], documents, max_k)
                per_query_topk.append({
                    "query": query,
                    "top_k": topk,
                })
            model_result.update({
                "metrics": metrics,
                "per_query": per_query_topk,
            })
        except Exception as e:
            logger.exception(f"Model {model_name} failed")
            model_result.update({
                "error": str(e),
            })
        results[model_name] = model_result

    return results


def save_results(results: Dict[str, Any], output_dir: str = "experiments/runs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"unified_query_results_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Unified results saved to: {path}")
    return path


def main():
    # Example data (replace with your dataset)
    QUERIES: List[str] = [
        "What is knowledge?",
        "The soul is eternal",
    ]

    DOCUMENTS: List[str] = [
        "ज्ञानं किम्?",
        "धर्मः कर्तव्यम्",
        "आत्मा नित्यः",
        "योगः युक्तिः",
    ]

    # Relevant doc indices per query (0-based)
    # For query 0 ("What is knowledge?") -> doc 0 is relevant
    # For query 1 ("The soul is eternal") -> doc 2 is relevant
    RELEVANCE_INDICES: List[List[int]] = [
        [0],
        [2],
    ]

    true_relevance = build_true_relevance_from_indices(
        num_queries=len(QUERIES),
        num_documents=len(DOCUMENTS),
        relevance_indices_per_query=RELEVANCE_INDICES,
    )

    k_values = [1, 3]

    results = run_all_models(
        queries=QUERIES,
        documents=DOCUMENTS,
        true_relevance=true_relevance,
        k_values=k_values,
        include_advanced=False,  # Enable later if resources allow
        use_advanced_cross_script=False,
        use_advanced_few_shot=False,
    )

    save_results(results, output_dir="experiments/runs")


if __name__ == "__main__":
    main()


