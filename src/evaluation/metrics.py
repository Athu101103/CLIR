"""
Evaluation metrics for CLIR systems.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CLIREvaluator:
    """Evaluate CLIR system performance using various metrics."""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of k values for evaluation
        """
        self.k_values = k_values
    
    def calculate_ndcg(self, relevance_scores: List[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            relevance_scores: List of relevance scores (0 or 1)
            k: Cutoff value
            
        Returns:
            NDCG score
        """
        if not relevance_scores:
            return 0.0
        
        # Take only first k scores
        scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            dcg += score / np.log2(i + 2)  # log2(i+2) because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_map(self, relevance_scores: List[int], k: int) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            relevance_scores: List of relevance scores (0 or 1)
            k: Cutoff value
            
        Returns:
            MAP score
        """
        if not relevance_scores:
            return 0.0
        
        # Take only first k scores
        scores = relevance_scores[:k]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, score in enumerate(scores):
            if score == 1:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        # Calculate average precision
        if not precisions:
            return 0.0
        
        return np.mean(precisions)
    
    def calculate_recall(self, relevance_scores: List[int], k: int, total_relevant: int) -> float:
        """
        Calculate Recall.
        
        Args:
            relevance_scores: List of relevance scores (0 or 1)
            k: Cutoff value
            total_relevant: Total number of relevant documents
            
        Returns:
            Recall score
        """
        if total_relevant == 0:
            return 0.0
        
        # Take only first k scores
        scores = relevance_scores[:k]
        
        # Count relevant documents in top k
        relevant_retrieved = sum(scores)
        
        return relevant_retrieved / total_relevant
    
    def calculate_precision(self, relevance_scores: List[int], k: int) -> float:
        """
        Calculate Precision.
        
        Args:
            relevance_scores: List of relevance scores (0 or 1)
            k: Cutoff value
            
        Returns:
            Precision score
        """
        if k == 0:
            return 0.0
        
        # Take only first k scores
        scores = relevance_scores[:k]
        
        # Count relevant documents in top k
        relevant_retrieved = sum(scores)
        
        return relevant_retrieved / k
    
    def evaluate_query(self, 
                      predicted_scores: List[float], 
                      true_relevance: List[int], 
                      k_values: Optional[List[int]] = None) -> Dict[str, Dict[int, float]]:
        """
        Evaluate a single query.
        
        Args:
            predicted_scores: Predicted relevance scores
            true_relevance: True relevance labels
            k_values: List of k values (defaults to self.k_values)
            
        Returns:
            Dictionary of metrics for each k value
        """
        if k_values is None:
            k_values = self.k_values
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        # Ensure indices don't exceed the length of true_relevance
        sorted_indices = [i for i in sorted_indices if i < len(true_relevance)]
        sorted_relevance = [true_relevance[i] for i in sorted_indices]
        
        results = {
            'ndcg': {},
            'map': {},
            'recall': {},
            'precision': {}
        }
        
        total_relevant = sum(true_relevance)
        
        for k in k_values:
            results['ndcg'][k] = self.calculate_ndcg(sorted_relevance, k)
            results['map'][k] = self.calculate_map(sorted_relevance, k)
            results['recall'][k] = self.calculate_recall(sorted_relevance, k, total_relevant)
            results['precision'][k] = self.calculate_precision(sorted_relevance, k)
        
        return results
    
    def evaluate_batch(self, 
                      all_predictions: List[List[float]], 
                      all_true_relevance: List[List[int]], 
                      k_values: Optional[List[int]] = None) -> Dict[str, Dict[int, float]]:
        """
        Evaluate a batch of queries.
        
        Args:
            all_predictions: List of predicted scores for each query
            all_true_relevance: List of true relevance for each query
            k_values: List of k values (defaults to self.k_values)
            
        Returns:
            Average metrics across all queries
        """
        if k_values is None:
            k_values = self.k_values
        
        if len(all_predictions) != len(all_true_relevance):
            raise ValueError("Number of predictions and true relevance must match")
        
        # Evaluate each query
        query_results = []
        for pred_scores, true_rel in zip(all_predictions, all_true_relevance):
            query_result = self.evaluate_query(pred_scores, true_rel, k_values)
            query_results.append(query_result)
        
        # Calculate averages
        avg_results = {
            'ndcg': {},
            'map': {},
            'recall': {},
            'precision': {}
        }
        
        for metric in ['ndcg', 'map', 'recall', 'precision']:
            for k in k_values:
                values = [qr[metric][k] for qr in query_results]
                avg_results[metric][k] = np.mean(values)
        
        return avg_results
    
    def format_results(self, results: Dict[str, Dict[int, float]]) -> str:
        """
        Format results as a readable string.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted string
        """
        output = []
        for metric in ['ndcg', 'map', 'recall', 'precision']:
            metric_line = f"{metric.upper()}:"
            for k in self.k_values:
                value = results[metric][k]
                metric_line += f" @{k}={value:.4f}"
            output.append(metric_line)
        
        return "\n".join(output)
    
    def save_results(self, results: Dict[str, Dict[int, float]], filepath: str) -> None:
        """
        Save results to a file.
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for metric, k_dict in results.items():
            serializable_results[metric] = {str(k): float(v) for k, v in k_dict.items()}
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def create_evaluation_table(results_dict: Dict[str, Dict[str, Dict[int, float]]]) -> str:
    """
    Create a formatted table from multiple model results.
    
    Args:
        results_dict: Dictionary of model results
        
    Returns:
        Formatted table string
    """
    if not results_dict:
        return "No results to display"
    
    # Get all k values and metrics
    first_result = next(iter(results_dict.values()))
    k_values = sorted(first_result['ndcg'].keys())
    metrics = ['ndcg', 'map', 'recall', 'precision']
    
    # Create header
    header = "Framework\tModel\t"
    for metric in metrics:
        for k in k_values:
            header += f"{metric.upper()}@{k}\t"
    header = header.rstrip('\t')
    
    # Create rows
    rows = []
    for framework_model, results in results_dict.items():
        row = f"{framework_model}\t"
        for metric in metrics:
            for k in k_values:
                value = results[metric][k]
                row += f"{value:.4f}\t"
        row = row.rstrip('\t')
        rows.append(row)
    
    # Combine
    table = header + "\n" + "\n".join(rows)
    return table 