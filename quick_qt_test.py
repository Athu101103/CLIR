#!/usr/bin/env python3
"""
Quick test for QT framework with mock translation.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qt_framework.qt_runner import QTRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_qt_test():
    """Quick test of QT framework with mock translation."""
    logger.info("Quick QT Framework Test...")
    
    try:
        # Initialize runner
        runner = QTRunner()
        
        # Load data
        runner.load_data()
        
        # Test BM25 with small subset
        logger.info("Testing BM25 with small subset...")
        small_train = runner.train_data.select(range(min(50, len(runner.train_data))))
        small_test = runner.test_data.select(range(min(10, len(runner.test_data))))
        
        # Temporarily replace data
        original_train = runner.train_data
        original_test = runner.test_data
        runner.train_data = small_train
        runner.test_data = small_test
        
        # Run BM25 experiment
        bm25_results = runner.run_bm25_experiment()
        
        # Restore data
        runner.train_data = original_train
        runner.test_data = original_test
        
        # Display results
        print("\n" + "="*60)
        print("QUICK QT TEST RESULTS")
        print("="*60)
        
        print(f"\nBM25 QT Results (with mock translation):")
        print("-" * 40)
        eval_results = bm25_results['evaluation_results']
        for metric in ['ndcg', 'map', 'recall', 'precision']:
            print(f"{metric.upper()}:", end=" ")
            for k in [1, 3, 5, 10]:
                value = eval_results[metric][k]
                print(f"@{k}={value:.4f}", end=" ")
            print()
        
        print(f"\nTraining Info:")
        print(f"  - Documents indexed: {bm25_results['train_metrics']['documents_indexed']}")
        print(f"  - Model: {bm25_results['model_name']}")
        
        print("\n" + "="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_qt_test()
    sys.exit(0 if success else 1)
