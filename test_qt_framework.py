#!/usr/bin/env python3
"""
Test script for Query Translation (QT) framework.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qt_framework.qt_runner import QTRunner
from utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_qt_framework():
    """Test the QT framework implementation."""
    logger.info("Testing QT Framework...")
    
    try:
        # Test configuration loading
        logger.info("1. Testing configuration loading...")
        config_loader = ConfigLoader()
        qt_config = config_loader.get_model_config('qt')
        
        assert 'bm25' in qt_config, "BM25 config missing"
        assert 'xlm_roberta' in qt_config, "XLM-RoBERTa config missing"
        logger.info("✓ Configuration loading successful")
        
        # Test QT runner initialization
        logger.info("2. Testing QT runner initialization...")
        runner = QTRunner()
        logger.info("✓ QT runner initialization successful")
        
        # Test data loading
        logger.info("3. Testing data loading...")
        runner.load_data()
        logger.info(f"✓ Data loading successful - Train: {len(runner.train_data)}, Test: {len(runner.test_data)}")
        
        # Test BM25 model (quick test without full training)
        logger.info("4. Testing BM25 model...")
        bm25_config = runner.qt_config.get('bm25', {})
        from qt_framework.bm25_qt import BM25QTModel
        bm25_model = BM25QTModel(bm25_config, runner.translation_config)
        
        # Test with a small subset
        small_train = runner.train_data.select(range(min(10, len(runner.train_data))))
        train_metrics = bm25_model.train(small_train)
        logger.info(f"✓ BM25 model test successful - {train_metrics}")
        
        # Test XLM-RoBERTa model initialization
        logger.info("5. Testing XLM-RoBERTa model initialization...")
        xlm_config = runner.qt_config.get('xlm_roberta', {})
        from qt_framework.xlm_roberta_qt import XLMQTModel
        xlm_model = XLMQTModel(xlm_config, runner.translation_config)
        logger.info("✓ XLM-RoBERTa model initialization successful")
        
        logger.info("✓ All QT framework tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ QT framework test failed: {e}")
        return False


def test_translation():
    """Test translation functionality."""
    logger.info("Testing translation functionality...")
    
    try:
        from utils.translation_utils import TranslationManager
        from utils.config_loader import ConfigLoader
        
        config_loader = ConfigLoader()
        translation_config = config_loader.get_translation_config()
        
        translation_manager = TranslationManager(translation_config)
        
        # Test with a simple query
        test_queries = ["What is the meaning of life?"]
        translated = translation_manager.translate_queries_to_sanskrit(test_queries)
        
        logger.info(f"✓ Translation test successful: '{test_queries[0]}' -> '{translated[0]}'")
        return True
        
    except Exception as e:
        logger.error(f"✗ Translation test failed: {e}")
        return False


def main():
    """Run all QT framework tests."""
    logger.info("Starting QT Framework Tests...")
    
    tests = [
        test_qt_framework,
        test_translation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"QT Framework Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✓ All QT framework tests passed! Ready for experiments.")
        return True
    else:
        logger.error("✗ Some QT framework tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
