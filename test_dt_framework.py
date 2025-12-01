#!/usr/bin/env python3
"""
Test script for Document Translation (DT) framework.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dt_framework.dt_runner import DTRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dt_framework():
    """Test DT framework components."""
    logger.info("Testing DT Framework...")
    
    try:
        # Test 1: Configuration loading
        logger.info("Test 1: Configuration loading")
        runner = DTRunner()
        logger.info("✓ Configuration loaded successfully")
        
        # Test 2: Data loading
        logger.info("Test 2: Data loading")
        runner.load_data()
        logger.info("✓ Data loaded successfully")
        
        # Test 3: BM25 DT model initialization
        logger.info("Test 3: BM25 DT model initialization")
        from dt_framework.bm25_dt import BM25DTModel
        bm25_config = runner.dt_config['bm25']
        bm25_model = BM25DTModel(bm25_config, runner.translation_config)
        logger.info("✓ BM25 DT model initialized successfully")
        
        # Test 4: Contriever DT model initialization
        logger.info("Test 4: Contriever DT model initialization")
        from dt_framework.contriever_dt import ContrieverDTModel
        contriever_config = runner.dt_config['contriever']
        contriever_model = ContrieverDTModel(contriever_config, runner.translation_config)
        logger.info("✓ Contriever DT model initialized successfully")
        
        # Test 5: Translation functionality
        logger.info("Test 5: Translation functionality")
        test_docs = ["Test Sanskrit document 1", "Test Sanskrit document 2"]
        translated_docs = bm25_model.translate_documents(test_docs)
        logger.info(f"✓ Translation test successful: '{test_docs[0]}' -> '{translated_docs[0]}'")
        
        logger.info("--------------------------------------------------")
        logger.info("DT Framework Tests completed: 5/5 passed")
        logger.info("✓ All DT framework tests passed! Ready for experiments.")
        
        return True
        
    except Exception as e:
        logger.error(f"DT framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dt_framework()
    sys.exit(0 if success else 1)
