#!/usr/bin/env python3
"""
Test script for Direct Retrieve (DR) framework.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dr_framework.dr_runner import DRRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dr_framework():
    """Test DR framework components."""
    logger.info("Testing DR Framework...")
    
    try:
        # Test 1: Configuration loading
        logger.info("Test 1: Configuration loading")
        runner = DRRunner()
        logger.info("✓ Configuration loaded successfully")
        
        # Test 2: Data loading
        logger.info("Test 2: Data loading")
        runner.load_data()
        logger.info("✓ Data loaded successfully")
        
        # Test 3: mDPR DR model initialization
        logger.info("Test 3: mDPR DR model initialization")
        from dr_framework.mdpr_dr import MDPRDRModel
        mdpr_config = runner.dr_config['mdpr']
        mdpr_model = MDPRDRModel(mdpr_config)
        logger.info("✓ mDPR DR model initialized successfully")
        
        # Test 4: XLM-RoBERTa DR model initialization
        logger.info("Test 4: XLM-RoBERTa DR model initialization")
        from dr_framework.xlm_roberta_dr import XLMRoBERTaDRModel
        xlm_config = runner.dr_config['xlm_roberta']
        xlm_model = XLMRoBERTaDRModel(xlm_config)
        logger.info("✓ XLM-RoBERTa DR model initialized successfully")
        
        # Test 5: Direct retrieval functionality
        logger.info("Test 5: Direct retrieval functionality")
        test_queries = ["What is the meaning of life?"]
        test_docs = ["Test Sanskrit document 1", "Test Sanskrit document 2"]
        scores = mdpr_model.retrieve_documents(test_queries, test_docs, k=2)
        logger.info(f"✓ Direct retrieval test successful: {len(scores[0])} scores generated")
        
        logger.info("--------------------------------------------------")
        logger.info("DR Framework Tests completed: 5/5 passed")
        logger.info("✓ All DR framework tests passed! Ready for experiments.")
        
        return True
        
    except Exception as e:
        logger.error(f"DR framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dr_framework()
    sys.exit(0 if success else 1)
