#!/usr/bin/env python3
"""
Test script for Zero-shot (ZS) framework.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from zero_shot_framework.zs_runner import ZSRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_zs_framework():
    """Test ZS framework components and run all zero-shot models on a small subset."""
    logger.info("Testing Zero-shot Framework...")
    try:
        runner = ZSRunner()
        runner.load_data()
        # Reduce test set for speed
        runner.test_data = runner.test_data.select(range(min(10, len(runner.test_data))))
        runner.run_all_experiments()
        runner.print_results()
        logger.info("--------------------------------------------------")
        logger.info("ZS Framework: All zero-shot models evaluated on small subset.")
        return True
    except Exception as e:
        logger.error(f"ZS framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zs_framework()
    sys.exit(0 if success else 1)
