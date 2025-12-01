#!/usr/bin/env python3
"""
Test script to verify the basic setup and data loading.
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config_loader import ConfigLoader
from utils.data_loader import AnveshanaDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.get_full_config()
        
        # Test basic config structure
        assert 'dataset' in config, "Dataset config missing"
        assert 'models' in config, "Models config missing"
        assert 'evaluation' in config, "Evaluation config missing"
        
        logger.info("✓ Configuration loading successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False


def test_data_loading():
    """Test dataset loading."""
    logger.info("Testing dataset loading...")
    
    try:
        # Load config
        config_loader = ConfigLoader()
        dataset_config = config_loader.get_dataset_config()
        
        # Initialize data loader
        data_loader = AnveshanaDataLoader(dataset_config)
        
        # Load dataset
        data_loader.load_dataset()
        
        # Test preprocessing
        data_loader.preprocess_dataset()
        
        # Test splits
        data_loader.create_splits()
        
        # Get data
        train_data = data_loader.get_train_data()
        validation_data = data_loader.get_validation_data()
        test_data = data_loader.get_test_data()
        
        logger.info(f"✓ Dataset loading successful")
        logger.info(f"  - Train: {len(train_data)} samples")
        logger.info(f"  - Validation: {len(validation_data)} samples")
        logger.info(f"  - Test: {len(test_data)} samples")
        
        # Test sample data
        sample = train_data[0]
        logger.info(f"  - Sample query: {sample['query'][:100]}...")
        logger.info(f"  - Sample document: {sample['document'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        return False


def test_preprocessing():
    """Test text preprocessing."""
    logger.info("Testing text preprocessing...")
    
    try:
        config_loader = ConfigLoader()
        preprocessing_config = config_loader.get_preprocessing_config()
        
        # Test Sanskrit preprocessing
        sanskrit_text = "——1.1.3—— यथा सूर्यस्य रश्मयः सर्वे तद्गतास्तथा सर्वे लोकाः परमात्मगताः"
        expected_processed = "—— यथा सूर्यस्य रश्मयः सर्वे तद्गतास्तथा सर्वे लोकाः परमात्मगताः"
        
        data_loader = AnveshanaDataLoader({'sanskrit': preprocessing_config.get('sanskrit', {})})
        processed = data_loader.preprocess_sanskrit_text(sanskrit_text)
        
        assert "——1.1.3——" not in processed, "Poetic structure not handled correctly"
        logger.info("✓ Sanskrit preprocessing successful")
        
        # Test English preprocessing
        english_text = "What are the nine different ways of rendering devotional service?"
        processed_eng = data_loader.preprocess_english_text(english_text)
        
        assert processed_eng == english_text, "English preprocessing should preserve text"
        logger.info("✓ English preprocessing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting setup tests...")
    
    tests = [
        test_config_loading,
        test_data_loading,
        test_preprocessing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✓ All tests passed! Setup is ready.")
        return True
    else:
        logger.error("✗ Some tests failed. Please check the setup.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 