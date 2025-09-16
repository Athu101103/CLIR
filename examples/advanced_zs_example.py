#!/usr/bin/env python3
"""
Example usage of Advanced Zero-Shot Learning for Sanskrit-English CLIR

This example demonstrates:
1. Setting up the advanced zero-shot model
2. Using cross-script transfer learning
3. Implementing few-shot learning with examples
4. Combining multiple techniques
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.zero_shot_framework.advanced_zs_learning import AdvancedZeroShotModel, create_few_shot_examples

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example usage of advanced zero-shot learning"""
    print("🔥 Advanced Zero-Shot Learning for Sanskrit-English CLIR")
    print("=" * 70)
    
    # Sample Sanskrit-English data
    english_queries = [
        "What is the nature of the soul?",
        "How can one achieve enlightenment?",
        "What is the meaning of dharma?",
        "What are the principles of yoga?",
        "What is the purpose of meditation?"
    ]
    
    sanskrit_documents = [
        "आत्मा नित्यः शुद्धः बुद्धः मुक्तः स्वभावः।",  # The soul is eternal, pure, enlightened, liberated by nature
        "योगश्चित्तवृत्तिनिरोधः तदा द्रष्टुः स्वरूपे अवस्थानम्।",  # Yoga is the cessation of mental fluctuations
        "धर्मः धारयते जगत् धर्मेण धार्यते प्रजाः।",  # Dharma sustains the world; people are sustained by dharma
        "ध्यानं निर्विषयं मनसः प्रत्याहारो इन्द्रियाणाम्।",  # Meditation is the mind without objects
        "मोक्षः सर्वदुःखनिवृत्तिः परमानन्दप्राप्तिः।"  # Liberation is freedom from all suffering
    ]
    
    try:
        print("\n1. Setting up Advanced Zero-Shot Model")
        print("-" * 40)
        
        # Configure the advanced model
        config = {
            'model_name': 'distilbert-base-multilingual-cased',  # Smaller model for example
            'source_languages': ['hindi', 'marathi'],
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'llm_model': 'google/mt5-small'
        }
        
        print(f"Base model: {config['model_name']}")
        print(f"Cross-script languages: {config['source_languages']}")
        print(f"LoRA configuration: r={config['lora_r']}, alpha={config['lora_alpha']}")
        
        # Initialize the model
        advanced_model = AdvancedZeroShotModel(config)
        print("✅ Advanced model initialized successfully!")
        
        print("\n2. Creating Few-Shot Examples")
        print("-" * 40)
        
        # Get few-shot examples
        few_shot_examples = create_few_shot_examples()
        print(f"Created {len(few_shot_examples)} few-shot examples:")
        
        for i, example in enumerate(few_shot_examples[:3]):
            print(f"  {i+1}. English: {example['english']}")
            print(f"     Sanskrit: {example['sanskrit']}")
            print(f"     Context: {example['explanation']}")
            print()
        
        print("\n3. Standard Retrieval (Baseline)")
        print("-" * 40)
        
        standard_scores = advanced_model.retrieve_documents(
            english_queries, 
            sanskrit_documents, 
            k=3,
            use_cross_script=False,
            use_few_shot=False
        )
        
        print("Top matches for each query (standard approach):")
        for i, query in enumerate(english_queries):
            print(f"\nQuery: {query}")
            query_scores = standard_scores[i]
            top_indices = sorted(range(len(query_scores)), key=lambda x: query_scores[x], reverse=True)[:2]
            
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. {sanskrit_documents[idx][:50]}... (Score: {query_scores[idx]:.3f})")
        
        print("\n4. Cross-Script Transfer Learning")
        print("-" * 40)
        
        cross_script_scores = advanced_model.retrieve_documents(
            english_queries, 
            sanskrit_documents, 
            k=3,
            use_cross_script=True,
            use_few_shot=False
        )
        
        print("Top matches with cross-script transfer:")
        for i, query in enumerate(english_queries[:2]):  # Show first 2 for brevity
            print(f"\nQuery: {query}")
            query_scores = cross_script_scores[i]
            top_indices = sorted(range(len(query_scores)), key=lambda x: query_scores[x], reverse=True)[:2]
            
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. {sanskrit_documents[idx][:50]}... (Score: {query_scores[idx]:.3f})")
        
        print("\n5. Few-Shot Learning with Examples")
        print("-" * 40)
        
        few_shot_scores = advanced_model.retrieve_documents(
            english_queries[:2],  # Limit for demo
            sanskrit_documents, 
            k=3,
            use_cross_script=False,
            use_few_shot=True,
            few_shot_examples=few_shot_examples
        )
        
        print("Top matches with few-shot learning:")
        for i, query in enumerate(english_queries[:2]):
            print(f"\nQuery: {query}")
            query_scores = few_shot_scores[i]
            top_indices = sorted(range(len(query_scores)), key=lambda x: query_scores[x], reverse=True)[:2]
            
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. {sanskrit_documents[idx][:50]}... (Score: {query_scores[idx]:.3f})")
        
        print("\n6. Combined Advanced Techniques")
        print("-" * 40)
        
        combined_scores = advanced_model.retrieve_documents(
            english_queries[:2],  # Limit for demo
            sanskrit_documents, 
            k=3,
            use_cross_script=True,
            use_few_shot=True,
            few_shot_examples=few_shot_examples
        )
        
        print("Top matches with combined techniques:")
        for i, query in enumerate(english_queries[:2]):
            print(f"\nQuery: {query}")
            query_scores = combined_scores[i]
            top_indices = sorted(range(len(query_scores)), key=lambda x: query_scores[x], reverse=True)[:2]
            
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. {sanskrit_documents[idx][:50]}... (Score: {query_scores[idx]:.3f})")
        
        print("\n" + "=" * 70)
        print("🎉 Advanced Zero-Shot Learning Example Completed Successfully!")
        print("=" * 70)
        
        print("\nKey Features Demonstrated:")
        print("✅ Cross-Script Transfer Learning from Hindi/Marathi to Sanskrit")
        print("✅ Few-Shot Learning with Sanskrit-English examples")
        print("✅ Multilingual prompting capabilities")
        print("✅ Combined technique approach")
        print("✅ Parameter-efficient fine-tuning setup (LoRA)")
        
        print("\nNext Steps:")
        print("• Run full experiments: python src/zero_shot_framework/advanced_zs_runner.py")
        print("• Test implementation: python test_advanced_zs.py")
        print("• Customize configuration in config.yaml")
        print("• Add your own few-shot examples for better performance")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")
        print("\nThis is expected if you haven't installed all dependencies.")
        print("To fix, run: pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())