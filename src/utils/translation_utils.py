"""
Translation utilities for Google Translate API integration.
"""
import time
import logging
from typing import List, Dict, Optional
from googletrans import Translator
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manage translation operations using Google Translate API."""
    
    def __init__(self, config: Dict):
        """
        Initialize translation manager.
        
        Args:
            config: Translation configuration
        """
        self.config = config
        self.batch_size = config.get('batch_size', 50)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.delay_between_requests = config.get('delay_between_requests', 1.0)
        self.fallback_lang = config.get('fallback_lang', 'hi')  # Fallback for Sanskrit
        
        # Using default service_urls; customize if needed
        self.translator = Translator()
        self.cache = {}  # Simple cache for translations
    
    def _translate_once(self, text: str, source_lang: str, target_lang: str) -> str:
        result = self.translator.translate(text, src=source_lang, dest=target_lang)
        return result.text

    def translate_text(self, text: str, source_lang: str = 'en', target_lang: str = 'sa') -> str:
        """
        Translate a single text with graceful fallbacks.
        """
        if not text or not text.strip():
            return text
        
        cache_key = f"{text}_{source_lang}_{target_lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try requested languages with retries
        for attempt in range(self.retry_attempts):
            try:
                translated_text = self._translate_once(text, source_lang, target_lang)
                self.cache[cache_key] = translated_text
                return translated_text
            except Exception as e:
                msg = str(e).lower()
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                # Fallbacks for Sanskrit target
                if target_lang == 'sa' and ('invalid destination language' in msg or 'unsupported' in msg):
                    logger.warning(f"Falling back target language from 'sa' to '{self.fallback_lang}'")
                    try:
                        translated_text = self._translate_once(text, source_lang, self.fallback_lang)
                        self.cache[cache_key] = translated_text
                        return translated_text
                    except Exception as e2:
                        logger.warning(f"Fallback to '{self.fallback_lang}' failed: {e2}")
                        # Final no-op fallback
                        self.cache[cache_key] = text
                        return text
                # Fallback for Sanskrit source
                if source_lang == 'sa' and ('invalid source language' in msg or 'unsupported' in msg):
                    logger.warning("Falling back source language from 'sa' to 'auto'")
                    try:
                        translated_text = self._translate_once(text, 'auto', target_lang)
                        self.cache[cache_key] = translated_text
                        return translated_text
                    except Exception as e2:
                        logger.warning(f"Fallback from 'auto' failed: {e2}")
                        self.cache[cache_key] = text
                        return text
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.delay_between_requests)
                else:
                    logger.error(f"Translation failed after {self.retry_attempts} attempts: {e}")
                    self.cache[cache_key] = text
                    return text
    
    def translate_batch(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'sa') -> List[str]:
        """
        Translate a batch of texts with graceful fallbacks.
        """
        if not texts:
            return []
        
        logger.info(f"Translating batch of {len(texts)} texts from {source_lang} to {target_lang}")
        translated_texts = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_translations = []
            for text in batch:
                translated = self.translate_text(text, source_lang, target_lang)
                batch_translations.append(translated)
                time.sleep(self.delay_between_requests)
            translated_texts.extend(batch_translations)
            logger.info(f"Translated batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
        
        return translated_texts
    
    def translate_queries_to_sanskrit(self, queries: List[str]) -> List[str]:
        """
        Translate English queries to Sanskrit (for QT framework) with fallback.
        """
        return self.translate_batch(queries, source_lang='en', target_lang='sa')
    
    def translate_documents_to_english(self, documents: List[str]) -> List[str]:
        """
        Translate Sanskrit documents to English (for DT framework) with fallback.
        """
        # Use 'sa' as intended; internal logic falls back to 'auto' on failures
        return self.translate_batch(documents, source_lang='sa', target_lang='en')
    
    def get_translation_stats(self) -> Dict:
        """Get translation statistics."""
        return {
            'cache_size': len(self.cache),
            'batch_size': self.batch_size,
            'retry_attempts': self.retry_attempts,
            'delay_between_requests': self.delay_between_requests
        }
    
    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self.cache.clear()
        logger.info("Translation cache cleared")


class LanguageDetector:
    """Detect language of text."""
    
    def __init__(self):
        """Initialize language detector."""
        self.translator = Translator()
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Language code
        """
        if not text or not text.strip():
            return 'unknown'
        
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'unknown'
    
    def is_sanskrit(self, text: str) -> bool:
        """
        Check if text is in Sanskrit.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be Sanskrit
        """
        # Simple heuristic: check for Devanagari characters
        devanagari_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        return devanagari_chars / total_chars > 0.5
    
    def is_english(self, text: str) -> bool:
        """
        Check if text is in English.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be English
        """
        # Simple heuristic: check for Latin characters
        latin_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        return latin_chars / total_chars > 0.8 