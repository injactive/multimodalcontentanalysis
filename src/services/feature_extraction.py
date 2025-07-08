"""
Feature extraction service for multimodal content analysis.

This service provides async feature extraction from text and images,
with proper error handling, caching, and MLflow integration.
"""

import asyncio
import time
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import io

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from ..models.domain import (
    AnalysisRequest, TextFeatures, ImageFeatures, 
    MultiModalFeatures, ContentType
)
from ..utils.http_client import AsyncHTTPClient
from ..utils.mlflow_tracker import MLflowTracker
from ..utils.logger import get_logger
from config.settings import APISettings


logger = get_logger(__name__)


class FeatureExtractionService:
    """Service for extracting features from text and images."""
    
    def __init__(
        self,
        http_client: AsyncHTTPClient,
        mlflow_tracker: MLflowTracker,
        settings: APISettings
    ):
        self.http_client = http_client
        self.mlflow_tracker = mlflow_tracker
        self.settings = settings
        self.logger = logger.bind(service="feature_extraction")
        
        # Initialize components
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._feature_cache: Dict[str, Any] = {}
        self._initialized = False
        
        # Sentiment keywords (simple approach for demo)
        self._positive_words = {
            'amazing', 'awesome', 'beautiful', 'best', 'brilliant', 'excellent',
            'fantastic', 'good', 'great', 'incredible', 'love', 'perfect',
            'wonderful', 'outstanding', 'superb', 'magnificent', 'spectacular'
        }
        self._negative_words = {
            'awful', 'bad', 'boring', 'disappointing', 'hate', 'horrible',
            'terrible', 'ugly', 'worst', 'disgusting', 'annoying', 'frustrating'
        }
    
    async def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return
        
        self.logger.info("Initializing feature extraction service...")
        
        try:
            # Initialize TF-IDF vectorizer with common social media vocabulary
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            # Pre-fit with sample social media text
            sample_texts = [
                "Check out this amazing product! #love #awesome",
                "Having a great day with friends! So happy 😊",
                "This is the worst experience ever. Very disappointed.",
                "Beautiful sunset today! Nature is incredible 🌅",
                "New blog post is live! Check it out and let me know what you think"
            ]
            self._tfidf_vectorizer.fit(sample_texts)
            
            self._initialized = True
            self.logger.info("Feature extraction service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature extraction service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup service resources."""
        self.logger.info("Cleaning up feature extraction service...")
        self._feature_cache.clear()
        self._initialized = False
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Generate cache key for content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_type}:{content_hash}"
    
    async def extract_features(
        self,
        request: AnalysisRequest,
        request_id: str
    ) -> MultiModalFeatures:
        """Extract features from the analysis request."""
        start_time = time.time()
        
        # Start MLflow run
        run_id = await self.mlflow_tracker.start_run(
            run_name=f"feature_extraction_{request_id}",
            tags={
                "request_id": request_id,
                "content_type": str(request.content_type),
                "has_image": request.image_url is not None
            }
        )
        
        try:
            # Extract text features
            text_features = await self._extract_text_features(
                request.text, request_id
            )
            
            # Extract image features if image URL provided
            image_features = None
            text_image_similarity = None
            
            if request.image_url:
                image_features = await self._extract_image_features(
                    str(request.image_url), request_id
                )
                text_image_similarity = await self._calculate_text_image_similarity(
                    request.text, image_features
                )
            
            # Calculate content coherence
            content_coherence = self._calculate_content_coherence(
                text_features, image_features, text_image_similarity
            )
            
            # Create multimodal features
            features = MultiModalFeatures(
                text_features=text_features,
                image_features=image_features,
                text_image_similarity=text_image_similarity,
                content_coherence=content_coherence
            )
            
            processing_time = time.time() - start_time
            
            # Log metrics to MLflow
            await self.mlflow_tracker.log_metrics(
                run_id=run_id,
                metrics={
                    "processing_time": processing_time,
                    "text_length": len(request.text),
                    "sentiment_score": text_features.sentiment_score,
                    "readability_score": text_features.readability_score,
                    "content_coherence": content_coherence,
                    "text_image_similarity": text_image_similarity or 0.0
                }
            )
            
            # Log parameters
            await self.mlflow_tracker.log_params(
                run_id=run_id,
                params={
                    "content_type": str(request.content_type),
                    "has_image": request.image_url is not None,
                    "text_word_count": text_features.word_count,
                    "hashtag_count": len(text_features.hashtags),
                    "mention_count": len(text_features.mentions)
                }
            )
            
            await self.mlflow_tracker.end_run(run_id, "FINISHED")
            
            self.logger.info(
                "Feature extraction completed",
                request_id=request_id,
                processing_time=processing_time,
                content_coherence=content_coherence
            )
            
            return features
            
        except Exception as e:
            await self.mlflow_tracker.end_run(run_id, "FAILED")
            self.logger.error(
                f"Feature extraction failed: {e}",
                request_id=request_id,
                error=str(e)
            )
            raise
    
    async def _extract_text_features(
        self,
        text: str,
        request_id: str
    ) -> TextFeatures:
        """Extract features from text content."""
        cache_key = self._generate_cache_key(text, "text")
        
        # Check cache
        if cache_key in self._feature_cache:
            self.logger.debug("Using cached text features", request_id=request_id)
            return self._feature_cache[cache_key]
        
        # Basic text statistics
        word_count = len(text.split())
        character_count = len(text)
        
        # Extract hashtags, mentions, and URLs
        hashtags = re.findall(r'#\w+', text.lower())
        mentions = re.findall(r'@\w+', text.lower())
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        # Count emojis (simple approach)
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿]', text))
        
        # Sentiment analysis (simple keyword-based approach)
        sentiment_score = self._calculate_sentiment(text)
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Readability score (simplified Flesch Reading Ease)
        readability_score = self._calculate_readability(text)
        
        # Extract keywords using TF-IDF
        keywords = self._extract_keywords(text)
        
        features = TextFeatures(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            readability_score=readability_score,
            word_count=word_count,
            character_count=character_count,
            keywords=keywords,
            hashtags=hashtags,
            mentions=mentions,
            emoji_count=emoji_count,
            url_count=len(urls)
        )
        
        # Cache the result
        self._feature_cache[cache_key] = features
        
        return features
    
    async def _extract_image_features(
        self,
        image_url: str,
        request_id: str
    ) -> ImageFeatures:
        """Extract features from image content."""
        cache_key = self._generate_cache_key(image_url, "image")
        
        # Check cache
        if cache_key in self._feature_cache:
            self.logger.debug("Using cached image features", request_id=request_id)
            return self._feature_cache[cache_key]
        
        try:
            # Download image
            image_data = await self.http_client.download_image(image_url)
            
            # Basic image properties (mock implementation)
            # In a real implementation, you would use PIL or similar
            features = ImageFeatures(
                width=1024,  # Mock values
                height=768,
                aspect_ratio=1024/768,
                file_size=len(image_data),
                format="jpg",
                has_faces=False,  # Would use face detection
                dominant_colors=["#FF5733", "#33FF57"],  # Would use color analysis
                brightness=0.6,
                contrast=0.7,
                clip_embedding=None  # Would use CLIP model
            )
            
            # Cache the result
            self._feature_cache[cache_key] = features
            
            return features
            
        except Exception as e:
            self.logger.error(
                f"Failed to extract image features: {e}",
                request_id=request_id,
                image_url=image_url
            )
            # Return default features on error
            return ImageFeatures(
                width=0,
                height=0,
                aspect_ratio=1.0,
                file_size=0,
                format="unknown",
                has_faces=False,
                dominant_colors=[],
                brightness=0.5,
                contrast=0.5
            )
    
    async def _calculate_text_image_similarity(
        self,
        text: str,
        image_features: Optional[ImageFeatures]
    ) -> Optional[float]:
        """Calculate text-image similarity using CLIP (mock implementation)."""
        if not image_features or not image_features.clip_embedding:
            return None
        
        # Mock CLIP similarity calculation
        # In real implementation, you would use CLIP model
        text_length_factor = min(len(text) / 100, 1.0)
        image_size_factor = min(image_features.file_size / (1024 * 1024), 1.0)
        
        # Simple heuristic based on content characteristics
        similarity = (text_length_factor + image_size_factor) / 2
        return min(max(similarity, 0.0), 1.0)
    
    def _calculate_content_coherence(
        self,
        text_features: TextFeatures,
        image_features: Optional[ImageFeatures],
        text_image_similarity: Optional[float]
    ) -> float:
        """Calculate overall content coherence score."""
        coherence_factors = []
        
        # Text coherence factors
        if text_features.readability_score > 50:
            coherence_factors.append(0.8)
        else:
            coherence_factors.append(0.4)
        
        if text_features.word_count > 10:
            coherence_factors.append(0.7)
        else:
            coherence_factors.append(0.3)
        
        # Image coherence factors
        if image_features:
            if image_features.aspect_ratio > 0.5 and image_features.aspect_ratio < 2.0:
                coherence_factors.append(0.8)
            else:
                coherence_factors.append(0.5)
        
        # Text-image alignment
        if text_image_similarity:
            coherence_factors.append(text_image_similarity)
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using keyword-based approach."""
        words = set(text.lower().split())
        
        positive_count = len(words.intersection(self._positive_words))
        negative_count = len(words.intersection(self._negative_words))
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        return max(min(sentiment_score, 1.0), -1.0)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(min(score, 100.0), 0.0)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified approach)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF."""
        if not self._tfidf_vectorizer:
            return []
        
        try:
            # Transform text
            tfidf_matrix = self._tfidf_vectorizer.transform([text])
            feature_names = self._tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 10 keywords
            return [keyword for keyword, score in keyword_scores[:10] if score > 0]
            
        except Exception as e:
            self.logger.warning(f"Failed to extract keywords: {e}")
            return []

