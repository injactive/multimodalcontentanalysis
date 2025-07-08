"""
Unit tests for the Feature Extraction Service.

This module provides comprehensive tests for text and image feature extraction,
multimodal analysis, and MLflow integration with performance monitoring.

Author: Christian Kruschel
Version: 0.0.1
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.services.feature_extraction import FeatureExtractionService
from src.models.domain import (
    AnalysisRequest, TextFeatures, ImageFeatures, 
    MultiModalFeatures, ContentType
)
from src.utils.http_client import AsyncHTTPClient, DownloadError, SecurityError
from src.utils.mlflow_tracker import MLflowTracker
from config.settings import APISettings

from tests.conftest import (
    assert_valid_sentiment_score, assert_valid_readability_score,
    assert_valid_engagement_score, assert_valid_confidence_score
)


class TestFeatureExtractionService:
    """Test suite for FeatureExtractionService."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(
        self,
        mock_http_client: AsyncHTTPClient,
        mock_mlflow_tracker: MLflowTracker,
        test_settings: APISettings
    ):
        """Test service initialization."""
        service = FeatureExtractionService(
            http_client=mock_http_client,
            mlflow_tracker=mock_mlflow_tracker,
            settings=test_settings
        )
        
        assert not service._initialized
        
        await service.initialize()
        
        assert service._initialized
        assert service._tfidf_vectorizer is not None
        
        await service.cleanup()
        assert not service._initialized
    
    @pytest.mark.asyncio
    async def test_text_feature_extraction_positive(
        self,
        feature_service: FeatureExtractionService,
        positive_text_samples: List[str]
    ):
        """Test text feature extraction with positive sentiment."""
        for text in positive_text_samples:
            request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
            
            features = await feature_service.extract_features(request, "test_request_1")
            
            # Validate text features
            text_features = features.text_features
            assert isinstance(text_features, TextFeatures)
            
            # Sentiment should be positive
            assert_valid_sentiment_score(text_features.sentiment_score)
            assert text_features.sentiment_score > 0
            assert text_features.sentiment_label == "positive"
            
            # Basic text statistics
            assert text_features.word_count > 0
            assert text_features.character_count > 0
            assert_valid_readability_score(text_features.readability_score)
            
            # Keywords should be extracted
            assert isinstance(text_features.keywords, list)
    
    @pytest.mark.asyncio
    async def test_text_feature_extraction_negative(
        self,
        feature_service: FeatureExtractionService,
        negative_text_samples: List[str]
    ):
        """Test text feature extraction with negative sentiment."""
        for text in negative_text_samples:
            request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
            
            features = await feature_service.extract_features(request, "test_request_2")
            
            text_features = features.text_features
            
            # Sentiment should be negative
            assert_valid_sentiment_score(text_features.sentiment_score)
            assert text_features.sentiment_score < 0
            assert text_features.sentiment_label == "negative"
    
    @pytest.mark.asyncio
    async def test_text_feature_extraction_neutral(
        self,
        feature_service: FeatureExtractionService,
        neutral_text_samples: List[str]
    ):
        """Test text feature extraction with neutral sentiment."""
        for text in neutral_text_samples:
            request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
            
            features = await feature_service.extract_features(request, "test_request_3")
            
            text_features = features.text_features
            
            # Sentiment should be neutral
            assert_valid_sentiment_score(text_features.sentiment_score)
            assert abs(text_features.sentiment_score) <= 0.1
            assert text_features.sentiment_label == "neutral"
    
    @pytest.mark.asyncio
    async def test_hashtag_and_mention_extraction(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test extraction of hashtags and mentions."""
        text = "Check out this #amazing product! Thanks @john_doe for the recommendation! #love #awesome"
        request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
        
        features = await feature_service.extract_features(request, "test_request_4")
        text_features = features.text_features
        
        # Check hashtags
        assert len(text_features.hashtags) == 3
        assert "#amazing" in text_features.hashtags
        assert "#love" in text_features.hashtags
        assert "#awesome" in text_features.hashtags
        
        # Check mentions
        assert len(text_features.mentions) == 1
        assert "@john_doe" in text_features.mentions
    
    @pytest.mark.asyncio
    async def test_emoji_counting(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test emoji counting functionality."""
        text = "I love this! 😍😊🎉 So amazing! 🚀"
        request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
        
        features = await feature_service.extract_features(request, "test_request_5")
        text_features = features.text_features
        
        # Should detect emojis
        assert text_features.emoji_count >= 3  # At least 3 emojis
    
    @pytest.mark.asyncio
    async def test_url_detection(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test URL detection in text."""
        text = "Check out https://example.com and http://test.org for more info!"
        request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
        
        features = await feature_service.extract_features(request, "test_request_6")
        text_features = features.text_features
        
        # Should detect URLs
        assert text_features.url_count == 2
    
    @pytest.mark.asyncio
    async def test_multimodal_feature_extraction(
        self,
        feature_service: FeatureExtractionService,
        sample_analysis_request: AnalysisRequest
    ):
        """Test multimodal feature extraction with text and image."""
        features = await feature_service.extract_features(
            sample_analysis_request, 
            "test_request_7"
        )
        
        # Should have both text and image features
        assert isinstance(features, MultiModalFeatures)
        assert isinstance(features.text_features, TextFeatures)
        assert isinstance(features.image_features, ImageFeatures)
        
        # Should have text-image similarity
        assert features.text_image_similarity is not None
        assert 0.0 <= features.text_image_similarity <= 1.0
        
        # Should have content coherence
        assert 0.0 <= features.content_coherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_image_feature_extraction_error_handling(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test image feature extraction with download errors."""
        # Mock HTTP client to raise an error
        feature_service.http_client.download_image = AsyncMock(
            side_effect=DownloadError("Failed to download image")
        )
        
        request = AnalysisRequest(
            text="Test text",
            image_url="https://example.com/invalid.jpg",
            content_type=ContentType.MULTIMODAL
        )
        
        features = await feature_service.extract_features(request, "test_request_8")
        
        # Should handle error gracefully
        assert features.image_features is not None
        assert features.image_features.width == 0  # Default error values
        assert features.image_features.height == 0
    
    @pytest.mark.asyncio
    async def test_feature_caching(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test feature caching functionality."""
        text = "This is a test for caching functionality."
        request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
        
        # First extraction
        start_time = time.time()
        features1 = await feature_service.extract_features(request, "test_request_9a")
        first_duration = time.time() - start_time
        
        # Second extraction (should use cache)
        start_time = time.time()
        features2 = await feature_service.extract_features(request, "test_request_9b")
        second_duration = time.time() - start_time
        
        # Results should be identical
        assert features1.text_features.sentiment_score == features2.text_features.sentiment_score
        assert features1.text_features.word_count == features2.text_features.word_count
        
        # Second call should be faster (cached)
        assert second_duration < first_duration
    
    @pytest.mark.asyncio
    async def test_readability_calculation(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test readability score calculation."""
        # Simple text (should have high readability)
        simple_text = "This is easy to read. Short sentences work well."
        simple_request = AnalysisRequest(text=simple_text, content_type=ContentType.TEXT)
        
        # Complex text (should have lower readability)
        complex_text = """
        The implementation of sophisticated algorithmic methodologies necessitates
        comprehensive understanding of multidimensional computational paradigms
        that facilitate optimization of performance metrics across heterogeneous
        distributed systems architectures.
        """
        complex_request = AnalysisRequest(text=complex_text, content_type=ContentType.TEXT)
        
        simple_features = await feature_service.extract_features(simple_request, "test_request_10a")
        complex_features = await feature_service.extract_features(complex_request, "test_request_10b")
        
        # Simple text should have higher readability
        assert simple_features.text_features.readability_score > complex_features.text_features.readability_score
        
        # Both should be valid scores
        assert_valid_readability_score(simple_features.text_features.readability_score)
        assert_valid_readability_score(complex_features.text_features.readability_score)
    
    @pytest.mark.asyncio
    async def test_content_coherence_calculation(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test content coherence calculation."""
        # High coherence: good text + image
        high_coherence_request = AnalysisRequest(
            text="Beautiful sunset photo! Amazing colors and perfect lighting! 🌅",
            image_url="https://example.com/sunset.jpg",
            content_type=ContentType.MULTIMODAL
        )
        
        # Low coherence: poor text quality
        low_coherence_request = AnalysisRequest(
            text="a",  # Very short, low quality text
            image_url="https://example.com/image.jpg",
            content_type=ContentType.MULTIMODAL
        )
        
        high_features = await feature_service.extract_features(high_coherence_request, "test_request_11a")
        low_features = await feature_service.extract_features(low_coherence_request, "test_request_11b")
        
        # High coherence should score better
        assert high_features.content_coherence > low_features.content_coherence
        
        # Both should be valid scores
        assert 0.0 <= high_features.content_coherence <= 1.0
        assert 0.0 <= low_features.content_coherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_mlflow_integration(
        self,
        feature_service: FeatureExtractionService,
        sample_analysis_request: AnalysisRequest
    ):
        """Test MLflow integration during feature extraction."""
        # Extract features (should trigger MLflow logging)
        features = await feature_service.extract_features(
            sample_analysis_request,
            "test_request_12"
        )
        
        # Verify MLflow tracker was called
        mlflow_tracker = feature_service.mlflow_tracker
        
        # Check that MLflow methods were called
        assert mlflow_tracker.start_run.called
        assert mlflow_tracker.log_metrics.called
        assert mlflow_tracker.log_params.called
        assert mlflow_tracker.end_run.called
    
    @pytest.mark.asyncio
    async def test_performance_with_multiple_requests(
        self,
        feature_service: FeatureExtractionService,
        performance_test_requests: List[AnalysisRequest]
    ):
        """Test performance with multiple concurrent requests."""
        start_time = time.time()
        
        # Process requests concurrently
        tasks = [
            feature_service.extract_features(request, f"perf_test_{i}")
            for i, request in enumerate(performance_test_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # All requests should complete successfully
        assert len(results) == len(performance_test_requests)
        
        # Each result should be valid
        for features in results:
            assert isinstance(features, MultiModalFeatures)
            assert isinstance(features.text_features, TextFeatures)
        
        # Performance should be reasonable (less than 1 second per request on average)
        avg_time_per_request = total_time / len(performance_test_requests)
        assert avg_time_per_request < 1.0
        
        print(f"Processed {len(performance_test_requests)} requests in {total_time:.2f}s")
        print(f"Average time per request: {avg_time_per_request:.3f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_text(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test error handling with invalid text input."""
        # Empty text should be handled gracefully
        with pytest.raises(ValueError):
            request = AnalysisRequest(text="", content_type=ContentType.TEXT)
    
    @pytest.mark.asyncio
    async def test_security_validation_malicious_url(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test security validation for malicious URLs."""
        # Mock HTTP client to raise security error
        feature_service.http_client.download_image = AsyncMock(
            side_effect=SecurityError("Blocked domain")
        )
        
        request = AnalysisRequest(
            text="Test text",
            image_url="https://malicious-site.com/image.jpg",
            content_type=ContentType.MULTIMODAL
        )
        
        # Should handle security error gracefully
        features = await feature_service.extract_features(request, "test_request_13")
        
        # Should return default image features on security error
        assert features.image_features is not None
        assert features.image_features.width == 0
    
    @pytest.mark.asyncio
    async def test_keyword_extraction_quality(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test quality of keyword extraction."""
        text = """
        Machine learning and artificial intelligence are transforming
        the technology industry. Deep learning models and neural networks
        are enabling breakthrough innovations in computer vision and
        natural language processing applications.
        """
        
        request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
        features = await feature_service.extract_features(request, "test_request_14")
        
        keywords = features.text_features.keywords
        
        # Should extract relevant keywords
        assert len(keywords) > 0
        
        # Keywords should be relevant to the content
        tech_keywords = [
            'machine', 'learning', 'artificial', 'intelligence',
            'deep', 'neural', 'networks', 'computer', 'vision',
            'language', 'processing'
        ]
        
        # At least some tech keywords should be found
        found_tech_keywords = [kw for kw in keywords if any(tech in kw.lower() for tech in tech_keywords)]
        assert len(found_tech_keywords) > 0
    
    @pytest.mark.asyncio
    async def test_syllable_counting_accuracy(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test syllable counting accuracy for readability calculation."""
        # Test known syllable counts
        test_words = {
            "cat": 1,
            "hello": 2,
            "beautiful": 3,
            "information": 4,
            "university": 5
        }
        
        for word, expected_syllables in test_words.items():
            actual_syllables = feature_service._count_syllables(word)
            
            # Allow some tolerance for simplified algorithm
            assert abs(actual_syllables - expected_syllables) <= 1
    
    @pytest.mark.asyncio
    async def test_sentiment_edge_cases(
        self,
        feature_service: FeatureExtractionService
    ):
        """Test sentiment analysis with edge cases."""
        edge_cases = [
            ("", 0.0),  # Empty text (after validation)
            ("neutral text without sentiment words", 0.0),  # Neutral
            ("amazing terrible good bad", 0.0),  # Mixed sentiment
            ("AMAZING FANTASTIC WONDERFUL!!!", 1.0),  # All caps positive
        ]
        
        for text, expected_range in edge_cases:
            if text:  # Skip empty text as it would fail validation
                request = AnalysisRequest(text=text, content_type=ContentType.TEXT)
                features = await feature_service.extract_features(request, f"edge_case_{hash(text)}")
                
                sentiment = features.text_features.sentiment_score
                assert_valid_sentiment_score(sentiment)
                
                # For mixed sentiment, should be close to neutral
                if "amazing terrible good bad" in text:
                    assert abs(sentiment) < 0.5

