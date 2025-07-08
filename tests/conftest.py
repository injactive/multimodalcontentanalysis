"""
Pytest configuration and fixtures for the Multi-Modal Content Analysis API tests.

This module provides shared fixtures, test configuration, and utilities
for comprehensive testing with MLflow integration.

Author: Christian Kruschel
Version: 0.0.1
"""

import asyncio
import pytest
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import os

from fastapi.testclient import TestClient
import httpx

from src.models.domain import AnalysisRequest, ContentType
from src.services.feature_extraction import FeatureExtractionService
from src.utils.http_client import AsyncHTTPClient
from src.utils.mlflow_tracker import MLflowTracker
from src.utils.logger import setup_logging, get_logger
from config.settings import APISettings


# Setup test logging
setup_logging(level="DEBUG", format_type="text")
logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> APISettings:
    """Create test settings."""
    return APISettings(
        debug=True,
        mlflow_tracking_uri="./test_mlruns",
        mlflow_experiment_name="test-multimodal-analysis",
        mlflow_enable_logging=True,
        log_level="DEBUG",
        max_request_size=1024 * 1024,  # 1MB for tests
        model_timeout=10.0
    )


@pytest.fixture
async def mock_http_client() -> AsyncHTTPClient:
    """Create a mock HTTP client for testing."""
    client = AsyncMock(spec=AsyncHTTPClient)
    
    # Mock image download
    client.download_image.return_value = b"fake_image_data"
    
    # Mock JSON requests
    client.get_json.return_value = {"status": "ok"}
    client.post_json.return_value = {"result": "success"}
    
    # Mock metrics
    client.metrics = {
        "request_count": 0,
        "total_bytes_downloaded": 0,
        "session_active": True
    }
    
    return client


@pytest.fixture
async def mock_mlflow_tracker(test_settings: APISettings) -> MLflowTracker:
    """Create a mock MLflow tracker for testing."""
    tracker = MLflowTracker(
        tracking_uri=test_settings.mlflow_tracking_uri,
        experiment_name=test_settings.mlflow_experiment_name,
        enable_logging=True
    )
    
    # Mock the MLflow operations
    tracker._client = MagicMock()
    tracker._experiment_id = "test_experiment_123"
    tracker._initialized = True
    
    # Mock run operations
    async def mock_start_run(*args, **kwargs):
        return "test_run_123"
    
    async def mock_log_params(*args, **kwargs):
        pass
    
    async def mock_log_metrics(*args, **kwargs):
        pass
    
    async def mock_end_run(*args, **kwargs):
        pass
    
    tracker.start_run = mock_start_run
    tracker.log_params = mock_log_params
    tracker.log_metrics = mock_log_metrics
    tracker.end_run = mock_end_run
    
    return tracker


@pytest.fixture
async def feature_service(
    mock_http_client: AsyncHTTPClient,
    mock_mlflow_tracker: MLflowTracker,
    test_settings: APISettings
) -> FeatureExtractionService:
    """Create a feature extraction service for testing."""
    service = FeatureExtractionService(
        http_client=mock_http_client,
        mlflow_tracker=mock_mlflow_tracker,
        settings=test_settings
    )
    
    await service.initialize()
    yield service
    await service.cleanup()


@pytest.fixture
def sample_analysis_request() -> AnalysisRequest:
    """Create a sample analysis request for testing."""
    return AnalysisRequest(
        text="This is an amazing product! I love it so much! 😍 #awesome #love",
        image_url="https://example.com/image.jpg",
        content_type=ContentType.MULTIMODAL,
        metadata={"test": True}
    )


@pytest.fixture
def sample_text_only_request() -> AnalysisRequest:
    """Create a sample text-only analysis request."""
    return AnalysisRequest(
        text="Just a simple text post without any images.",
        content_type=ContentType.TEXT
    )


@pytest.fixture
def sample_long_text_request() -> AnalysisRequest:
    """Create a sample request with long text."""
    long_text = " ".join([
        "This is a very long text that contains multiple sentences.",
        "It has various emotional expressions and should trigger different",
        "sentiment analysis results. The text includes positive words like",
        "amazing, wonderful, and fantastic. It also has some neutral content",
        "and potentially negative aspects. This helps test the robustness",
        "of our text analysis pipeline and ensures that longer content",
        "is processed correctly without any issues."
    ])
    
    return AnalysisRequest(
        text=long_text,
        content_type=ContentType.TEXT
    )


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for API testing."""
    async with httpx.AsyncClient() as client:
        yield client


class MockImageResponse:
    """Mock image response for testing."""
    
    def __init__(self, content: bytes = b"fake_image_data", status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers = {
            "content-type": "image/jpeg",
            "content-length": str(len(content))
        }
    
    async def read(self) -> bytes:
        return self.content


@pytest.fixture
def mock_image_response() -> MockImageResponse:
    """Create a mock image response."""
    return MockImageResponse()


# Test data fixtures
@pytest.fixture
def positive_text_samples() -> list[str]:
    """Sample positive texts for testing."""
    return [
        "I love this amazing product! It's fantastic! 😍",
        "What a wonderful day! Everything is perfect!",
        "This is the best experience ever! Highly recommended!",
        "Absolutely brilliant work! Outstanding quality!",
        "So happy with this purchase! Excellent service!"
    ]


@pytest.fixture
def negative_text_samples() -> list[str]:
    """Sample negative texts for testing."""
    return [
        "This is terrible! I hate it so much! 😡",
        "Worst experience ever! Very disappointing!",
        "Awful quality! Complete waste of money!",
        "Horrible service! Never buying again!",
        "Disgusting product! Totally frustrated!"
    ]


@pytest.fixture
def neutral_text_samples() -> list[str]:
    """Sample neutral texts for testing."""
    return [
        "This is a product description with technical specifications.",
        "The weather today is partly cloudy with mild temperatures.",
        "Meeting scheduled for 2 PM in conference room A.",
        "Please review the attached document and provide feedback.",
        "The system will undergo maintenance this weekend."
    ]


# Performance testing fixtures
@pytest.fixture
def performance_test_requests() -> list[AnalysisRequest]:
    """Create multiple requests for performance testing."""
    requests = []
    
    for i in range(10):
        requests.append(AnalysisRequest(
            text=f"Performance test request {i} with some content to analyze.",
            content_type=ContentType.TEXT,
            metadata={"test_id": i, "performance_test": True}
        ))
    
    return requests


# MLflow testing utilities
@pytest.fixture
def mlflow_test_cleanup():
    """Cleanup MLflow test artifacts after tests."""
    yield
    
    # Cleanup test MLflow directory
    test_mlruns_dir = "./test_mlruns"
    if os.path.exists(test_mlruns_dir):
        shutil.rmtree(test_mlruns_dir)


# Async testing utilities
def pytest_configure(config):
    """Configure pytest for async testing."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


# Custom assertions
def assert_valid_sentiment_score(score: float):
    """Assert that sentiment score is valid."""
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def assert_valid_readability_score(score: float):
    """Assert that readability score is valid."""
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def assert_valid_engagement_score(score: float):
    """Assert that engagement score is valid."""
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def assert_valid_confidence_score(score: float):
    """Assert that confidence score is valid."""
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

