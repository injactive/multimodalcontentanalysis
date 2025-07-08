"""
Domain models for the Multi-Modal Content Analysis API.

This module defines all data models used throughout the application,
including request/response models and internal data structures.

Author: Christian Kruschel
Version: 0.0.1
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, validator, ConfigDict
import uuid


class ContentType(str, Enum):
    """Supported content types."""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class EngagementLevel(str, Enum):
    """Engagement level categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AnalysisRequest(BaseModel):
    """Request model for content analysis."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content to analyze"
    )
    image_url: Optional[HttpUrl] = Field(
        None,
        description="URL of the image to analyze"
    )
    content_type: ContentType = Field(
        default=ContentType.MULTIMODAL,
        description="Type of content being analyzed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the analysis"
    )
    
    @validator('text')
    def validate_text_content(cls, v):
        """Validate text content."""
        if not v or v.isspace():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v.strip()


class TextFeatures(BaseModel):
    """Text analysis features."""
    model_config = ConfigDict(validate_assignment=True)
    
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score between -1 (negative) and 1 (positive)"
    )
    sentiment_label: str = Field(..., description="Sentiment label (positive/negative/neutral)")
    readability_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Readability score (0-100, higher is more readable)"
    )
    word_count: int = Field(..., ge=0, description="Number of words")
    character_count: int = Field(..., ge=0, description="Number of characters")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    hashtags: List[str] = Field(default_factory=list, description="Extracted hashtags")
    mentions: List[str] = Field(default_factory=list, description="Extracted mentions")
    emoji_count: int = Field(default=0, ge=0, description="Number of emojis")
    url_count: int = Field(default=0, ge=0, description="Number of URLs")


class ImageFeatures(BaseModel):
    """Image analysis features."""
    model_config = ConfigDict(validate_assignment=True)
    
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    aspect_ratio: float = Field(..., gt=0, description="Image aspect ratio (width/height)")
    file_size: int = Field(..., ge=0, description="Image file size in bytes")
    format: str = Field(..., description="Image format (jpg, png, etc.)")
    has_faces: bool = Field(default=False, description="Whether image contains faces")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors in hex format")
    brightness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Image brightness (0-1)"
    )
    contrast: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Image contrast (0-1)"
    )
    clip_embedding: Optional[List[float]] = Field(
        None,
        description="CLIP embedding vector (if available)"
    )


class MultiModalFeatures(BaseModel):
    """Combined multimodal features."""
    model_config = ConfigDict(validate_assignment=True)
    
    text_features: TextFeatures
    image_features: Optional[ImageFeatures] = None
    text_image_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Text-image similarity score using CLIP"
    )
    content_coherence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall content coherence score"
    )



class EngagementPrediction(BaseModel):
    """Engagement prediction results."""
    model_config = ConfigDict(validate_assignment=True)
    
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Engagement prediction score (0-100)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence (0-1)"
    )
    level: EngagementLevel = Field(..., description="Engagement level category")
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Contributing factors and their weights"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )


class AnalysisResponse(BaseModel):
    """Response model for content analysis."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )
    content_type: ContentType = Field(..., description="Type of content analyzed")
    features: MultiModalFeatures = Field(..., description="Extracted features")
    prediction: EngagementPrediction = Field(..., description="Engagement prediction")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    model_version: str = Field(default="2.0.0", description="Model version used")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(default="2.0.0", description="API version")
    uptime: float = Field(..., ge=0, description="Service uptime in seconds")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")


class ErrorResponse(BaseModel):
    """Error response model."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID if available"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )


class MetricsResponse(BaseModel):
    """Metrics response model."""
    model_config = ConfigDict(validate_assignment=True)
    
    total_requests: int = Field(..., ge=0, description="Total number of requests")
    successful_requests: int = Field(..., ge=0, description="Number of successful requests")
    failed_requests: int = Field(..., ge=0, description="Number of failed requests")
    average_processing_time: float = Field(..., ge=0, description="Average processing time")
    uptime: float = Field(..., ge=0, description="Service uptime in seconds")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")



# Additional models for comprehensive API functionality

class ModelInfo(BaseModel):
    """Model information."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type (e.g., 'jinaclip-v2')")
    loaded: bool = Field(..., description="Whether model is loaded")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch content analysis."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    items: List[AnalysisRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of content items to analyze"
    )
    batch_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch identifier"
    )


class BatchAnalysisResponse(BaseModel):
    """Response model for batch content analysis."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    batch_id: str = Field(..., description="Batch identifier")
    total_items: int = Field(..., ge=0, description="Total number of items")
    successful_items: int = Field(..., ge=0, description="Number of successfully processed items")
    failed_items: int = Field(..., ge=0, description="Number of failed items")
    results: List[AnalysisResponse] = Field(..., description="Analysis results")
    errors: List[ErrorResponse] = Field(default_factory=list, description="Processing errors")
    total_processing_time: float = Field(..., ge=0, description="Total processing time")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Batch completion timestamp"
    )


class ConfigurationUpdate(BaseModel):
    """Model for configuration updates."""
    model_config = ConfigDict(validate_assignment=True)
    
    model_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration updates"
    )
    api_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="API configuration updates"
    )
    logging_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Logging configuration updates"
    )


class SystemStatus(BaseModel):
    """System status information."""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(..., description="Component status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    disk_usage: Dict[str, float] = Field(..., description="Disk usage statistics")
    active_connections: int = Field(..., ge=0, description="Number of active connections")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )


class ExperimentConfig(BaseModel):
    """MLflow experiment configuration."""
    model_config = ConfigDict(validate_assignment=True)
    
    experiment_name: str = Field(..., description="Experiment name")
    run_name: Optional[str] = Field(None, description="Run name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Experiment tags")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Experiment parameters")
    tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI")


class PredictionInterval(BaseModel):
    """Prediction interval for uncertainty quantification."""
    model_config = ConfigDict(validate_assignment=True)
    
    lower_bound: float = Field(..., description="Lower bound of prediction interval")
    upper_bound: float = Field(..., description="Upper bound of prediction interval")
    confidence_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level (e.g., 0.95 for 95%)"
    )
    method: str = Field(..., description="Method used for interval calculation")


class EnhancedEngagementPrediction(EngagementPrediction):
    """Enhanced engagement prediction with uncertainty quantification."""
    model_config = ConfigDict(validate_assignment=True)
    
    prediction_interval: Optional[PredictionInterval] = Field(
        None,
        description="Prediction interval for uncertainty quantification"
    )
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores"
    )
    model_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction"
    )
    alternative_predictions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative prediction scenarios"
    )


class ContentOptimizationSuggestion(BaseModel):
    """Content optimization suggestion."""
    model_config = ConfigDict(validate_assignment=True)
    
    category: str = Field(..., description="Optimization category")
    suggestion: str = Field(..., description="Optimization suggestion")
    impact_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected impact score"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in suggestion"
    )
    implementation_difficulty: str = Field(
        ...,
        description="Implementation difficulty (easy/medium/hard)"
    )


class ComprehensiveAnalysisResponse(AnalysisResponse):
    """Comprehensive analysis response with additional insights."""
    model_config = ConfigDict(validate_assignment=True)
    
    enhanced_prediction: EnhancedEngagementPrediction = Field(
        ...,
        description="Enhanced engagement prediction with uncertainty"
    )
    optimization_suggestions: List[ContentOptimizationSuggestion] = Field(
        default_factory=list,
        description="Content optimization suggestions"
    )
    comparative_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparative analysis with similar content"
    )
    trend_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Trend analysis and predictions"
    )


# Validation functions and utilities

def validate_engagement_score(score: float) -> float:
    """Validate engagement score is within valid range."""
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"Engagement score must be between 0 and 100, got {score}")
    return score


def validate_confidence_score(confidence: float) -> float:
    """Validate confidence score is within valid range."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence score must be between 0 and 1, got {confidence}")
    return confidence


def validate_sentiment_score(sentiment: float) -> float:
    """Validate sentiment score is within valid range."""
    if not -1.0 <= sentiment <= 1.0:
        raise ValueError(f"Sentiment score must be between -1 and 1, got {sentiment}")
    return sentiment


# Export all models for easy importing
__all__ = [
    # Enums
    "ContentType",
    "EngagementLevel",
    
    # Core models
    "AnalysisRequest",
    "TextFeatures", 
    "ImageFeatures",
    "MultiModalFeatures",
    "EngagementPrediction",
    "AnalysisResponse",
    
    # System models
    "HealthCheckResponse",
    "ErrorResponse",
    "MetricsResponse",
    "ModelInfo",
    "SystemStatus",
    
    # Batch processing
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    
    # Configuration
    "ConfigurationUpdate",
    "ExperimentConfig",
    
    # Enhanced models
    "PredictionInterval",
    "EnhancedEngagementPrediction",
    "ContentOptimizationSuggestion",
    "ComprehensiveAnalysisResponse",
    
    # Validation functions
    "validate_engagement_score",
    "validate_confidence_score", 
    "validate_sentiment_score",
]

