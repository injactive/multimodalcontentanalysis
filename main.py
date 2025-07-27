"""
Multi-Modal Content Analysis API

FastAPI application that analyzes influencer posts (text + images)
to predict engagement potential using pre-trained models

Additional Components:
- MLflow tracking
- structured logging
- Health checks and metrics
- JinaCLIP v2 - the latest SOTA multimodal model

Author: Christian Kruschel
Version: 0.0.1
"""

# Standard library imports
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

# FastAPI core imports
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# Domain models and ASPI schemas
from src.models.domain import (
    AnalysisRequest, AnalysisResponse, HealthCheckResponse, 
    ErrorResponse, MetricsResponse, EngagementPrediction,
    MultiModalFeatures
)

# Service for multimodal (text + image) feature extraction
from src.services.jinaclip_feature_extraction import JinaCLIPFeatureExtractionService

from src.services.engagement_prediction import generate_engagement_prediction

# Dependency injection and configuration utilities
from src.core.dependencies import (
    get_service_container, get_feature_service, get_mlflow_tracker,
    get_settings_dependency, get_uptime, health_check, lifespan_manager
)

# Logging utilities and decorators
from src.utils.logger import (
    setup_logging, get_logger, with_correlation_id, with_performance_logging
)
from src.utils.test_cases import complete_testcase_request

# Configuration settings
from config.settings import APISettings

from src.utils.mlflow_tracker import MLflowTracker


# Initialize structured logging
setup_logging()
logger = get_logger(__name__)

# App lifecycle management: setup and teardown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the API."""
    logger.info("Starting Multi-Modal Content Analysis API v2.0...")
    
    async with lifespan_manager() as container:
        app.state.service_container = container
        logger.info("Application startup completed")
        yield
    
    logger.info("Application shutdown completed")


# Instantiate FastAPI app with documentation and metadata
app = FastAPI(
    title="Multi-Modal Content Analysis API",
    description="""
    An API that analyzes influencer posts (text + images) 
    to predict engagement potential using JinaCLIP v2 - the latest state-of-the-art 
    multimodal model with superior performance over traditional CLIP.
    """,
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware for securizy, logging, and caching
def setup_middleware(app: FastAPI, settings: APISettings):
    """Configure CORS and security middleware."""
    
    # Cross-Origin Resource Sharing: Allows requests from different domains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Trusted host middleware (security)
    # Check host header
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # !! Configure appropriately for production
    )

# Global exception handler for HTTP errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles HTTP exceptions and returns standardized error responses."""
    logger.error(
        f"HTTP exception: {exc.status_code}",
        detail=exc.detail,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.error(
        "Request validation error",
        errors=exc.errors(),
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()}
        ).model_dump()
    )

# Catch-all handler for uncaught exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        error=str(exc),
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An internal error occurred",
            details={"type": type(exc).__name__}
        ).model_dump()
    )


# Root endpoint - provides metadata and links
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing basic API info."""
    return {
        "name": "Multi-Modal Content Analysis API",
        "version": "0.0.1",
        "description": "Multimodal content analysis for social media",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint for monitoring system status
@app.get("/health", response_model=HealthCheckResponse)
async def health_check_endpoint(
    uptime: float = Depends(get_uptime)
):
    """Returns health status, uptime, and model loading state."""
    health_data = health_check()
    
    return HealthCheckResponse(
        status=health_data["status"],
        uptime=uptime,
        models_loaded=health_data["models_loaded"]
    )

# Metrics endpoint - reports request counts and performance stats
@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint(
    feature_service: JinaCLIPFeatureExtractionService = Depends(get_feature_service),
    mlflow_tracker = Depends(get_mlflow_tracker),
    uptime: float = Depends(get_uptime)
):
    """Returns aggregated performance and tracking metrics from MLflow."""
    
    # Get experiment metrics from MLflow
    experiment_metrics = await mlflow_tracker.get_experiment_metrics()
    
    return MetricsResponse(
        total_requests=experiment_metrics.get("total_runs", 0),
        successful_requests=experiment_metrics.get("successful_runs", 0),
        failed_requests=experiment_metrics.get("failed_runs", 0),
        average_processing_time=experiment_metrics.get("avg_processing_time", 0.0),
        uptime=uptime,
        models_loaded=True
    )

# Main endpoint for analyzing a social media post
@app.post("/analyze-post", response_model=AnalysisResponse)
@with_correlation_id()
@with_performance_logging("analyze_post")
async def analyze_post(
    request: AnalysisRequest,
    feature_service: JinaCLIPFeatureExtractionService = Depends(get_feature_service),
    settings: APISettings = Depends(get_settings_dependency)
):
    """
    Analyze a social media post for engagement potential.
    
    This endpoint performs comprehensive multimodal analysis including:
    - Text sentiment analysis and feature extraction
    - Image analysis (if image URL provided)
    - Text-image alignment scoring
    - Engagement prediction with confidence intervals
    - MLflow experiment tracking
    
    Args:
        request: Analysis request containing text and optional image URL
        
    Returns:
        Comprehensive analysis results with engagement prediction
        
    Raises:
        HTTPException: If analysis fails or input is invalid
    """

    if request.text is None and request.image_url is None:
        return AnalysisResponse()

    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(
        "Starting content analysis",
        request_id=request_id,
        content_type=request.content_type,
        text_length=len(request.text),
        has_image=request.image_url is not None
    )

    request = complete_testcase_request(request)

    try:
        # Step 1: Extract multimodal features
        features = await feature_service.extract_features(request, request_id)
        
        # Step 2: Predict engagement level from features
        engagement_prediction = generate_engagement_prediction(features)
        
        processing_time = time.time() - start_time
        
        # Step 3: Log full analysis to MLflow
        await _log_comprehensive_analysis_to_mlflow(
            request=request,
            request_id=request_id,
            features=features,
            engagement_prediction=engagement_prediction,
            processing_time=processing_time
        )
        
        # Step 4: Construct API response
        response = AnalysisResponse(
            request_id=request_id,
            content_type=request.content_type,
            features=features,
            prediction=engagement_prediction,
            processing_time=processing_time
        )
        
        logger.info(
            "Content analysis completed",
            request_id=request_id,
            processing_time=processing_time,
            engagement_score=engagement_prediction.score,
            confidence=engagement_prediction.confidence
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Content analysis failed: {e}",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


async def _log_comprehensive_analysis_to_mlflow(
    request: AnalysisRequest,
    request_id: str,
    features: MultiModalFeatures,
    engagement_prediction: EngagementPrediction,
    processing_time: float
):
    """
    Log comprehensive analysis results to MLflow for complete observability.
    
    This function creates a single MLflow run that contains all aspects of the analysis:
    - Input parameters and metadata
    - Extracted features (text, image, multimodal)
    - Engagement prediction with confidence intervals
    - Processing performance metrics
    - Feature contributions and model metadata
    """
    try:
        # Get MLflow tracker from dependency injection
        container = get_service_container()
        mlflow_tracker = container.get_service('mlflow_tracker')
        
        print("#")
        # Start comprehensive analysis run
        run_id = await mlflow_tracker.start_run(
            run_name=f"comprehensive_analysis_{request_id}",
            tags={
                "request_id": request_id,
                "analysis_type": "comprehensive",
                "content_type": str(request.content_type),
                "has_image": request.image_url is not None,
                "model_version": "jinaclip-v2",
                "api_version": "0.0.1"
            }
        )
        print("##")
        # Prepare request data for logging
        request_data = {
            "text": request.text,
            "content_type": str(request.content_type),
            "image_url": request.image_url,
            "request_id": request_id
        }
        print("###")
        # Prepare features data
        features_data = {
            "text_features": features.text_features.__dict__ if features.text_features else {},
            "image_features": features.image_features.__dict__ if features.image_features else {},
            "content_coherence": features.content_coherence,
            "text_image_similarity": features.text_image_similarity,
            "processing_time": processing_time
        }
        print("####")
        # Prepare engagement prediction data
        engagement_data = {
            "score": engagement_prediction.score,
            "confidence": engagement_prediction.confidence,
            "level": engagement_prediction.level.value if hasattr(engagement_prediction.level, 'value') else str(engagement_prediction.level),
            "factors": getattr(engagement_prediction, 'factors', {})
        }
        print("#####")
        # Prepare processing metrics
        processing_metrics = {
            "total_time": processing_time,
            "feature_time": processing_time * 0.7,  # Estimated feature extraction time
            "prediction_time": processing_time * 0.2,  # Estimated prediction time
            "response_time": processing_time * 0.1  # Estimated response preparation time
        }
        print("######")
        # Log comprehensive analysis
        await mlflow_tracker.log_comprehensive_analysis(
            run_id=run_id,
            request_data=request_data,
            features=features_data,
            engagement_prediction=engagement_data,
            processing_metrics=processing_metrics
        )
        print("#######")
        # Calculate feature contributions for detailed logging
        feature_contributions = _calculate_feature_contributions(features, engagement_prediction)
        
        # Prepare model metadata
        model_metadata = {
            "model_name": "jinaclip-v2",
            "embedding_dimension": 768,
            "prediction_algorithm": "weighted_combination",
            "confidence_method": "feature_quality_based",
            "version": "0.0.1"
        }
        print("########")
        # Log detailed engagement prediction
        await mlflow_tracker.log_engagement_prediction_details(
            run_id=run_id,
            prediction=engagement_data,
            feature_contributions=feature_contributions,
            model_metadata=model_metadata
        )
        print("#########")
        # End the run
        await mlflow_tracker.end_run(run_id, "FINISHED")
        print("##########")
        logger.debug(
            "Comprehensive analysis logged to MLflow",
            request_id=request_id,
            run_id=run_id,
            engagement_score=engagement_prediction.score,
            confidence=engagement_prediction.confidence
        )
        
    except Exception as e:
        logger.error(f"Failed to log comprehensive analysis to MLflow: {e}")
        # Don't fail the main request if MLflow logging fails


def _calculate_feature_contributions(
    features: MultiModalFeatures,
    engagement_prediction: EngagementPrediction
) -> Dict[str, float]:
    """
    Calculate individual feature contributions to the engagement score.
    
    This provides explainability for the ML model predictions.
    """
    contributions = {}
    
    if features.text_features:
        # Text feature contributions
        contributions["sentiment"] = features.text_features.sentiment_score * 0.3
        contributions["readability"] = (features.text_features.readability_score / 100.0) * 0.2
        contributions["word_count"] = min(features.text_features.word_count / 100.0, 1.0) * 0.15
        contributions["social_signals"] = min(
            (len(features.text_features.hashtags) + len(features.text_features.mentions) + features.text_features.emoji_count) / 5.0,
            1.0
        ) * 0.2
    
    if features.image_features:
        # Image feature contributions
        contributions["image_quality"] = 0.1  # Base contribution for having an image
        contributions["text_image_alignment"] = (features.text_image_similarity or 0.0) * 0.05
    
    # Multimodal contributions
    contributions["content_coherence"] = features.content_coherence * 0.1
    
    return contributions


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = APISettings()
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        log_file=settings.log_file
    )
    
    return app


if __name__ == "__main__":
    settings = APISettings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True
    )

