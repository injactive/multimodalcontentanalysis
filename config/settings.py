"""
Configuration settings for the Multi-Modal Content Analysis API.

This module provides centralized configuration management using Pydantic Settings
with environment variable support and validation.

Author: Christian Kruschel
Version: 0.0.1
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Application Settings
    app_name: str = Field(default="Multi-Modal Content Analysis API", description="Application name")
    app_version: str = Field(default="0.0.1", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(default=["GET", "POST"], description="Allowed CORS methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute per IP")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Security Settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Max request size in bytes (10MB)")
    
    # ML Model Configuration - JinaCLIP v2
    model_name: str = Field(
        default="jinaai/jina-clip-v2", 
        description="JinaCLIP v2 model name for multimodal embeddings"
    )
    model_cache_size: int = Field(default=100, description="Model cache size for embeddings")
    model_timeout: float = Field(default=30.0, description="Model inference timeout in seconds")
    embedding_dimension: int = Field(default=768, description="JinaCLIP v2 embedding dimension")
    max_image_size: int = Field(default=512, description="Maximum image size for processing")
    trust_remote_code: bool = Field(default=True, description="Trust remote code for JinaCLIP v2")
    
    # Feature Extraction Settings
    max_text_length: int = Field(default=5000, description="Maximum text length for analysis")
    max_image_file_size: int = Field(default=5 * 1024 * 1024, description="Maximum image file size in bytes (5MB)")
    supported_image_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"],
        description="Supported image formats"
    )
    
    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="multimodal-content-analysis", description="MLflow experiment name")
    mlflow_enable_logging: bool = Field(default=True, description="Enable MLflow logging")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Monitoring Settings
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8001, description="Metrics server port")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = APISettings()


def get_settings() -> APISettings:
    """Get application settings."""
    return settings

