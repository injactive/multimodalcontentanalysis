"""
Dependency injection system for the Multi-Modal Content Analysis API.

This module provides dependency injection for services, ensuring proper
separation of concerns and testability.
"""

from typing import Optional, Dict, Any
from functools import lru_cache
import time
from contextlib import asynccontextmanager

from fastapi import HTTPException, status
from src.services.jinaclip_feature_extraction import JinaCLIPFeatureExtractionService
from ..utils.logger import get_logger
from ..utils.mlflow_tracker import MLflowTracker
from ..utils.http_client import AsyncHTTPClient
from config.settings import get_settings, APISettings


logger = get_logger(__name__)


class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._startup_time = time.time()
    
    async def initialize(self):
        """Initialize all services."""
        if self._initialized:
            return
        
        logger.info("Initializing service container...")
        
        try:
            settings = get_settings()
            
            # Initialize HTTP client
            http_client = AsyncHTTPClient(
                timeout=settings.model_timeout,
                max_request_size=settings.max_request_size
            )
            self._services['http_client'] = http_client
            
            # Initialize MLflow tracker
            mlflow_tracker = MLflowTracker(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
                enable_logging=settings.mlflow_enable_logging
            )
            await mlflow_tracker.initialize()
            self._services['mlflow_tracker'] = mlflow_tracker
            
            # Initialize feature extraction service with JinaCLIP v2
            feature_service = JinaCLIPFeatureExtractionService(
                http_client=http_client,
                mlflow_tracker=mlflow_tracker,
                settings=settings
            )
            await feature_service.initialize()
            self._services['feature_service'] = feature_service
            
            self._initialized = True
            logger.info("Service container initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service container: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all services."""
        logger.info("Cleaning up service container...")
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                logger.debug(f"Cleaned up service: {service_name}")
            except Exception as e:
                logger.error(f"Error cleaning up service {service_name}: {e}")
        
        self._services.clear()
        self._initialized = False
        logger.info("Service container cleanup completed")
    
    def get_service(self, service_name: str) -> Any:
        """Get a service by name."""
        if not self._initialized:
            raise RuntimeError("Service container not initialized")
        
        service = self._services.get(service_name)
        if service is None:
            raise ValueError(f"Service '{service_name}' not found")
        
        return service
    
    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized
    
    @property
    def uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._startup_time


# Global service container
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global service container."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


@asynccontextmanager
async def lifespan_manager():
    """Manage application lifespan."""
    container = get_service_container()
    try:
        await container.initialize()
        yield container
    finally:
        await container.cleanup()


# Dependency functions for FastAPI
def get_feature_service() -> JinaCLIPFeatureExtractionService:
    """Get feature extraction service dependency."""
    container = get_service_container()
    if not container.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return container.get_service('feature_service')


def get_mlflow_tracker() -> MLflowTracker:
    """Get MLflow tracker dependency."""
    container = get_service_container()
    if not container.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return container.get_service('mlflow_tracker')


def get_http_client() -> AsyncHTTPClient:
    """Get HTTP client dependency."""
    container = get_service_container()
    if not container.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return container.get_service('http_client')


@lru_cache()
def get_settings_dependency() -> APISettings:
    """Get settings dependency (cached)."""
    return get_settings()


def get_uptime() -> float:
    """Get service uptime."""
    container = get_service_container()
    return container.uptime


def health_check() -> Dict[str, Any]:
    """Perform health check."""
    container = get_service_container()
    
    return {
        "status": "healthy" if container.is_initialized else "unhealthy",
        "uptime": container.uptime,
        "models_loaded": container.is_initialized,
        "services": list(container._services.keys()) if container.is_initialized else []
    }

