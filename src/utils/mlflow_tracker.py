"""
MLflow tracking utility for the Multi-Modal Content Analysis API.

This module provides comprehensive experiment tracking, model versioning,
and performance monitoring using MLflow.

Author: Christian Kruschel
Version: 0.0.1
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json
import uuid

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import structlog

from ..utils.logger import get_logger


logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model management."""
    
    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "multimodal-content-analysis",
        enable_logging: bool = True
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enable_logging = enable_logging
        self.logger = logger.bind(component="mlflow_tracker")
        
        self._client: Optional[MlflowClient] = None
        self._experiment_id: Optional[str] = None
        self._active_runs: Dict[str, Any] = {}
        self._initialized = False
        
        # Performance metrics
        self._total_runs = 0
        self._successful_runs = 0
        self._failed_runs = 0
        self._start_time = time.time()
    
    async def initialize(self):
        """Initialize MLflow tracking."""
        if self._initialized or not self.enable_logging:
            return
        
        try:
            self.logger.info("Initializing MLflow tracking...")
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Initialize client
            self._client = MlflowClient()
            
            # Create or get experiment
            try:
                experiment = self._client.get_experiment_by_name(self.experiment_name)
                if experiment:
                    self._experiment_id = experiment.experiment_id
                else:
                    self._experiment_id = self._client.create_experiment(
                        self.experiment_name,
                        tags={
                            "project": "multimodal-content-analysis",
                            "version": "0.0.1",
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to create/get experiment: {e}")
                # Create with unique name
                unique_name = f"{self.experiment_name}_{int(time.time())}"
                self._experiment_id = self._client.create_experiment(unique_name)
            
            self._initialized = True
            self.logger.info(
                "MLflow tracking initialized",
                experiment_id=self._experiment_id,
                tracking_uri=self.tracking_uri
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            self.enable_logging = False  # Disable logging on failure
    
    async def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags for the run
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        if not self.enable_logging or not self._initialized:
            return str(uuid.uuid4())  # Return dummy ID
        
        try:
            # Generate run name if not provided
            if not run_name:
                run_name = f"run_{int(time.time())}"
            
            # Prepare tags
            run_tags = {
                "start_time": datetime.utcnow().isoformat(),
                "api_version": "0.0.1",
                "run_type": "feature_extraction"
            }
            if tags:
                run_tags.update(tags)
            
            # Start run
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                tags=run_tags,
                run_name=run_name
            )
            
            run_id = run.info.run_id
            self._active_runs[run_id] = {
                "start_time": time.time(),
                "run_name": run_name,
                "tags": run_tags
            }
            
            self._total_runs += 1
            
            self.logger.debug(
                "Started MLflow run",
                run_id=run_id,
                run_name=run_name
            )
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            return str(uuid.uuid4())  # Return dummy ID on failure
    
    async def log_params(
        self,
        run_id: str,
        params: Dict[str, Union[str, int, float, bool]]
    ):
        """Log parameters to MLflow run."""
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # Convert all params to strings (MLflow requirement)
            string_params = {k: str(v) for k, v in params.items()}
            
            self._client.log_batch(
                run_id=run_id,
                params=[
                    mlflow.entities.Param(key, value)
                    for key, value in string_params.items()
                ]
            )
            
            self.logger.debug(
                "Logged parameters to MLflow",
                run_id=run_id,
                param_count=len(params)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ):
        """Log metrics to MLflow run."""
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            timestamp = int(time.time() * 1000)  # MLflow expects milliseconds
            
            self._client.log_batch(
                run_id=run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key=key,
                        value=float(value),
                        timestamp=timestamp,
                        step=step or 0
                    )
                    for key, value in metrics.items()
                ]
            )
            
            self.logger.debug(
                "Logged metrics to MLflow",
                run_id=run_id,
                metric_count=len(metrics)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_data: Union[str, bytes, Dict[str, Any]]
    ):
        """Log artifact to MLflow run."""
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # Create temporary file
            temp_dir = f"/tmp/mlflow_artifacts_{run_id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            file_path = os.path.join(temp_dir, os.path.basename(artifact_path))
            
            # Write artifact data
            if isinstance(artifact_data, str):
                with open(file_path, 'w') as f:
                    f.write(artifact_data)
            elif isinstance(artifact_data, bytes):
                with open(file_path, 'wb') as f:
                    f.write(artifact_data)
            elif isinstance(artifact_data, dict):
                with open(file_path, 'w') as f:
                    json.dump(artifact_data, f, indent=2)
            
            # Log artifact
            self._client.log_artifact(run_id, file_path, artifact_path)
            
            # Cleanup
            os.remove(file_path)
            
            self.logger.debug(
                "Logged artifact to MLflow",
                run_id=run_id,
                artifact_path=artifact_path
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {e}")
    
    async def end_run(
        self,
        run_id: str,
        status: str = "FINISHED"
    ):
        """End MLflow run."""
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # Calculate run duration
            run_info = self._active_runs[run_id]
            duration = time.time() - run_info["start_time"]
            
            # Log final metrics
            await self.log_metrics(run_id, {
                "run_duration": duration,
                "end_timestamp": time.time()
            })
            
            # End run
            self._client.set_terminated(run_id, status)
            
            # Update counters
            if status == "FINISHED":
                self._successful_runs += 1
            else:
                self._failed_runs += 1
            
            # Remove from active runs
            del self._active_runs[run_id]
            
            self.logger.debug(
                "Ended MLflow run",
                run_id=run_id,
                status=status,
                duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {e}")
    
    async def log_model_performance(
        self,
        run_id: str,
        model_name: str,
        performance_metrics: Dict[str, float],
        model_params: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """Log comprehensive model performance data."""
        if not self.enable_logging:
            return
        
        try:
            # Log model parameters
            await self.log_params(run_id, {
                f"model_{k}": v for k, v in model_params.items()
            })
            
            # Log performance metrics
            await self.log_metrics(run_id, {
                f"performance_{k}": v for k, v in performance_metrics.items()
            })
            
            # Log feature importance if provided
            if feature_importance:
                await self.log_metrics(run_id, {
                    f"feature_importance_{k}": v 
                    for k, v in feature_importance.items()
                })
                
                # Also log as artifact
                await self.log_artifact(
                    run_id,
                    "feature_importance.json",
                    feature_importance
                )
            
            self.logger.info(
                "Logged model performance",
                run_id=run_id,
                model_name=model_name
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log model performance: {e}")
    
    async def get_experiment_metrics(self) -> Dict[str, Any]:
        """Get experiment-level metrics."""
        if not self.enable_logging or not self._initialized:
            return {}
        
        try:
            experiment = self._client.get_experiment(self._experiment_id)
            runs = self._client.search_runs(
                experiment_ids=[self._experiment_id],
                max_results=1000
            )
            
            # Calculate aggregate metrics
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r.info.status == "FINISHED"])
            failed_runs = total_runs - successful_runs
            
            # Get average metrics
            avg_processing_time = 0
            avg_sentiment_score = 0
            avg_coherence = 0
            
            if runs:
                processing_times = []
                sentiment_scores = []
                coherence_scores = []
                
                for run in runs:
                    metrics = run.data.metrics
                    if "processing_time" in metrics:
                        processing_times.append(metrics["processing_time"])
                    if "sentiment_score" in metrics:
                        sentiment_scores.append(metrics["sentiment_score"])
                    if "content_coherence" in metrics:
                        coherence_scores.append(metrics["content_coherence"])
                
                if processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                if sentiment_scores:
                    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
                if coherence_scores:
                    avg_coherence = sum(coherence_scores) / len(coherence_scores)
            
            return {
                "experiment_id": self._experiment_id,
                "experiment_name": experiment.name,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "avg_processing_time": avg_processing_time,
                "avg_sentiment_score": avg_sentiment_score,
                "avg_content_coherence": avg_coherence,
                "active_runs": len(self._active_runs),
                "uptime": time.time() - self._start_time
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment metrics: {e}")
            return {}
    
    @property
    def is_enabled(self) -> bool:
        """Check if MLflow logging is enabled."""
        return self.enable_logging and self._initialized
    
    async def log_comprehensive_analysis(
        self,
        run_id: str,
        request_data: Dict[str, Any],
        features: Dict[str, Any],
        engagement_prediction: Dict[str, Any],
        processing_metrics: Dict[str, float]
    ):
        """
        Log comprehensive content analysis results in a single run.
        
        This method logs all aspects of the analysis:
        - Input parameters
        - Extracted features (text, image, multimodal)
        - Engagement prediction with confidence
        - Processing performance metrics
        """
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # 1. LOG INPUT PARAMETERS
            input_params = {
                "content_type": request_data.get("content_type", "unknown"),
                "text_length": len(request_data.get("text", "")),
                "has_image": bool(request_data.get("image_url")),
                "request_timestamp": datetime.utcnow().isoformat(),
                "model_version": "jinaclip-v2"
            }
            await self.log_params(run_id, input_params)
            
            # 2. LOG TEXT FEATURES
            if "text_features" in features:
                text_features = features["text_features"]
                text_metrics = {
                    "text_sentiment_score": text_features.get("sentiment_score", 0.0),
                    "text_readability_score": text_features.get("readability_score", 0.0),
                    "text_word_count": text_features.get("word_count", 0),
                    "text_character_count": text_features.get("character_count", 0),
                    "text_hashtag_count": len(text_features.get("hashtags", [])),
                    "text_mention_count": len(text_features.get("mentions", [])),
                    "text_emoji_count": text_features.get("emoji_count", 0),
                    "text_url_count": text_features.get("url_count", 0)
                }
                await self.log_metrics(run_id, text_metrics)
                
                # Log sentiment label as parameter
                await self.log_params(run_id, {
                    "text_sentiment_label": text_features.get("sentiment_label", "neutral")
                })
            
            # 3. LOG IMAGE FEATURES (if available)
            if "image_features" in features:
                image_features = features["image_features"]
                image_metrics = {
                    "image_embedding_norm": float(image_features.get("embedding_norm", 0.0)),
                    "image_processing_time": float(image_features.get("processing_time", 0.0))
                }
                await self.log_metrics(run_id, image_metrics)
                
                # Log image metadata as parameters
                if "metadata" in image_features:
                    metadata = image_features["metadata"]
                    await self.log_params(run_id, {
                        "image_width": str(metadata.get("width", 0)),
                        "image_height": str(metadata.get("height", 0)),
                        "image_format": metadata.get("format", "unknown")
                    })
            
            # 4. LOG MULTIMODAL FEATURES
            multimodal_metrics = {
                "content_coherence": features.get("content_coherence", 0.0),
                "feature_extraction_time": features.get("processing_time", 0.0)
            }
            await self.log_metrics(run_id, multimodal_metrics)
            
            # 5. LOG ENGAGEMENT PREDICTION (MAIN FOCUS)
            engagement_metrics = {
                "engagement_score": engagement_prediction.get("score", 0.0),
                "engagement_confidence": engagement_prediction.get("confidence", 0.0),
                "engagement_score_normalized": engagement_prediction.get("score", 0.0) / 100.0
            }
            await self.log_metrics(run_id, engagement_metrics)
            
            # Log engagement level as parameter
            await self.log_params(run_id, {
                "engagement_level": engagement_prediction.get("level", "unknown")
            })
            
            # Log engagement factors if available
            if "factors" in engagement_prediction:
                factors = engagement_prediction["factors"]
                factor_metrics = {
                    f"engagement_factor_{k}": float(v) 
                    for k, v in factors.items() 
                    if isinstance(v, (int, float))
                }
                await self.log_metrics(run_id, factor_metrics)
            
            # 6. LOG PROCESSING PERFORMANCE
            performance_metrics = {
                "total_processing_time": processing_metrics.get("total_time", 0.0),
                "feature_extraction_time": processing_metrics.get("feature_time", 0.0),
                "prediction_time": processing_metrics.get("prediction_time", 0.0),
                "api_response_time": processing_metrics.get("response_time", 0.0)
            }
            await self.log_metrics(run_id, performance_metrics)
            
            # 7. LOG COMPREHENSIVE ARTIFACTS
            
            # Complete analysis results as JSON artifact
            complete_results = {
                "input": request_data,
                "features": features,
                "engagement_prediction": engagement_prediction,
                "processing_metrics": processing_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.log_artifact(
                run_id,
                "analysis_results/complete_analysis.json",
                complete_results
            )
            
            # Feature summary for quick analysis
            feature_summary = {
                "text_features": {
                    "sentiment": text_features.get("sentiment_score", 0.0) if "text_features" in features else 0.0,
                    "readability": text_features.get("readability_score", 0.0) if "text_features" in features else 0.0,
                    "word_count": text_features.get("word_count", 0) if "text_features" in features else 0
                },
                "engagement": {
                    "score": engagement_prediction.get("score", 0.0),
                    "confidence": engagement_prediction.get("confidence", 0.0),
                    "level": engagement_prediction.get("level", "unknown")
                },
                "performance": {
                    "total_time": processing_metrics.get("total_time", 0.0),
                    "success": True
                }
            }
            await self.log_artifact(
                run_id,
                "summaries/feature_summary.json",
                feature_summary
            )
            
            self.logger.info(
                "Logged comprehensive analysis to MLflow",
                run_id=run_id,
                engagement_score=engagement_prediction.get("score", 0.0),
                confidence=engagement_prediction.get("confidence", 0.0),
                processing_time=processing_metrics.get("total_time", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log comprehensive analysis: {e}")
    
    async def log_engagement_prediction_details(
        self,
        run_id: str,
        prediction: Dict[str, Any],
        feature_contributions: Dict[str, float],
        model_metadata: Dict[str, Any]
    ):
        """
        Log detailed engagement prediction information.
        
        Args:
            run_id: MLflow run ID
            prediction: Engagement prediction results
            feature_contributions: Individual feature contributions to the score
            model_metadata: Model configuration and metadata
        """
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # Log prediction metrics
            prediction_metrics = {
                "engagement_score_raw": prediction.get("score", 0.0),
                "engagement_confidence_raw": prediction.get("confidence", 0.0),
                "engagement_score_percentile": prediction.get("score", 0.0) / 100.0,
                "confidence_interval_lower": prediction.get("confidence", 0.0) - 0.1,
                "confidence_interval_upper": prediction.get("confidence", 0.0) + 0.1
            }
            await self.log_metrics(run_id, prediction_metrics)
            
            # Log feature contributions
            contribution_metrics = {
                f"contribution_{feature}": contribution
                for feature, contribution in feature_contributions.items()
            }
            await self.log_metrics(run_id, contribution_metrics)
            
            # Log model metadata as parameters
            model_params = {
                f"model_{k}": str(v) for k, v in model_metadata.items()
            }
            await self.log_params(run_id, model_params)
            
            # Create detailed prediction artifact
            prediction_details = {
                "prediction": prediction,
                "feature_contributions": feature_contributions,
                "model_metadata": model_metadata,
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "confidence_analysis": {
                    "high_confidence": prediction.get("confidence", 0.0) > 0.8,
                    "medium_confidence": 0.5 <= prediction.get("confidence", 0.0) <= 0.8,
                    "low_confidence": prediction.get("confidence", 0.0) < 0.5
                }
            }
            await self.log_artifact(
                run_id,
                "predictions/engagement_prediction_details.json",
                prediction_details
            )
            
            self.logger.debug(
                "Logged engagement prediction details",
                run_id=run_id,
                score=prediction.get("score", 0.0),
                confidence=prediction.get("confidence", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log engagement prediction details: {e}")
    
    async def log_feature_analysis(
        self,
        run_id: str,
        text_features: Dict[str, Any],
        image_features: Optional[Dict[str, Any]] = None,
        multimodal_features: Optional[Dict[str, Any]] = None
    ):
        """
        Log detailed feature analysis for ML observability.
        
        Args:
            run_id: MLflow run ID
            text_features: Extracted text features
            image_features: Extracted image features (optional)
            multimodal_features: Combined multimodal features (optional)
        """
        if not self.enable_logging or run_id not in self._active_runs:
            return
        
        try:
            # Text feature metrics
            text_metrics = {
                "text_sentiment_positive": 1.0 if text_features.get("sentiment_label") == "positive" else 0.0,
                "text_sentiment_negative": 1.0 if text_features.get("sentiment_label") == "negative" else 0.0,
                "text_sentiment_neutral": 1.0 if text_features.get("sentiment_label") == "neutral" else 0.0,
                "text_readability_grade": text_features.get("readability_score", 0.0) / 10.0,  # Normalize to 0-10 scale
                "text_length_category": self._categorize_text_length(text_features.get("word_count", 0)),
                "text_social_signals": len(text_features.get("hashtags", [])) + len(text_features.get("mentions", [])),
                "text_emoji_density": text_features.get("emoji_count", 0) / max(text_features.get("word_count", 1), 1)
            }
            await self.log_metrics(run_id, text_metrics)
            
            # Image feature metrics (if available)
            if image_features:
                image_metrics = {
                    "image_available": 1.0,
                    "image_embedding_magnitude": image_features.get("embedding_norm", 0.0),
                    "image_processing_success": 1.0 if image_features.get("success", False) else 0.0
                }
                await self.log_metrics(run_id, image_metrics)
            else:
                await self.log_metrics(run_id, {"image_available": 0.0})
            
            # Multimodal feature metrics (if available)
            if multimodal_features:
                multimodal_metrics = {
                    "multimodal_coherence": multimodal_features.get("content_coherence", 0.0),
                    "multimodal_alignment": multimodal_features.get("text_image_alignment", 0.0),
                    "multimodal_complexity": multimodal_features.get("complexity_score", 0.0)
                }
                await self.log_metrics(run_id, multimodal_metrics)
            
            # Create comprehensive feature artifact
            feature_analysis = {
                "text_features": text_features,
                "image_features": image_features,
                "multimodal_features": multimodal_features,
                "feature_extraction_timestamp": datetime.utcnow().isoformat(),
                "feature_quality": {
                    "text_quality": self._assess_text_quality(text_features),
                    "image_quality": self._assess_image_quality(image_features) if image_features else None,
                    "overall_quality": self._assess_overall_quality(text_features, image_features)
                }
            }
            await self.log_artifact(
                run_id,
                "features/feature_analysis.json",
                feature_analysis
            )
            
            self.logger.debug(
                "Logged feature analysis",
                run_id=run_id,
                text_features_count=len(text_features),
                has_image_features=image_features is not None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log feature analysis: {e}")
    
    def _categorize_text_length(self, word_count: int) -> float:
        """Categorize text length for ML analysis."""
        if word_count < 10:
            return 0.0  # Very short
        elif word_count < 50:
            return 0.25  # Short
        elif word_count < 150:
            return 0.5  # Medium
        elif word_count < 300:
            return 0.75  # Long
        else:
            return 1.0  # Very long
    
    def _assess_text_quality(self, text_features: Dict[str, Any]) -> float:
        """Assess text feature quality for ML observability."""
        quality_score = 0.0
        
        # Readability contribution (0-0.4)
        readability = text_features.get("readability_score", 0.0)
        quality_score += min(readability / 100.0, 0.4)
        
        # Length contribution (0-0.3)
        word_count = text_features.get("word_count", 0)
        if 20 <= word_count <= 200:  # Optimal range
            quality_score += 0.3
        elif 10 <= word_count <= 300:  # Acceptable range
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Social signals contribution (0-0.3)
        hashtags = len(text_features.get("hashtags", []))
        mentions = len(text_features.get("mentions", []))
        emojis = text_features.get("emoji_count", 0)
        
        social_score = min((hashtags + mentions + emojis) / 5.0, 0.3)
        quality_score += social_score
        
        return min(quality_score, 1.0)
    
    def _assess_image_quality(self, image_features: Dict[str, Any]) -> float:
        """Assess image feature quality for ML observability."""
        if not image_features or not image_features.get("success", False):
            return 0.0
        
        quality_score = 0.5  # Base score for successful processing
        
        # Embedding quality
        embedding_norm = image_features.get("embedding_norm", 0.0)
        if embedding_norm > 0.5:
            quality_score += 0.3
        elif embedding_norm > 0.1:
            quality_score += 0.2
        
        # Processing time (faster is better)
        processing_time = image_features.get("processing_time", 1.0)
        if processing_time < 0.5:
            quality_score += 0.2
        elif processing_time < 1.0:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _assess_overall_quality(
        self, 
        text_features: Dict[str, Any], 
        image_features: Optional[Dict[str, Any]]
    ) -> float:
        """Assess overall feature quality."""
        text_quality = self._assess_text_quality(text_features)
        
        if image_features:
            image_quality = self._assess_image_quality(image_features)
            return (text_quality * 0.7) + (image_quality * 0.3)  # Text weighted higher
        else:
            return text_quality


    @property
    def metrics(self) -> Dict[str, Any]:
        """Get tracker metrics."""
        return {
            "enabled": self.enable_logging,
            "initialized": self._initialized,
            "total_runs": self._total_runs,
            "successful_runs": self._successful_runs,
            "failed_runs": self._failed_runs,
            "active_runs": len(self._active_runs),
            "experiment_id": self._experiment_id,
            "uptime": time.time() - self._start_time
        }

