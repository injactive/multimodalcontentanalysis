#!/usr/bin/env python3
"""
Comprehensive Demo for Improved Multi-Modal Content Analysis API v2.0

This demo showcases all the improvements made based on the brutal code review:
- Professional architecture with dependency injection
- MLflow experiment tracking and logging
- Async I/O and performance optimizations
- Comprehensive error handling and security
- Structured logging with correlation IDs
- Professional testing framework

Run this demo to see the API in action with real examples and MLflow tracking.
"""

import asyncio
import json
import time
from typing import List, Dict, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.domain import AnalysisRequest, ContentType
from src.services.feature_extraction import FeatureExtractionService
from src.utils.http_client import AsyncHTTPClient
from src.utils.mlflow_tracker import MLflowTracker
from src.utils.logger import setup_logging, get_logger
from src.utils.test_cases import import_testcases
from config.settings import APISettings


# Setup logging for demo
setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


class ComprehensiveDemo:
    """Comprehensive demo showcasing all API improvements."""
    
    def __init__(self):
        self.settings = APISettings(
            mlflow_tracking_uri="./mlruns",
            mlflow_experiment_name="comprehensive-demo",
            mlflow_enable_logging=True,
            debug=True
        )
        
        self.http_client = None
        self.mlflow_tracker = None
        self.feature_service = None
        
        # Demo test cases
        self.test_cases = import_testcases(file_str="testcases.json")
    
    async def initialize(self):
        """Initialize all services."""
        logger.info("🚀 Initializing Comprehensive Demo...")
        
        # Initialize HTTP client
        self.http_client = AsyncHTTPClient(timeout=10.0)
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker(
            tracking_uri=self.settings.mlflow_tracking_uri,
            experiment_name=self.settings.mlflow_experiment_name,
            enable_logging=True
        )
        await self.mlflow_tracker.initialize()
        
        # Initialize feature service
        self.feature_service = FeatureExtractionService(
            http_client=self.http_client,
            mlflow_tracker=self.mlflow_tracker,
            settings=self.settings
        )
        await self.feature_service.initialize()
        
        logger.info("✅ All services initialized successfully!")
    
    async def cleanup(self):
        """Cleanup all services."""
        logger.info("🧹 Cleaning up services...")
        
        if self.feature_service:
            await self.feature_service.cleanup()
        
        if self.http_client:
            await self.http_client.close()
        
        logger.info("✅ Cleanup completed!")
    
    async def run_demo(self):
        """Run the comprehensive demo."""
        logger.info("🎬 Starting Comprehensive Demo of Improved API v2.0")
        logger.info("=" * 60)
        
        try:
            await self.initialize()
            
            # Run all test cases
            results = []
            for i, test_case in enumerate(self.test_cases, 1):
                logger.info(f"\n📋 Test Case {i}: {test_case['name']}")
                logger.info("-" * 40)
                
                result = await self.run_test_case(test_case, f"demo_case_{i}")
                results.append(result)
                
                # Add delay between test cases
                await asyncio.sleep(1)
            
            # Generate summary
            await self.generate_summary(results)
            
            # Show MLflow metrics
            await self.show_mlflow_metrics()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def run_test_case(self, test_case: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Run a single test case."""
        request = test_case["request"]
        
        logger.info(f"📝 Text: {request.text[:100]}{'...' if len(request.text) > 100 else ''}")
        logger.info(f"🖼️  Image: {'Yes' if request.image_url else 'No'}")
        logger.info(f"📊 Content Type: {request.content_type}")
        
        start_time = time.time()
        
        try:
            # Extract features (this will log to MLflow)
            features = await self.feature_service.extract_features(request, request_id)
            
            processing_time = time.time() - start_time
            
            # Log results
            text_features = features.text_features
            logger.info(f"💭 Sentiment: {text_features.sentiment_label} ({text_features.sentiment_score:.2f})")
            logger.info(f"📖 Readability: {text_features.readability_score:.1f}/100")
            logger.info(f"📏 Word Count: {text_features.word_count}")
            logger.info(f"🏷️  Hashtags: {len(text_features.hashtags)}")
            logger.info(f"👥 Mentions: {len(text_features.mentions)}")
            logger.info(f"😊 Emojis: {text_features.emoji_count}")
            logger.info(f"🔗 URLs: {text_features.url_count}")
            
            if features.image_features:
                logger.info(f"🖼️  Image Size: {features.image_features.width}x{features.image_features.height}")
                logger.info(f"📐 Aspect Ratio: {features.image_features.aspect_ratio:.2f}")
            
            logger.info(f"🎯 Content Coherence: {features.content_coherence:.2f}")
            logger.info(f"⚡ Processing Time: {processing_time:.3f}s")
            logger.info(f"Text: {request.text[:100]}")
            
            # Generate engagement prediction (simplified)
            engagement_score = self._calculate_demo_engagement_score(features)
            logger.info(f"📈 Predicted Engagement: {engagement_score:.1f}/100")
            
            result = {
                "test_case": test_case["name"],
                "processing_time": processing_time,
                "sentiment_score": text_features.sentiment_score,
                "readability_score": text_features.readability_score,
                "content_coherence": features.content_coherence,
                "engagement_score": engagement_score,
                "word_count": text_features.word_count,
                "has_image": features.image_features is not None,
                "success": True
            }
            
            logger.info("✅ Test case completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Test case failed: {e}")
            return {
                "test_case": test_case["name"],
                "error": str(e),
                "success": False
            }
    
    def _calculate_demo_engagement_score(self, features) -> float:
        """Calculate a demo engagement score."""
        text_features = features.text_features
        
        # Base score
        score = 50.0
        
        # Sentiment contribution
        score += text_features.sentiment_score * 20
        
        # Readability contribution
        score += (text_features.readability_score - 50) * 0.3
        
        # Social signals
        social_signals = (
            len(text_features.hashtags) * 3 +
            len(text_features.mentions) * 2 +
            text_features.emoji_count * 2
        )
        score += min(social_signals, 15)
        
        # Image bonus
        if features.image_features:
            score += 10
        
        # Content coherence
        score += features.content_coherence * 10
        
        return max(0, min(100, score))
    
    async def generate_summary(self, results: List[Dict[str, Any]]):
        """Generate demo summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 DEMO SUMMARY")
        logger.info("=" * 60)
        
        successful_tests = [r for r in results if r.get("success", False)]
        failed_tests = [r for r in results if not r.get("success", False)]
        
        logger.info(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
        logger.info(f"❌ Failed Tests: {len(failed_tests)}")
        
        if successful_tests:
            avg_processing_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
            avg_engagement = sum(r["engagement_score"] for r in successful_tests) / len(successful_tests)
            avg_sentiment = sum(r["sentiment_score"] for r in successful_tests) / len(successful_tests)
            
            logger.info(f"⚡ Average Processing Time: {avg_processing_time:.3f}s")
            logger.info(f"📈 Average Engagement Score: {avg_engagement:.1f}/100")
            logger.info(f"💭 Average Sentiment Score: {avg_sentiment:.2f}")
        
        if failed_tests:
            logger.info("\n❌ Failed Test Cases:")
            for test in failed_tests:
                logger.info(f"  - {test['test_case']}: {test.get('error', 'Unknown error')}")
    
    async def show_mlflow_metrics(self):
        """Show MLflow experiment metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MLFLOW EXPERIMENT METRICS")
        logger.info("=" * 60)
        
        try:
            metrics = await self.mlflow_tracker.get_experiment_metrics()
            
            logger.info(f"🧪 Experiment: {metrics.get('experiment_name', 'N/A')}")
            logger.info(f"🆔 Experiment ID: {metrics.get('experiment_id', 'N/A')}")
            logger.info(f"🏃 Total Runs: {metrics.get('total_runs', 0)}")
            logger.info(f"✅ Successful Runs: {metrics.get('successful_runs', 0)}")
            logger.info(f"❌ Failed Runs: {metrics.get('failed_runs', 0)}")
            logger.info(f"📊 Success Rate: {metrics.get('success_rate', 0):.1%}")
            logger.info(f"⚡ Avg Processing Time: {metrics.get('avg_processing_time', 0):.3f}s")
            logger.info(f"💭 Avg Sentiment Score: {metrics.get('avg_sentiment_score', 0):.2f}")
            logger.info(f"🎯 Avg Content Coherence: {metrics.get('avg_content_coherence', 0):.2f}")
            
            logger.info(f"\n📁 MLflow Tracking URI: {self.settings.mlflow_tracking_uri}")
            logger.info("💡 Run 'mlflow ui' to view detailed experiment tracking!")
            
        except Exception as e:
            logger.error(f"❌ Failed to get MLflow metrics: {e}")
    
    async def run_performance_test(self):
        """Run performance test with multiple concurrent requests."""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 PERFORMANCE TEST")
        logger.info("=" * 60)
        
        # Create multiple requests
        requests = []
        for i in range(10):
            requests.append(AnalysisRequest(
                text=f"Performance test request {i} with some engaging content! #test #performance 🚀",
                content_type=ContentType.TEXT,
                metadata={"performance_test": True, "request_id": i}
            ))
        
        logger.info(f"📊 Running {len(requests)} concurrent requests...")
        
        start_time = time.time()
        
        # Run requests concurrently
        tasks = [
            self.feature_service.extract_features(req, f"perf_test_{i}")
            for i, req in enumerate(requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"✅ Completed: {len(successful_results)}/{len(requests)} requests")
        logger.info(f"⚡ Total Time: {total_time:.2f}s")
        logger.info(f"📊 Avg Time per Request: {total_time/len(requests):.3f}s")
        logger.info(f"🚀 Requests per Second: {len(requests)/total_time:.1f}")


async def main():
    """Main demo function."""
    print("🎬 Multi-Modal Content Analysis API")
    print("=" * 70)
    
    demo = ComprehensiveDemo()
    
    try:
        # Run main demo
        await demo.run_demo()
        
        # Run performance test
        await demo.run_performance_test()
        
        print("\n🎉 Demo completed successfully!")
        print("💡 Check the MLflow UI for detailed experiment tracking:")
        print(f"   cd {os.path.dirname(os.path.abspath(__file__))}")
        print("   mlflow ui")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

