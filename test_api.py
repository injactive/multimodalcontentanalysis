#!/usr/bin/env python3
"""
Test Script for Multi-Modal Content Analysis API

This script demonstrates the API functionality by sending test requests
and displaying the results in a formatted way.

Usage:
    python test_api.py

Requirements:
    - API server running on localhost:8000
    - requests library installed
"""

import json
import time
import requests
from typing import Dict, Any


class APITester:
    """Test client for the Multi-Modal Content Analysis API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("🔍 Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {health_data['status']}")
            print(f"   Version: {health_data['version']}")
            print(f"   Models loaded: {health_data['models_loaded']}")
            
            return health_data['status'] == 'healthy'
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_analyze_post(self, text: str, image_url: str) -> Dict[str, Any]:
        """Test the analyze post endpoint."""
        print(f"\n🔍 Testing post analysis...")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Image URL: {image_url}")
        
        try:
            payload = {
                "text": text,
                "image_url": image_url
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/analyze-post",
                json=payload
            )
            response.raise_for_status() # checks the HTTP status code of the response
            request_time = (time.time() - start_time) * 1000
            
            result = response.json()
            
            print(f"✅ Analysis completed in {request_time:.2f}ms")
            self._display_results(result)
            
            return result
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if e.response.text:
                try:
                    error_data = e.response.json()
                    print(f"   Error details: {error_data}")
                except:
                    print(f"   Response: {e.response.text}")
            return {}
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {}
    
    def _display_results(self, result: Dict[str, Any]):
        """Display analysis results in a formatted way."""
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        # Text Features
        text_features = result.get('text_features', {})
        print(f"\n📝 TEXT ANALYSIS:")
        print(f"   Sentiment: {text_features.get('sentiment_label', 'N/A')} "
              f"(score: {text_features.get('sentiment_score', 0):.3f}, "
              f"confidence: {text_features.get('confidence', 0):.3f})")
        print(f"   Readability: {text_features.get('readability_score', 0):.1f}/100")
        print(f"   Word count: {text_features.get('word_count', 0)}")
        print(f"   Hashtags: {text_features.get('hashtag_count', 0)}")
        print(f"   Mentions: {text_features.get('mention_count', 0)}")
        print(f"   Emojis: {text_features.get('emoji_count', 0)}")
        print(f"   Keywords: {', '.join(text_features.get('keywords', []))}")
        
        # Image Features
        image_features = result.get('image_features', {})
        print(f"\n🖼️  IMAGE ANALYSIS:")
        print(f"   Dimensions: {image_features.get('width', 0)}x{image_features.get('height', 0)}")
        print(f"   Aspect ratio: {image_features.get('aspect_ratio', 0):.2f}")
        print(f"   File size: {image_features.get('file_size_kb', 0):.1f} KB")
        print(f"   Brightness: {image_features.get('brightness', 0):.3f}")
        print(f"   Contrast: {image_features.get('contrast', 0):.3f}")
        
        # Multi-modal Features
        print(f"\n🔗 MULTI-MODAL ANALYSIS:")
        print(f"   Text-Image Alignment: {result.get('text_image_alignment', 0):.3f}")
        print(f"   Overall Quality Score: {result.get('overall_quality_score', 0):.3f}")
        
        # Engagement Prediction
        engagement = result.get('engagement_prediction', {})
        print(f"\n🎯 ENGAGEMENT PREDICTION:")
        print(f"   Score: {engagement.get('engagement_score', 0):.1f}/100")
        
        confidence_interval = engagement.get('confidence_interval', [0, 0])
        print(f"   Confidence Interval: [{confidence_interval[0]:.1f}, {confidence_interval[1]:.1f}]")
        
        print(f"   Explanation: {engagement.get('explanation', 'N/A')}")
        
        # Success Factors
        success_factors = engagement.get('success_factors', [])
        if success_factors:
            print(f"\n✅ SUCCESS FACTORS:")
            for factor in success_factors:
                print(f"   • {factor}")
        
        # Risk Factors
        risk_factors = engagement.get('risk_factors', [])
        if risk_factors:
            print(f"\n⚠️  RISK FACTORS:")
            for factor in risk_factors:
                print(f"   • {factor}")
        
        # Feature Importance
        feature_importance = engagement.get('feature_importance', {})
        if feature_importance:
            print(f"\n📈 FEATURE IMPORTANCE:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:  # Top 5
                print(f"   {feature}: {importance:.1f}%")
        
        print(f"\n⏱️  Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        print("=" * 50)


def main():
    """Main test function."""
    print("🚀 Multi-Modal Content Analysis API Test")
    print("=" * 50)
    
    # Initialize tester
    tester = APITester()
    
    # Test health check
    if not tester.test_health_check():
        print("\n❌ API is not healthy. Please check if the server is running.")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Positive Travel Post",
            "text": "Just had the most amazing sunset experience! 🌅 The colors were absolutely breathtaking and I feel so grateful for this moment. Perfect end to a beautiful day exploring nature. #sunset #travel #grateful #nature #photography",
            "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop"
        },
        {
            "name": "Food Post with Emojis",
            "text": "Homemade pizza night! 🍕✨ Nothing beats fresh ingredients and good company. Who else loves cooking at home? #homemade #pizza #cooking #foodie",
            "image_url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=800&h=600&fit=crop"
        },
        {
            "name": "Motivational Fitness Post",
            "text": "Early morning workout complete! 💪 Remember, every small step counts towards your goals. Stay consistent and believe in yourself! #fitness #motivation #morningworkout #goals #health",
            "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800&h=600&fit=crop"
        },
        {
            "name": "Simple Product Post",
            "text": "New coffee blend arrived today. Testing it out.",
            "image_url": "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=800&h=600&fit=crop"
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}: {test_case['name']}")
        print("-" * 30)
        
        result = tester.test_analyze_post(
            text=test_case['text'],
            image_url=test_case['image_url']
        )
        
        if result:
            engagement_score = result.get('engagement_prediction', {}).get('engagement_score', 0)
            print(f"\n🏆 Final Score: {engagement_score:.1f}/100")
        
        # Wait between requests to be nice to the API
        if i < len(test_cases):
            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    print(f"\n🎉 All tests completed!")
    print("\n💡 Tips for running your own tests:")
    print("   • Use high-quality, relevant images")
    print("   • Include emojis and hashtags in your text")
    print("   • Ensure text and image are well-aligned")
    print("   • Try different sentiment tones")
    print("   • Test various image aspect ratios")


if __name__ == "__main__":
    main()

