"""
Author: Christian Kruschel
Version: 0.0.1 with JinaCLIP v2
"""

import hashlib
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import re
import io
import base64

# Core dependencies
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# Internal dependencies
from src.models.domain import (
    AnalysisRequest, MultiModalFeatures, TextFeatures, ImageFeatures
)
from src.utils.http_client import AsyncHTTPClient
from src.utils.mlflow_tracker import MLflowTracker
from src.utils.logger import get_logger
from config.settings import APISettings

from sklearn.metrics.pairwise import cosine_similarity

class JinaCLIPFeatureExtractionService:
    """
    Advanced feature extraction service using JinaCLIP v2.
    
    This service provides state-of-the-art multimodal feature extraction
    with improved performance over traditional CLIP models.
    """
    
    def __init__(
        self,
        http_client: AsyncHTTPClient,
        mlflow_tracker: MLflowTracker,
        settings: APISettings
    ):
        """
        Constructor to initialize the feature extraction service.

        Args:
            http_client (AsyncHTTPClient): Asynchronous HTTP client for making requests.
            mlflow_tracker (MLflowTracker): Tracker for logging model parameters and metrics.
            settings (APISettings): Configuration settings for the API.
        """
        self.http_client = http_client
        self.mlflow_tracker = mlflow_tracker
        self.settings = settings
        self.logger = get_logger(__name__) # Initialize logger
        
        # Model components (initialized in initialize())
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Feature cache for performance optimization
        self._feature_cache: Dict[str, Any] = {}
        
        # Security settings to block specific domains (e.g., localhost, cloud metadata services)
        self.blocked_domains = {
            "localhost", "127.0.0.1", "0.0.0.0", "::1",
            "169.254.169.254",  # AWS metadata
            "metadata.google.internal"  # GCP metadata
        }

        # Allowed URL schemes (http and https only)
        self.allowed_schemes = {"http", "https"}
        
        # Predefined sets of positive and negative sentiment keywords for enhanced analysis
        self.positive_keywords = {
            "amazing", "awesome", "fantastic", "incredible", "love", "perfect",
            "excellent", "wonderful", "brilliant", "outstanding", "great",
            "beautiful", "stunning", "impressive", "remarkable", "superb"
        }
        
        self.negative_keywords = {
            "terrible", "awful", "horrible", "hate", "worst", "disgusting",
            "disappointing", "frustrating", "annoying", "bad", "poor",
            "ugly", "boring", "useless", "pathetic", "ridiculous"
        }
    
    async def initialize(self):
        """
        Initialize the JinaCLIP v2 model and its components.
        
        This method loads the model, tokenizer, and processor, and sets up the model for use.
        It also handles GPU setup if available.
        """
        self.logger.info("Initializing JinaCLIP v2 feature extraction service...")
        
        try:
            # Load the pretrained JinaCLIP v2 model
            self.logger.info(f"Loading JinaCLIP v2 model: {self.settings.model_name}")
            
            self.model = AutoModel.from_pretrained(
                self.settings.model_name,
                trust_remote_code=self.settings.trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
            
            # Try loading a custom processor for JinaCLIP v2, otherwise fall back to model’s built-in processor
            try:
                self.processor = AutoProcessor.from_pretrained(self.settings.model_name)
            except Exception:
                self.logger.info("Using model's built-in processing methods")
                self.processor = None
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.logger.info("JinaCLIP v2 model loaded on GPU")
            else:
                self.logger.info("JinaCLIP v2 model loaded on CPU")
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.logger.info("JinaCLIP feature extraction service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize JinaCLIP v2 model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    async def cleanup(self):
        """
        Cleanup resources, including clearing cache and freeing GPU memory (if used).
        """
        self.logger.info("Cleaning up JinaCLIP v2 feature extraction service...")
        
        # Clear cached feature data
        self._feature_cache.clear()
        
        # If using CUDA (GPU), clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("JinaCLIP v2 cleanup completed")
    
    async def extract_features(
        self, 
        request: AnalysisRequest, 
        request_id: str
    ) -> MultiModalFeatures:
        """
        Extract multimodal features (text, image, and combined features) using JinaCLIP v2.
        
        Args:
            request: Analysis request containing text and optional image
            request_id: Unique request identifier
            
        Returns:
            ContentFeatures with text, image, and multimodal features
        """
        start_time = time.time()
        
        # Start an MLflow run to track the process
        run_id = await self.mlflow_tracker.start_run(
            run_name=f"jinaclip_extraction_{request_id}",
            tags={
                "request_id": request_id,
                "content_type": str(request.content_type),
                "has_image": request.image_url is not None,
                "model": "jinaclip-v2"
            }
        )
        
        try:
            # Extract text features
            text_features = await self._extract_text_features(request.text)
            
            # Initialize variables for image features and similarity
            image_features = None
            text_image_similarity = None
            
            # If an image URL is provided, process the image as well
            if request.image_url:
                try:
                    image_features = await self._extract_image_features(request.image_url)
                    
                    # Calculate text-image similarity using JinaCLIP v2
                    text_image_similarity = await self._calculate_text_image_similarity(
                        request.text, request.image_url
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Image processing failed: {e}")
                    # Proceed without image features if image processing fails
            
            # Calculate content coherence (i.e., how well the text and image align)
            content_coherence = self._calculate_content_coherence(
                text_features, image_features, text_image_similarity
            )
            
            # Measure the processing time for the extraction
            processing_time = time.time() - start_time
            
            # Log metrics to MLflow
            await self.mlflow_tracker.log_metrics(
                run_id=run_id,
                metrics={
                    "processing_time": processing_time,
                    "sentiment_score": text_features.sentiment_score,
                    "readability_score": text_features.readability_score,
                    "content_coherence": content_coherence,
                    "text_image_similarity": text_image_similarity or 0.0,
                    "word_count": text_features.word_count,
                    "hashtag_count": len(text_features.hashtags),
                    "emoji_count": text_features.emoji_count
                }
            )
            
            # Log parameters to MLflow
            await self.mlflow_tracker.log_params(
                run_id=run_id,
                params={
                    "text": request.text,
                    "content_type": str(request.content_type),
                    "has_image": request.image_url is not None,
                    "text_word_count": text_features.word_count,
                    "hashtag_count": len(text_features.hashtags),
                    "mention_count": len(text_features.mentions),
                    "model_name": self.settings.model_name,
                    "embedding_dimension": self.settings.embedding_dimension
                }
            )
            
            # End the MLflow run and mark it as finished
            await self.mlflow_tracker.end_run(run_id, "FINISHED")
            
            self.logger.info(
                "JinaCLIP v2 feature extraction completed",
                request_id=request_id,
                processing_time=processing_time,
                content_coherence=content_coherence,
                image_features=image_features,
                text_image_similarity=text_image_similarity
            )
            
            # Return the extracted features in a structured format
            return MultiModalFeatures(
                text_features=text_features,
                image_features=image_features,
                text_image_similarity=text_image_similarity,
                content_coherence=content_coherence
            )
            
        except Exception as e:
            # If extraction fails, end the MLflow run with a failed status
            await self.mlflow_tracker.end_run(run_id, "FAILED")
            self.logger.error(f"JinaCLIP v2 feature extraction failed: {e}")
            raise
    
    async def _extract_text_features(self, text: str) -> TextFeatures:
        """Extract comprehensive text features using JinaCLIP v2."""
        
        # Check cache first to avoid unnecessary recomputations
        cache_key = self._generate_cache_key(text, "text")
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Basic text analysis: count words and characters in the text
        word_count = len(text.split())
        char_count = len(text)
        
        # Extract social media elements like hashtags, mentions, and URLs
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        # Count emojis in the text (simplified)
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿]', text))
        
        # Sentiment analysis: uses a combination of keyword matching and JinaCLIP v2 embeddings
        sentiment_score, sentiment_label = await self._analyze_sentiment_jinaclip_cosine_similarity(text)
        
        # Calculate readability score (Flesch Reading Ease)
        readability_score = self._calculate_readability(text)
        
        # Extract keywords using simple frequency analysis
        keywords = self._extract_keywords(text)
        
        # Create a TextFeatures object with all the extracted features
        text_features = TextFeatures(
            word_count=word_count,
            character_count=char_count,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            readability_score=readability_score,
            keywords=keywords,
            hashtags=hashtags,
            mentions=mentions,
            emoji_count=emoji_count,
            url_count=len(urls)
        )
        
        # Cache the computed features for future requests
        self._feature_cache[cache_key] = text_features
        
        return text_features
    
    async def _extract_image_features(self, image_url: str) -> ImageFeatures:
        """Extract image features using JinaCLIP v2."""
        
        # Validate the image URL to ensure it's safe
        self._validate_url(str(image_url))
        self.logger.info("validation done")
        
        # Check cache first before downloading the image again
        cache_key = self._generate_cache_key(str(image_url), "image")
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        self.logger.info("cache key done")
        
        try:
            # Download image from the provided URL
            image_data = await self.http_client.download_image(str(image_url))
            self.logger.info("download done")
            
            # Validate the image size to ensure it's not too large
            if len(image_data) > self.settings.max_image_file_size:
                raise ValueError(f"Image too large: {len(image_data)} bytes")
            
            # Open and process the image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert the image to RGB if it's not already in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate image dimensions and aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            # Resize the image if it's too large
            if max(width, height) > self.settings.max_image_size:
                image.thumbnail((self.settings.max_image_size, self.settings.max_image_size))
                width, height = image.size
            
            # Extract image embeddings using JinaCLIP v2
            image_embedding = self.model.encode_image(str(image_url), truncate_dim=512)

            # Convert image to base64 for storage
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create an ImageFeatures object with all the extracted image features
            image_features = ImageFeatures(
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                file_size=len(image_data),
                format=image.format or "JPEG",
                embedding=image_embedding.tolist() if image_embedding is not None else None,
                base64_data=image_base64
            )
            
            # Cache the extracted image features
            self._feature_cache[cache_key] = image_features
            
            return image_features
            
        except Exception as e:
            self.logger.error(f"Image feature extraction failed: {e}")
            raise
    
    async def _analyze_sentiment_jinaclip_cosine_similarity(self, text: str) -> tuple[float, str]:
        
        label_texts = ["This is a positive statement.", 
                       "This is a negative statement.", 
                       "This is a neutral statement."]
        # Enhance with JinaCLIP v2 embeddings (simplified approach)
        try:
            # Get text embedding
            text_embedding = self.model.encode_text(text, truncate_dim=512)
            
            # Use embedding magnitude as confidence modifier
            if text_embedding is not None:
                label_embeddings = [self.model.encode_text(lbl, truncate_dim=512) for lbl in label_texts]
                sims = cosine_similarity([text_embedding], label_embeddings)[0]
                sentiment_label = ["positive", "negative", "neutral"][np.argmax(sims)]
                confidence = max(sims)

        except Exception as e:
            self.logger.warning(f"JinaCLIP sentiment enhancement failed: {e}")

        return confidence, sentiment_label

    async def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using JinaCLIP v2."""
        try:
            with torch.no_grad():
                # Tokenize the text (using either processor or tokenizer)
                if self.processor:
                    inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
                else:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
                # Move inputs to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get text embedding
                text_embedding = self.model.get_text_features(**inputs)
                
                # Convert to numpy
                embedding = text_embedding.cpu().numpy().flatten()
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Text embedding extraction failed: {e}")
            return None
    
    async def _get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get image embedding using JinaCLIP v2."""
        try:
            with torch.no_grad():
                self.logger.warning("torch.no_grad")
                # Preprocess image using the defined processor if available
                if False: #self.processor:
                    inputs = self.processor(images=image, return_tensors="pt")
                else:
                    # Fallback to manual preprocessing if no processor is defined
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),  # Resize image to 224x224
                        transforms.ToTensor(), # Convert image to tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    inputs = {"pixel_values": transform(image).unsqueeze(0)}
                    self.logger.warning("inputs")
                
                # Move inputs to GPU if available for faster computation
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Extract image features using JinaCLIP v2 model
                image_embedding = self.model.get_image_features(**inputs)
                
                # Convert tensor to numpy array and flatten for consistency
                embedding = image_embedding.cpu().numpy().flatten()
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Image embedding extraction failed: {e}")
            return None
    
    async def _calculate_text_image_similarity(self, text: str, image_url: str) -> float:
        """Calculate text-image similarity using JinaCLIP v2."""
        try:
            # Get text embedding using previously defined method
            text_embedding = self.model.encode_text(text, truncate_dim=512)
        
            # Extract image embedding using previously defined method
            image_embedding = self.model.encode_image(str(image_url), truncate_dim=512)

            # Calculate cosine similarity between text and image embeddings
            if text_embedding is not None and image_embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(text_embedding, image_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding)
                )
                return float(similarity)
            
            # Return similarity as 0 if embeddings are unavailable
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Text-image similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        # Split text into sentences based on punctuation marks
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        # Return a default score of 0 if there are no sentences or words
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
        
        # Return score between 0 and 100
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower() # Convert word to lowercase
        vowels = "aeiouy" # Define vowels
        syllable_count = 0
        previous_was_vowel = False
        
        # Count syllables based on vowel groupings
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Special rule: subtract 1 syllable if the word ends with 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least 1 syllable is counted
        return max(1, syllable_count)
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis."""
        # Define common stop words to be removed
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        # Extract words and count frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort words by frequency and return the top 'max_keywords' keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def _calculate_content_coherence(
        self, 
        text_features: TextFeatures, 
        image_features: Optional[ImageFeatures],
        text_image_similarity: Optional[float]
    ) -> float:
        """Calculate overall content coherence score."""
        coherence_factors = []
        
        # Text coherence based on readability and structure
        text_coherence = min(text_features.readability_score / 100.0, 1.0)
        coherence_factors.append(text_coherence)
        
        # Content length factor: penalize or reward based on ideal length (100 words)
        optimal_length = 100  # Optimal length in words
        length_factor = 1.0 - abs(text_features.word_count - optimal_length) / optimal_length
        length_factor = max(0.0, min(1.0, length_factor))
        coherence_factors.append(length_factor)
        
        # Social signals factor: normalize the count of hashtags and mentions
        social_signals = len(text_features.hashtags) + len(text_features.mentions)
        social_factor = min(social_signals / 5.0, 1.0)  # Normalize to 0-1
        coherence_factors.append(social_factor)
        
        # Text-image alignment factor if available
        if text_image_similarity is not None:
            coherence_factors.append(text_image_similarity)
        
        # Return weighted average of all coherence factors
        return sum(coherence_factors) / len(coherence_factors)
    
    def _validate_url(self, url: str) -> None:
        """Validate URL for security (SSRF protection)."""
        try:
            # Parse the URL and perform basic security checks
            parsed = urlparse(url)
            
            # Check if URL scheme is supported (e.g., http, https)
            if parsed.scheme not in self.allowed_schemes:
                raise SecurityError(f"Unsupported scheme: {parsed.scheme}")
            
            # Block certain domains to prevent SSRF attacks
            hostname = parsed.hostname
            if hostname in self.blocked_domains:
                raise SecurityError(f"Blocked domain: {hostname}")
            
            # Check if the hostname belongs to a private IP range (basic protection)
            if hostname and (
                hostname.startswith("10.") or
                hostname.startswith("192.168.") or
                hostname.startswith("172.")
            ):
                raise SecurityError(f"Private IP address not allowed: {hostname}")
                
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Invalid URL: {url}")
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Generate cache key for content."""
        # Generate a hash of the content and return the cache key based on content type
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_type}:{content_hash}"


class SecurityError(Exception):
    """Security-related error for URL validation."""
    pass

