"""
Async HTTP client utility for the Multi-Modal Content Analysis API.

This module provides a secure, async HTTP client with proper error handling,
timeouts, and security validations for downloading images and making API calls.

Author: Christian Kruschel
Version: 0.0.1
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import io

import aiohttp

from ..utils.logger import get_logger


logger = get_logger(__name__)


class HTTPClientError(Exception):
    """Base exception for HTTP client errors."""
    pass


class SecurityError(HTTPClientError):
    """Exception for security-related errors."""
    pass


class DownloadError(HTTPClientError):
    """Exception for download-related errors."""
    pass


class AsyncHTTPClient:
    """Async HTTP client with security and performance optimizations."""
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        max_redirects: int = 3,
        user_agent: str = "MultiModal-API/2.0"
    ):
        self.timeout = timeout
        self.max_request_size = max_request_size
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        self.logger = logger.bind(component="http_client")
        
        # Security: Allowed schemes and domains
        self.allowed_schemes = {'http', 'https'}
        self.blocked_domains = {
            'localhost', '127.0.0.1', '0.0.0.0',
            '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'
        }
        
        # Supported image formats
        self.supported_image_types = {
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/webp', 'image/gif'
        }
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._total_bytes_downloaded = 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': self.user_agent},
#                max_redirects=self.max_redirects
            )
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _validate_url(self, url: str) -> None:
        """Validate URL for security."""
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise SecurityError(f"Invalid URL format: {e}")
        
        # Check scheme
        if parsed.scheme not in self.allowed_schemes:
            raise SecurityError(f"Unsupported scheme: {parsed.scheme}")
        
        # Check for blocked domains (basic SSRF protection)
        hostname = parsed.hostname
        if hostname:
            hostname = hostname.lower()
            if hostname in self.blocked_domains:
                raise SecurityError(f"Blocked domain: {hostname}")
            
            # Check for private IP ranges (basic check)
            if (hostname.startswith('10.') or 
                hostname.startswith('192.168.') or 
                hostname.startswith('172.')):
                raise SecurityError(f"Private IP address not allowed: {hostname}")
    
    async def download_image(
        self,
        url: str,
        max_size: Optional[int] = None
    ) -> bytes:
        """
        Download an image from URL with security validations.
        
        Args:
            url: Image URL to download
            max_size: Maximum file size in bytes (defaults to max_request_size)
            
        Returns:
            Image data as bytes
            
        Raises:
            SecurityError: If URL fails security validation
            DownloadError: If download fails
        """
        if max_size is None:
            max_size = self.max_request_size
        
        # Validate URL
        self._validate_url(url)
        
        await self._ensure_session()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Downloading image from: {url}")
            
            async with self._session.get(url) as response:
                # Check response status
                if response.status != 200:
                    raise DownloadError(
                        f"HTTP {response.status}: {response.reason}"
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if content_type not in self.supported_image_types:
                    raise DownloadError(
                        f"Unsupported content type: {content_type}"
                    )
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    raise DownloadError(
                        f"File too large: {content_length} bytes (max: {max_size})"
                    )
                
                # Download with size limit
                data = io.BytesIO()
                downloaded = 0
                
                async for chunk in response.content.iter_chunked(8192):
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        raise DownloadError(
                            f"File too large: {downloaded} bytes (max: {max_size})"
                        )
                    data.write(chunk)
                
                image_data = data.getvalue()
                download_time = time.time() - start_time
                
                # Update metrics
                self._request_count += 1
                self._total_bytes_downloaded += len(image_data)
                
                self.logger.info(
                    "Image downloaded successfully",
                    url=url,
                    size=len(image_data),
                    download_time=download_time,
                    content_type=content_type
                )
                
                return image_data
                
        except aiohttp.ClientError as e:
            raise DownloadError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise DownloadError(f"Download timeout after {self.timeout}s")
        except Exception as e:
            if isinstance(e, (SecurityError, DownloadError)):
                raise
            raise DownloadError(f"Unexpected error: {e}")
    
    async def post_json(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request with JSON data.
        
        Args:
            url: Target URL
            data: JSON data to send
            headers: Optional additional headers
            
        Returns:
            Response JSON data
            
        Raises:
            SecurityError: If URL fails security validation
            HTTPClientError: If request fails
        """
        self._validate_url(url)
        await self._ensure_session()
        
        request_headers = {'Content-Type': 'application/json'}
        if headers:
            request_headers.update(headers)
        
        try:
            async with self._session.post(
                url,
                json=data,
                headers=request_headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise HTTPClientError(
                        f"HTTP {response.status}: {error_text}"
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise HTTPClientError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise HTTPClientError(f"Request timeout after {self.timeout}s")
    
    async def get_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request and return JSON data.
        
        Args:
            url: Target URL
            headers: Optional additional headers
            
        Returns:
            Response JSON data
            
        Raises:
            SecurityError: If URL fails security validation
            HTTPClientError: If request fails
        """
        self._validate_url(url)
        await self._ensure_session()
        
        try:
            async with self._session.get(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise HTTPClientError(
                        f"HTTP {response.status}: {error_text}"
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise HTTPClientError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise HTTPClientError(f"Request timeout after {self.timeout}s")
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            'request_count': self._request_count,
            'total_bytes_downloaded': self._total_bytes_downloaded,
            'session_active': self._session is not None and not self._session.closed
        }

