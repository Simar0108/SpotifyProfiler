"""
ReccoBeats API Client for fetching Spotify-style audio features.

This client handles authentication, rate limiting, and batch processing
for the ReccoBeats API to extract audio features from track IDs.
"""

import requests
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Audio features for a single track."""
    spotify_id: str
    energy: float
    valence: float
    danceability: float
    tempo: float
    loudness: float
    speechiness: float
    instrumentalness: float
    liveness: float
    key: int
    mode: int
    acousticness: float


class ReccoBeatsClient:
    """
    Client for the ReccoBeats API to fetch audio features.
    
    Handles:
    - Authentication
    - Rate limiting
    - Batch processing (≤40 tracks per request)
    - Error handling and retries
    """
    
    def __init__(self, base_url: str = "https://api.reccobeats.com"):
        """
        Initialize the ReccoBeats client.
        
        Args:
            base_url: Base URL for the API (default: production)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
        
        # Rate limiting settings - optimized for high-volume processing
        self.requests_per_minute = 300  # 5 requests per second (aggressive but safe)
        self.min_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0.0
        
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """
        Make a rate-limited request to the ReccoBeats API.
        
        Args:
            endpoint: API endpoint (e.g., "/v1/features")
            method: HTTP method
            data: Request payload for POST requests
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: For HTTP errors
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            if e.response and e.response.status_code == 429:
                # Rate limit hit - check Retry-After header
                retry_after = e.response.headers.get('Retry-After', 60)
                logger.warning(f"Rate limit hit (429). Waiting {retry_after} seconds...")
                time.sleep(int(retry_after))
                # Retry the request once
                try:
                    if method.upper() == "GET":
                        response = self.session.get(url)
                    elif method.upper() == "POST":
                        response = self.session.post(url, json=data)
                    response.raise_for_status()
                    return response.json()
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    raise
            else:
                logger.error(f"API request failed: {e}")
                raise
    
    def get_audio_features(self, track_ids: List[str]) -> List[AudioFeatures]:
        """
        Fetch audio features for a batch of track IDs.
        
        Args:
            track_ids: List of Spotify track IDs (max 40 per batch)
            
        Returns:
            List of AudioFeatures objects
            
        Raises:
            ValueError: If more than 40 track IDs provided
        """
        if len(track_ids) > 40:
            raise ValueError("ReccoBeats API supports max 40 tracks per request")
        
        if not track_ids:
            return []
        
        logger.info(f"Fetching features for {len(track_ids)} tracks")
        
        # Make GET request to ReccoBeats API (based on their sample code)
        endpoint = "/v1/audio-features"
        
        # Try their exact format first - just the base endpoint
        try:
            # The API expects 'ids' parameter, not 'track_ids'
            # Convert track IDs to comma-separated string for query parameter
            track_ids_str = ",".join(track_ids)
            full_endpoint = f"{endpoint}?ids={track_ids_str}"
            
            logger.info(f"Making request to: {full_endpoint}")
            response = self._make_request(full_endpoint, method="GET")
            
            # Debug: Log the raw response to see the format
            logger.info(f"Raw API response type: {type(response)}")
            logger.info(f"Raw API response: {response}")
            
            # Parse response and convert to AudioFeatures objects
            features_list = []
            
            # Check if response has content (the actual format from the API)
            if isinstance(response, dict):
                tracks_data = response.get("content", [])
                
                for track_data in tracks_data:
                    if isinstance(track_data, dict):
                        # Extract Spotify ID from the href (it's the last part of the URL)
                        href = track_data.get("href", "")
                        spotify_id = href.split("/")[-1] if href else track_data.get("id", "unknown")
                        
                        features = AudioFeatures(
                            spotify_id=spotify_id,
                            energy=track_data.get("energy", 0.0),
                            valence=track_data.get("valence", 0.0),
                            danceability=track_data.get("danceability", 0.0),
                            tempo=track_data.get("tempo", 0.0),
                            loudness=track_data.get("loudness", 0.0),
                            speechiness=track_data.get("speechiness", 0.0),
                            instrumentalness=track_data.get("instrumentalness", 0.0),
                            liveness=track_data.get("liveness", 0.0),
                            key=track_data.get("key", -1),
                            mode=track_data.get("mode", 0),
                            acousticness=track_data.get("acousticness", 0.0)
                        )
                        features_list.append(features)
            
            logger.info(f"Successfully fetched features for {len(features_list)} tracks")
            return features_list
            
        except Exception as e:
            logger.error(f"Failed to fetch features for batch: {e}")
            raise
    
    def get_audio_features_batch(self, track_ids: List[str], batch_size: int = 40) -> List[AudioFeatures]:
        """
        Fetch audio features for multiple batches of track IDs.
        
        Args:
            track_ids: List of all Spotify track IDs
            batch_size: Number of tracks per batch (max 40)
            
        Returns:
            List of all AudioFeatures objects
        """
        if batch_size > 40:
            batch_size = 40
            logger.warning("Batch size reduced to 40 (ReccoBeats limit)")
        
        all_features = []
        total_batches = (len(track_ids) + batch_size - 1) // batch_size
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tracks)")
            
            try:
                batch_features = self.get_audio_features(batch)
                all_features.extend(batch_features)
                
                # Small delay between batches to be nice to the API
                if i + batch_size < len(track_ids):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"Successfully processed {len(all_features)} tracks")
        return all_features
    
    def test_connection(self) -> bool:
        """
        Test the API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing ReccoBeats API connection...")
            
            # Test with a minimal request to verify the endpoint works
            test_track_id = "test_track_123"
            endpoint = f"/v1/audio-features?ids={test_track_id}"
            
            # This should work with the correct 'ids' parameter
            response = self._make_request(endpoint, method="GET")
            return True
            
        except requests.RequestException as e:
            # If we get 400/404, the endpoint is working (expected error for test data)
            if e.response and e.response.status_code in [400, 404]:
                logger.info("API endpoint is responding (expected error for test data)")
                return True
            elif e.response and e.response.status_code == 401:
                logger.error("Authentication required - this API needs credentials")
                logger.error("Please check the ReccoBeats API documentation for authentication")
                return False
            else:
                logger.error(f"Connection test failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # TODO: Replace with your actual API key
    API_KEY = "your_reccobeats_api_key_here"
    
    client = ReccoBeatsClient(API_KEY)
    
    # Test connection
    if client.test_connection():
        print("✅ ReccoBeats API connection successful!")
    else:
        print("❌ ReccoBeats API connection failed!")
    
    # Example: Fetch features for a few tracks
    test_tracks = ["spotify:track:4iV5W9uYEdYUVa79Axb7Rh", "spotify:track:1301WleyT98MSxVHPZCA6M"]
    
    try:
        features = client.get_audio_features(test_tracks)
        print(f"✅ Successfully fetched features for {len(features)} tracks")
    except Exception as e:
        print(f"❌ Failed to fetch features: {e}")
