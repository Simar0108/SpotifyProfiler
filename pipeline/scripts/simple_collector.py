#!/usr/bin/env python3
"""
Simple Spotify Data Collector
Basic version for manual testing and development
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy.orm import Session
from database.database import SessionLocal, engine
from database.simple_models import Base, ListeningHistory
from config.settings import settings
from datetime import datetime, timezone
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSpotifyCollector:
    def __init__(self):
        self.sp = None
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.setup_spotify_client()
        self.setup_database()
    
    def setup_spotify_client(self):
        """Initialize Spotify client with OAuth2"""
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET,
                redirect_uri=settings.SPOTIFY_REDIRECT_URI,
                scope="user-read-recently-played user-top-read user-read-private user-read-email"
            ))
            logger.info("Spotify client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    
    def categorize_time_of_day(self, hour: int) -> str:
        """Categorize hour into time of day"""
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Night"
    
    def get_day_of_week(self, dt: datetime) -> str:
        """Get day of week as string"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[dt.weekday()]
    
    def collect_recent_tracks(self):
        """Collect and store recent tracks"""
        db = SessionLocal()
        try:
            logger.info("🎵 Starting to collect recent tracks...")
            
            # Get recently played tracks with retry logic
            recent_tracks = None
            for attempt in range(self.max_retries):
                try:
                    recent_tracks = self.sp.current_user_recently_played(limit=50)
                    logger.info(f"✅ Successfully fetched {len(recent_tracks['items'])} recent tracks")
                    break
                except Exception as e:
                    logger.error(f"❌ Failed to fetch recent tracks (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            if not recent_tracks:
                logger.error("❌ No recent tracks data received")
                return 0
            
            tracks_added = 0
            tracks_failed = 0
            
            for item in recent_tracks['items']:
                try:
                    track_data = item['track']
                    played_at = datetime.fromisoformat(item['played_at'].replace('Z', '+00:00'))
                    
                    # Create listening history record (no duplicate checking)
                    listening_record = ListeningHistory(
                        track_name=track_data['name'],
                        artist_name=track_data['artists'][0]['name'],
                        spotify_id=track_data['id'],
                        played_at=played_at,
                        local_time=played_at,  # For now, same as UTC
                        day_of_week=self.get_day_of_week(played_at),
                        time_of_day=self.categorize_time_of_day(played_at.hour)
                    )
                    
                    db.add(listening_record)
                    tracks_added += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process track {track_data.get('name', 'Unknown')}: {e}")
                    tracks_failed += 1
                    continue
            
            # Commit all changes
            db.commit()
            
            logger.info(f"✅ Collection completed:")
            logger.info(f"   - Added: {tracks_added} tracks")
            logger.info(f"   - Failed: {tracks_failed} tracks")
            
            return tracks_added
            
        except Exception as e:
            logger.error(f"❌ Error during collection: {e}")
            db.rollback()
            raise
        finally:
            db.close()

def main():
    """Main function to run data collection"""
    try:
        collector = SimpleSpotifyCollector()
        tracks_added = collector.collect_recent_tracks()
        print(f"✅ Data collection completed! Added {tracks_added} new listening records.")
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        raise

if __name__ == "__main__":
    main() 