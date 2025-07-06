#!/usr/bin/env python3
"""
Robust Spotify Data Collector
Designed to run via cron every 12 hours
Handles errors gracefully and logs everything
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy.orm import Session
from database.database import SessionLocal, engine
from database.simple_models import Base, ListeningHistory
from config.settings import settings
from datetime import datetime, timezone
import logging
import time
from pathlib import Path

# Set up logging to both file and console
def setup_logging():
    """Set up comprehensive logging"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"spotify_collector_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class RobustSpotifyCollector:
    def __init__(self):
        self.logger = setup_logging()
        self.sp = None
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def setup_spotify_client(self):
        """Initialize Spotify client with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to initialize Spotify client (attempt {attempt + 1}/{self.max_retries})")
                
                self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id=settings.SPOTIFY_CLIENT_ID,
                    client_secret=settings.SPOTIFY_CLIENT_SECRET,
                    redirect_uri=settings.SPOTIFY_REDIRECT_URI,
                    scope="user-read-recently-played user-top-read user-read-private user-read-email"
                ))
                
                # Test the connection
                user = self.sp.current_user()
                self.logger.info(f"✅ Spotify client initialized successfully for user: {user['display_name']}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Spotify client (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("❌ Failed to initialize Spotify client after all retries")
                    return False
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=engine)
            self.logger.info("✅ Database tables created/verified")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to setup database: {e}")
            return False
    
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
        """Collect and store recent tracks with comprehensive error handling"""
        db = SessionLocal()
        try:
            self.logger.info("🎵 Starting to collect recent tracks...")
            
            # Get recently played tracks with retry logic
            recent_tracks = None
            for attempt in range(self.max_retries):
                try:
                    recent_tracks = self.sp.current_user_recently_played(limit=50)
                    self.logger.info(f"✅ Successfully fetched {len(recent_tracks['items'])} recent tracks")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Failed to fetch recent tracks (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            if not recent_tracks:
                self.logger.error("❌ No recent tracks data received")
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
                    self.logger.error(f"❌ Failed to process track {track_data.get('name', 'Unknown')}: {e}")
                    tracks_failed += 1
                    continue
            
            # Commit all changes
            db.commit()
            
            self.logger.info(f"✅ Collection completed:")
            self.logger.info(f"   - Added: {tracks_added} tracks")
            self.logger.info(f"   - Failed: {tracks_failed} tracks")
            
            return tracks_added
            
        except Exception as e:
            self.logger.error(f"❌ Error during collection: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def run_collection(self):
        """Main method to run the entire collection process"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Spotify data collection...")
        
        try:
            # Setup phase
            if not self.setup_spotify_client():
                return False
            
            if not self.setup_database():
                return False
            
            # Collection phase
            tracks_added = self.collect_recent_tracks()
            
            # Success summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"🎉 Collection completed successfully!")
            self.logger.info(f"   Duration: {duration:.2f} seconds")
            self.logger.info(f"   Tracks added: {tracks_added}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Collection failed: {e}")
            return False

def main():
    """Main entry point for the script"""
    collector = RobustSpotifyCollector()
    success = collector.run_collection()
    
    # Exit with appropriate code for cron
    if success:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()