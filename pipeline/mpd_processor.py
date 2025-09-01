#!/usr/bin/env python3
"""
MPD (Million Playlist Dataset) Processor
Handles parsing MPD JSON files and extracting track information for contrastive learning.
"""

import json
import logging
import os
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import sqlite3
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MPDProcessor:
    """Processes MPD dataset to extract tracks and playlist information"""
    
    def __init__(self, data_dir: str = "data/MPD/data", db_path: str = "/Volumes/Simar -Seagate /mpd_database.db"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.db_conn = None
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ MPD Processor initialized for directory: {self.data_dir}")
    
    def setup_database(self):
        """Create database tables for storing MPD data and track features"""
        try:
            # Add timeout to prevent hanging on slow external drives
            self.db_conn = sqlite3.connect(self.db_path, timeout=30.0)
            
            # Optimize database performance for bulk inserts
            self.db_conn.execute("PRAGMA journal_mode = WAL")
            self.db_conn.execute("PRAGMA synchronous = NORMAL")
            self.db_conn.execute("PRAGMA cache_size = 10000")
            self.db_conn.execute("PRAGMA temp_store = MEMORY")
            
            cursor = self.db_conn.cursor()
            
            # Table for tracking processing progress
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_progress (
                    file_name TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    playlists_count INTEGER,
                    tracks_count INTEGER,
                    file_size_bytes INTEGER,
                    status TEXT DEFAULT 'completed'
                )
            """)
            
            # Table for storing MPD playlists (updated to match actual MPD structure)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mpd_playlists (
                    playlist_id TEXT PRIMARY KEY,
                    name TEXT,
                    collaborative TEXT,
                    num_tracks INTEGER,
                    duration_ms INTEGER,
                    num_edits INTEGER,
                    modified_at INTEGER,
                    num_artists INTEGER,
                    num_albums INTEGER,
                    num_followers INTEGER
                )
            """)
            
            # Table for storing MPD tracks (updated to match actual MPD structure)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mpd_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    playlist_id TEXT,
                    track_uri TEXT,
                    track_name TEXT,
                    artist_name TEXT,
                    artist_uri TEXT,
                    album_name TEXT,
                    album_uri TEXT,
                    duration_ms INTEGER,
                    pos INTEGER,
                    FOREIGN KEY (playlist_id) REFERENCES mpd_playlists (playlist_id)
                )
            """)
            
            # Table for storing ReccoBeats audio features (will be populated later)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_features (
                    spotify_id TEXT PRIMARY KEY,
                    energy REAL,
                    valence REAL,
                    danceability REAL,
                    tempo REAL,
                    loudness REAL,
                    speechiness REAL,
                    instrumentalness REAL,
                    liveness REAL,
                    key INTEGER,
                    mode INTEGER,
                    acousticness REAL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_uri ON mpd_tracks (track_uri)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlist_id ON mpd_tracks (playlist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spotify_id ON track_features (spotify_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_name ON processing_progress (file_name)")
            
            self.db_conn.commit()
            logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            raise
    
    def setup_embedding_schema(self):
        """Add embedding-related tables to existing database"""
        try:
            cursor = self.db_conn.cursor()
            
            # Track vocabulary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_vocab (
                    track_id TEXT PRIMARY KEY,
                    embedding_index INTEGER UNIQUE,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Co-occurrence pairs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS co_occurrence_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track1_id TEXT,
                    track2_id TEXT,
                    playlist_id TEXT,
                    distance INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track1_id) REFERENCES track_vocab (track_id),
                    FOREIGN KEY (track2_id) REFERENCES track_vocab (track_id)
                )
            """)
            
            # Track embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_embeddings (
                    track_id TEXT PRIMARY KEY,
                    embedding_vector BLOB,
                    embedding_dim INTEGER DEFAULT 64,
                    model_version TEXT DEFAULT 'v1.0',
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES track_vocab (track_id)
                )
            """)
            
            # Clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    track_id TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_confidence REAL,
                    distance_to_centroid REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES track_vocab (track_id)
                )
            """)
            
            # Cluster metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_metadata (
                    cluster_id INTEGER PRIMARY KEY,
                    cluster_name TEXT,
                    description TEXT,
                    centroid_vector BLOB,
                    track_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_vocab_index ON track_vocab (embedding_index)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_occurrence_track1 ON co_occurrence_pairs (track1_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_occurrence_track2 ON co_occurrence_pairs (track2_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_occurrence_playlist ON co_occurrence_pairs (playlist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON clusters (cluster_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_metadata_id ON cluster_metadata (cluster_id)")
            
            self.db_conn.commit()
            logger.info("✅ Embedding schema tables added successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup embedding schema: {e}")
            raise
    
    def get_processed_files(self) -> Set[str]:
        """Get set of already processed file names"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT file_name FROM processing_progress WHERE status = 'completed'")
        processed_files = {row[0] for row in cursor.fetchall()}
        return processed_files
    
    def mark_file_processed(self, file_name: str, playlists_count: int, tracks_count: int, file_size_bytes: int):
        """Mark a file as successfully processed"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO processing_progress 
            (file_name, processed_at, playlists_count, tracks_count, file_size_bytes, status)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, 'completed')
        """, (file_name, playlists_count, tracks_count, file_size_bytes))
        self.db_conn.commit()
    
    def mark_file_failed(self, file_name: str, error_message: str):
        """Mark a file as failed with error details"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO processing_progress 
            (file_name, processed_at, status)
            VALUES (?, CURRENT_TIMESTAMP, ?)
        """, (file_name, f"failed: {error_message}"))
        self.db_conn.commit()
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processing progress"""
        cursor = self.db_conn.cursor()
        
        # Get total files processed
        cursor.execute("SELECT COUNT(*) FROM processing_progress WHERE status = 'completed'")
        files_processed = cursor.fetchone()[0]
        
        # Get total playlists and tracks
        cursor.execute("SELECT COUNT(*) FROM mpd_playlists")
        total_playlists = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mpd_tracks")
        total_tracks = cursor.fetchone()[0]
        
        # Get unique tracks
        cursor.execute("SELECT COUNT(DISTINCT track_uri) FROM mpd_tracks WHERE track_uri LIKE 'spotify:track:%'")
        unique_tracks = cursor.fetchone()[0]
        
        return {
            'files_processed': files_processed,
            'total_playlists': total_playlists,
            'total_tracks': total_tracks,
            'unique_tracks': unique_tracks
        }
    
    def extract_spotify_id(self, track_uri: str) -> Optional[str]:
        """Extract Spotify track ID from track URI"""
        if track_uri.startswith("spotify:track:"):
            return track_uri.replace("spotify:track:", "")
        return None
    
    def process_mpd_file(self, file_path: Path) -> Tuple[int, int]:
        """Process a single MPD JSON file"""
        try:
            import time
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            playlists = data.get('playlists', [])
            total_playlists = len(playlists)
            total_tracks = 0
            
            cursor = self.db_conn.cursor()
            
            for playlist in playlists:
                # Insert playlist info (updated to match actual MPD structure)
                cursor.execute("""
                    INSERT OR REPLACE INTO mpd_playlists 
                    (playlist_id, name, collaborative, num_tracks, duration_ms, num_edits, 
                     modified_at, num_artists, num_albums, num_followers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(playlist.get('pid')),
                    playlist.get('name', ''),
                    playlist.get('collaborative', 'false'),
                    playlist.get('num_tracks', 0),
                    playlist.get('duration_ms', 0),
                    playlist.get('num_edits', 0),
                    playlist.get('modified_at', 0),
                    playlist.get('num_artists', 0),
                    playlist.get('num_albums', 0),
                    playlist.get('num_followers', 0)
                ))
                
                # Insert tracks (updated to match actual MPD structure)
                tracks = playlist.get('tracks', [])
                for track in tracks:
                    spotify_id = self.extract_spotify_id(track.get('track_uri', ''))
                    if spotify_id:
                        cursor.execute("""
                            INSERT OR REPLACE INTO mpd_tracks 
                            (playlist_id, track_uri, track_name, artist_name, artist_uri, 
                             album_name, album_uri, duration_ms, pos)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(playlist.get('pid')),
                            track.get('track_uri', ''),
                            track.get('track_name', ''),
                            track.get('artist_name', ''),
                            track.get('artist_uri', ''),
                            track.get('album_name', ''),
                            track.get('album_uri', ''),
                            track.get('duration_ms', 0),
                            track.get('pos', 0)
                        ))
                        total_tracks += 1
            
            self.db_conn.commit()
            
            # Mark file as processed
            file_size = file_path.stat().st_size
            self.mark_file_processed(file_path.name, total_playlists, total_tracks, file_size)
            
            processing_time = time.time() - start_time
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"✅ Processed {file_path.name}: {total_playlists} playlists, {total_tracks} tracks in {processing_time:.2f}s (file: {file_size_mb:.1f}MB)")
            return total_playlists, total_tracks
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            self.mark_file_failed(file_path.name, str(e))
            return 0, 0
    
    def process_all_mpd_files(self) -> Tuple[int, int]:
        """Process all MPD JSON files in the data directory, skipping already processed files"""
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"⚠️ No JSON files found in {self.data_dir}")
            logger.info("💡 Download MPD dataset and place JSON files in data/MPD/data/ directory")
            return 0, 0
        
        # Get already processed files
        processed_files = self.get_processed_files()
        logger.info(f"📊 Found {len(processed_files)} already processed files")
        
        # Filter out already processed files
        unprocessed_files = [f for f in json_files if f.name not in processed_files]
        
        if not unprocessed_files:
            logger.info("🎉 All files have already been processed!")
            return 0, 0
        
        logger.info(f"🔄 Processing {len(unprocessed_files)} new files out of {len(json_files)} total...")
        
        total_playlists = 0
        total_tracks = 0
        
        for i, file_path in enumerate(unprocessed_files, 1):
            logger.info(f"📁 Processing file {i}/{len(unprocessed_files)}: {file_path.name}")
            playlists, tracks = self.process_mpd_file(file_path)
            total_playlists += playlists
            total_tracks += tracks
        
        logger.info(f"✅ New files processed: {len(unprocessed_files)} files, {total_playlists} playlists, {total_tracks} tracks")
        return total_playlists, total_tracks
    
    def process_subset_mpd_files(self, max_files: int = 100) -> Tuple[int, int]:
        """Process only a subset of MPD JSON files for testing/validation"""
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"⚠️ No JSON files found in {self.data_dir}")
            logger.info("💡 Download MPD dataset and place JSON files in data/MPD/data/ directory")
            return 0, 0
        
        # Get already processed files
        processed_files = self.get_processed_files()
        logger.info(f"📊 Found {len(processed_files)} already processed files")
        
        # Filter out already processed files
        unprocessed_files = [f for f in json_files if f.name not in processed_files]
        
        if not unprocessed_files:
            logger.info("🎉 All files have already been processed!")
            return 0, 0
        
        # Limit to subset for testing
        subset_files = unprocessed_files[:max_files]
        logger.info(f"🧪 Processing subset: {len(subset_files)} files out of {len(unprocessed_files)} unprocessed files")
        logger.info(f"📁 Total available files: {len(json_files)}")
        
        total_playlists = 0
        total_tracks = 0
        
        for i, file_path in enumerate(subset_files, 1):
            logger.info(f"📁 Processing file {i}/{len(subset_files)}: {file_path.name}")
            playlists, tracks = self.process_mpd_file(file_path)
            total_playlists += playlists
            total_tracks += tracks
        
        logger.info(f"✅ Subset processed: {len(subset_files)} files, {total_playlists} playlists, {total_tracks} tracks")
        logger.info(f"💡 To process remaining files, run with process_all_mpd_files()")
        return total_playlists, total_tracks
    
    def get_unique_track_ids(self) -> List[str]:
        """Get list of unique Spotify track IDs from processed MPD data"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT DISTINCT spotify_id FROM mpd_tracks WHERE spotify_id IS NOT NULL")
        track_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"📊 Found {len(track_ids)} unique track IDs")
        return track_ids
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Get all tracks for a specific playlist"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT track_uri, track_name, artist_name, artist_uri, album_name, album_uri, duration_ms, pos
            FROM mpd_tracks 
            WHERE playlist_id = ? 
            ORDER BY pos
        """, (playlist_id,))
        
        tracks = []
        for row in cursor.fetchall():
            tracks.append({
                'track_uri': row[0],
                'track_name': row[1],
                'artist_name': row[2],
                'artist_uri': row[3],
                'album_name': row[4],
                'album_uri': row[5],
                'duration_ms': row[6],
                'pos': row[7]
            })
        
        return tracks
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

def main():
    """Main function to test MPD processing"""
    print("🔌 Connecting to database...")
    processor = MPDProcessor(db_path="/Volumes/Simar -Seagate /mpd_database.db")
    
    try:
        # Setup database
        print("📊 Setting up database...")
        processor.setup_database()
        print("✅ Database setup complete")
        
        # Show current progress
        print("📈 Getting current progress...")
        try:
            summary = processor.get_processing_summary()
            print(f"\n📊 Current Progress:")
            print(f"   Files Processed: {summary['files_processed']}")
            print(f"   Total Playlists: {summary['total_playlists']:,}")
            print(f"   Total Tracks: {summary['total_tracks']:,}")
            print(f"   Unique Tracks: {summary['unique_tracks']:,}")
        except Exception as e:
            print(f"⚠️ Could not get progress summary: {e}")
            print("🔄 This is normal for first run or if database is empty")
            print("🔄 Continuing with processing...")
        
        # Process MPD files - START WITH SUBSET FOR TESTING
        print("\n🧪 Starting with subset processing for validation...")
        try:
            playlists, tracks = processor.process_subset_mpd_files(max_files=150)  # Process 150 files for robust training data
            
            if playlists > 0:
                # Get updated summary
                try:
                    updated_summary = processor.get_processing_summary()
                    print(f"\n📊 Updated Summary:")
                    print(f"   Files Processed: {updated_summary['files_processed']}")
                    print(f"   Total Playlists: {updated_summary['total_playlists']:,}")
                    print(f"   Total Tracks: {updated_summary['total_tracks']:,}")
                    print(f"   Unique Tracks: {updated_summary['unique_tracks']:,}")
                except Exception as e:
                    print(f"⚠️ Could not get final summary: {e}")
                    print(f"✅ But processing completed successfully!")
                
                print(f"\n💡 Next steps:")
                print(f"   1. Test ReccoBeats API with this subset")
                print(f"   2. Validate your pipeline works")
                print(f"   3. If successful, run: processor.process_all_mpd_files()")
            else:
                print("\n💡 No new MPD data to process!")
                print("   Download the dataset from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge")
                print("   Place JSON files in: data/MPD/data/")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            print("💡 Check your MPD data directory and database connection")
    
    finally:
        processor.close()

if __name__ == "__main__":
    main()
