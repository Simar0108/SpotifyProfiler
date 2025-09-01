#!/usr/bin/env python3
"""
Simple Schema Addition - Just adds new tables, no data processing
"""

import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_embedding_schema(db_path: str = "data/MPD/mpd_database.db"):
    """Just add the new schema tables - no data processing"""
    try:
        logger.info("🔧 Adding embedding schema tables...")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Track vocabulary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_vocab (
                    track_id TEXT PRIMARY KEY,
                    embedding_index INTEGER UNIQUE,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. Co-occurrence pairs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS co_occurrence_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track1_id TEXT,
                    track2_id TEXT,
                    playlist_id TEXT,
                    distance INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. Track embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_embeddings (
                    track_id TEXT PRIMARY KEY,
                    embedding_vector BLOB,
                    embedding_dim INTEGER DEFAULT 64,
                    model_version TEXT DEFAULT 'v1.0',
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 4. Clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    track_id TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_confidence REAL,
                    distance_to_centroid REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 5. Cluster metadata table
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
            
            # Create basic indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_vocab_index ON track_vocab (embedding_index)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_occurrence_track1 ON co_occurrence_pairs (track1_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_occurrence_track2 ON co_occurrence_pairs (track2_id)")
            
            conn.commit()
            logger.info("✅ Schema tables added successfully!")
            
    except Exception as e:
        logger.error(f"❌ Failed to add schema: {e}")
        raise

if __name__ == "__main__":
    add_embedding_schema()
