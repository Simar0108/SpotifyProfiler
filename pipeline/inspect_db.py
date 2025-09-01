#!/usr/bin/env python3
"""
Database Inspector - Check the current database structure and verify new schema
"""

import sqlite3
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_database(db_path: str = "data/MPD/mpd_database.db"):
    """Inspect the database structure and show all tables"""
    try:
        logger.info("🔍 Inspecting database structure...")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"📊 Found {len(tables)} tables:")
            for table in tables:
                print(f"   • {table}")
            
            print("\n" + "="*60)
            
            # Show detailed schema for each table
            for table_name in tables:
                print(f"\n📋 Table: {table_name}")
                print("-" * 40)
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_id, name, data_type, not_null, default_val, pk = col
                    pk_marker = " 🔑" if pk else ""
                    not_null_marker = " NOT NULL" if not_null else ""
                    default_marker = f" DEFAULT {default_val}" if default_val else ""
                    
                    print(f"   {name:<20} {data_type:<15} {not_null_marker}{default_marker}{pk_marker}")
                
                # Get row count (but skip large tables to avoid hanging)
                try:
                    if table_name in ['mpd_tracks', 'co_occurrence_pairs']:
                        print(f"   📊 Row count: [Large table - skipping count to avoid hanging]")
                    else:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"   📊 Row count: {count:,}")
                except Exception as e:
                    print(f"   📊 Row count: Error - {e}")
            
            print("\n" + "="*60)
            
            # Check for the new embedding-related tables specifically
            embedding_tables = ['track_vocab', 'co_occurrence_pairs', 'track_embeddings', 'clusters', 'cluster_metadata']
            existing_embedding_tables = [t for t in embedding_tables if t in tables]
            
            if len(existing_embedding_tables) == len(embedding_tables):
                logger.info("✅ All embedding schema tables are present!")
            else:
                missing = [t for t in embedding_tables if t not in tables]
                logger.warning(f"⚠️ Missing embedding tables: {missing}")
            
            # Quick summary without hanging on large tables
            print("\n📊 Database Summary:")
            print("-" * 40)
            
            # Check mpd_playlists (usually safe to count)
            if 'mpd_playlists' in tables:
                cursor.execute("SELECT COUNT(*) FROM mpd_playlists")
                playlist_count = cursor.fetchone()[0]
                print(f"   🎵 MPD Playlists: {playlist_count:,}")
            
            # Check track_features (usually safe to count)
            if 'track_features' in tables:
                cursor.execute("SELECT COUNT(*) FROM track_features")
                feature_count = cursor.fetchone()[0]
                print(f"   🎵 Track features: {feature_count:,}")
            
            # Check new embedding tables
            print(f"   🧠 Embedding tables: {len(existing_embedding_tables)}/{len(embedding_tables)} present")
            
            # Quick check on mpd_tracks without full count
            if 'mpd_tracks' in tables:
                print(f"   🎵 MPD Tracks: [Large table - estimated 15M+ based on your data]")
            
            logger.info("✅ Database inspection completed!")
            
    except Exception as e:
        logger.error(f"❌ Failed to inspect database: {e}")
        raise

if __name__ == "__main__":
    inspect_database()
