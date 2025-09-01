#!/usr/bin/env python3
"""
Quick Check - See what co-occurrence data survived the interruption
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_progress(db_path: str = "data/MPD/mpd_database.db"):
    """Check what co-occurrence data exists and estimate progress"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if co_occurrence_pairs table has data
            cursor.execute("SELECT COUNT(*) FROM co_occurrence_pairs")
            existing_pairs = cursor.fetchone()[0]
            
            # Check total playlists
            cursor.execute("SELECT COUNT(*) FROM mpd_playlists")
            total_playlists = cursor.fetchone()[0]
            
            # Check if we can estimate progress
            if existing_pairs > 0:
                logger.info(f"✅ Found {existing_pairs:,} existing co-occurrence pairs")
                
                # Get some sample data to understand what was processed
                cursor.execute("SELECT COUNT(DISTINCT playlist_id) FROM co_occurrence_pairs")
                processed_playlists = cursor.fetchone()[0]
                
                logger.info(f"📊 Progress Estimate:")
                logger.info(f"   Total playlists: {total_playlists:,}")
                logger.info(f"   Processed playlists: {processed_playlists:,}")
                logger.info(f"   Progress: {processed_playlists/total_playlists*100:.1f}%")
                
                # Check what the last processed playlist was
                cursor.execute("""
                    SELECT playlist_id, COUNT(*) as pair_count 
                    FROM co_occurrence_pairs 
                    GROUP BY playlist_id 
                    ORDER BY playlist_id DESC 
                    LIMIT 5
                """)
                last_playlists = cursor.fetchall()
                
                logger.info(f"🔍 Last processed playlists:")
                for playlist_id, pair_count in last_playlists:
                    logger.info(f"   {playlist_id[:8]}... ({pair_count:,} pairs)")
                
                # Estimate remaining work
                remaining_playlists = total_playlists - processed_playlists
                logger.info(f"📋 Remaining work: {remaining_playlists:,} playlists")
                
            else:
                logger.info("❌ No co-occurrence data found - starting from scratch")
                logger.info(f"📊 Total playlists to process: {total_playlists:,}")
            
    except Exception as e:
        logger.error(f"❌ Error checking progress: {e}")

if __name__ == "__main__":
    check_progress()
