#!/usr/bin/env python3
"""
Build Co-occurrence Pairs - Generate training data for item2vec model

This script:
1. Reads playlist sequences from mpd_tracks table
2. Generates co-occurrence pairs within ±3 position context window
3. Stores pairs in co_occurrence_pairs table for neural network training
4. Includes robust checkpointing and resume functionality

The result: training data that teaches the model which tracks "go together" musically.
"""

import sqlite3
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CoOccurrenceBuilder:
    """Builds co-occurrence pairs from playlist sequences with checkpointing"""
    
    def __init__(self, db_path: str = "data/MPD/mpd_database.db", context_window: int = 3):
        self.db_path = db_path
        self.db_conn = None
        self.context_window = context_window  # ±3 positions by default
        self.batch_size = 50000  # Process playlists in batches
        self.checkpoint_file = "co_occurrence_checkpoint.json"
        
    def connect_database(self):
        """Connect to database with optimizations for bulk operations"""
        try:
            self.db_conn = sqlite3.connect(self.db_path, timeout=60.0)
            
            # Optimize for bulk operations
            self.db_conn.execute("PRAGMA journal_mode = WAL")
            self.db_conn.execute("PRAGMA synchronous = OFF")
            self.db_conn.execute("PRAGMA cache_size = 50000")
            self.db_conn.execute("PRAGMA temp_store = MEMORY")
            self.db_conn.execute("PRAGMA mmap_size = 268435456")
            
            logger.info("✅ Connected to database with bulk operation optimizations")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def load_checkpoint(self):
        """Load checkpoint data if it exists"""
        try:
            if Path(self.checkpoint_file).exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                logger.info(f"📂 Loaded checkpoint from {self.checkpoint_file}")
                logger.info(f"   Last processed playlist: {checkpoint.get('last_playlist_id', 'None')}")
                logger.info(f"   Processed playlists: {checkpoint.get('processed_count', 0):,}")
                logger.info(f"   Total pairs generated: {checkpoint.get('total_pairs', 0):,}")
                logger.info(f"   Checkpoint time: {checkpoint.get('timestamp', 'Unknown')}")
                
                return checkpoint
            else:
                logger.info("📂 No checkpoint found - starting fresh")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def save_checkpoint(self, last_playlist_id, processed_count, total_pairs):
        """Save checkpoint data"""
        try:
            checkpoint = {
                'last_playlist_id': last_playlist_id,
                'processed_count': processed_count,
                'total_pairs': total_pairs,
                'timestamp': datetime.now().isoformat(),
                'context_window': self.context_window
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"💾 Checkpoint saved: {last_playlist_id} ({processed_count:,} playlists)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save checkpoint: {e}")
    
    def get_resume_point(self):
        """Determine where to resume processing"""
        try:
            cursor = self.db_conn.cursor()
            
            # Check if we have existing data
            cursor.execute("SELECT COUNT(*) FROM co_occurrence_pairs")
            existing_pairs = cursor.fetchone()[0]
            
            if existing_pairs == 0:
                logger.info("🔄 Starting fresh - no existing data")
                return 0, 0
            
            # Find the highest playlist ID that was processed
            cursor.execute("""
                SELECT MAX(playlist_id) as max_playlist
                FROM co_occurrence_pairs
            """)
            max_playlist = cursor.fetchone()[0]
            
            if max_playlist is None:
                logger.info("🔄 Starting fresh - no valid playlist data")
                return 0, 0
            
            # Count how many playlists we've processed
            cursor.execute("""
                SELECT COUNT(DISTINCT playlist_id) as processed_count
                FROM co_occurrence_pairs
            """)
            processed_count = cursor.fetchone()[0]
            
            # Find the next playlist ID to process
            cursor.execute("""
                SELECT playlist_id
                FROM mpd_playlists
                WHERE playlist_id > ?
                ORDER BY playlist_id
                LIMIT 1
            """, (max_playlist,))
            
            next_playlist = cursor.fetchone()
            if next_playlist:
                resume_offset = next_playlist[0]
            else:
                # We've processed all playlists
                resume_offset = None
            
            logger.info(f"🔄 Resume point: {max_playlist} (processed {processed_count:,} playlists)")
            logger.info(f"   Next playlist to process: {resume_offset}")
            
            return resume_offset, processed_count
            
        except Exception as e:
            logger.error(f"❌ Failed to determine resume point: {e}")
            return 0, 0
    
    def get_playlist_stats(self):
        """Get overview of playlists to process"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("📊 Analyzing playlist data...")
            
            # Get total playlists
            cursor.execute("SELECT COUNT(*) FROM mpd_playlists")
            total_playlists = cursor.fetchone()[0]
            
            # Get playlists with track counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_playlists,
                    AVG(track_count) as avg_tracks,
                    MIN(track_count) as min_tracks,
                    MAX(track_count) as max_tracks
                FROM (
                    SELECT playlist_id, COUNT(*) as track_count
                    FROM mpd_tracks 
                    WHERE track_uri LIKE 'spotify:track:%'
                    GROUP BY playlist_id
                )
            """)
            
            stats = cursor.fetchone()
            total_playlists, avg_tracks, min_tracks, max_tracks = stats
            
            logger.info(f"📊 Playlist Overview:")
            logger.info(f"   Total playlists: {total_playlists:,}")
            logger.info(f"   Average tracks per playlist: {avg_tracks:.1f}")
            logger.info(f"   Track range: {min_tracks} to {max_tracks}")
            
            return total_playlists, avg_tracks, min_tracks, max_tracks
            
        except Exception as e:
            logger.error(f"❌ Failed to get playlist stats: {e}")
            raise
    
    def get_playlist_batch(self, offset: int, limit: int):
        """Get a batch of playlists with their tracks ordered by position"""
        try:
            cursor = self.db_conn.cursor()
            
            # Get playlists with tracks ordered by position
            cursor.execute("""
                SELECT 
                    t.playlist_id,
                    t.track_uri,
                    t.pos
                FROM mpd_tracks t
                INNER JOIN (
                    SELECT DISTINCT playlist_id 
                    FROM mpd_tracks 
                    WHERE track_uri LIKE 'spotify:track:%'
                    ORDER BY playlist_id
                    LIMIT ? OFFSET ?
                ) p ON t.playlist_id = p.playlist_id
                WHERE t.track_uri LIKE 'spotify:track:%'
                ORDER BY t.playlist_id, t.pos
            """, (limit, offset))
            
            # Group by playlist
            playlists = defaultdict(list)
            for row in cursor.fetchall():
                playlist_id, track_uri, pos = row
                track_id = track_uri.replace('spotify:track:', '')
                playlists[playlist_id].append((track_id, pos))
            
            return playlists
            
        except Exception as e:
            logger.error(f"❌ Failed to get playlist batch: {e}")
            raise
    
    def generate_co_occurrence_pairs(self, playlists):
        """Generate co-occurrence pairs from playlist sequences"""
        try:
            pairs = []
            
            for playlist_id, tracks in playlists.items():
                if len(tracks) < 2:
                    continue
                
                # Sort by position to ensure correct order
                tracks.sort(key=lambda x: x[1])
                track_ids = [t[0] for t in tracks]
                
                # Generate pairs within context window
                for i, track1_id in enumerate(track_ids):
                    # Look at tracks within ±context_window positions
                    start_idx = max(0, i - self.context_window)
                    end_idx = min(len(track_ids), i + self.context_window + 1)
                    
                    for j in range(start_idx, end_idx):
                        if i != j:  # Don't pair track with itself
                            track2_id = track_ids[j]
                            distance = abs(i - j)
                            
                            # Store the pair
                            pairs.append((track1_id, track2_id, playlist_id, distance))
            
            return pairs
            
        except Exception as e:
            logger.error(f"❌ Failed to generate co-occurrence pairs: {e}")
            raise
    
    def populate_co_occurrence_table(self, pairs):
        """Populate co_occurrence_pairs table with generated pairs"""
        try:
            cursor = self.db_conn.cursor()
            
            if not pairs:
                logger.info("📝 No new pairs to insert")
                return
            
            logger.info(f"📝 Inserting {len(pairs):,} new co-occurrence pairs...")
            
            # Begin bulk insert
            start_time = time.time()
            total_pairs = len(pairs)
            
            # Process in batches
            for batch_start in range(0, total_pairs, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_pairs)
                batch = pairs[batch_start:batch_end]
                
                # Bulk insert batch
                cursor.executemany("""
                    INSERT INTO co_occurrence_pairs (track1_id, track2_id, playlist_id, distance)
                    VALUES (?, ?, ?, ?)
                """, batch)
                
                # Progress update
                elapsed = time.time() - start_time
                processed = batch_end
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_pairs - processed) / rate if rate > 0 else 0
                
                logger.info(f"📝 Batch {batch_start//self.batch_size + 1}: "
                          f"Processed {processed:,}/{total_pairs:,} pairs "
                          f"({processed/total_pairs*100:.1f}%) "
                          f"Rate: {rate:.0f} pairs/sec "
                          f"ETA: {eta/60:.1f} minutes")
                
                # Commit every 10 batches
                if (batch_start // self.batch_size + 1) % 10 == 0:
                    self.db_conn.commit()
                    logger.info(f"💾 Committed batch {batch_start//self.batch_size + 1}")
            
            # Final commit
            self.db_conn.commit()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Inserted {total_pairs:,} pairs in {elapsed:.1f}s")
            logger.info(f"   Overall rate: {total_pairs/elapsed:.0f} pairs/sec")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate co_occurrence_pairs: {e}")
            raise
    
    def verify_co_occurrence_data(self):
        """Verify the co-occurrence data was generated correctly"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("🔍 Verifying co-occurrence data...")
            
            # Get basic stats
            cursor.execute("SELECT COUNT(*) FROM co_occurrence_pairs")
            total_pairs = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(distance), MAX(distance), AVG(distance) FROM co_occurrence_pairs")
            min_dist, max_dist, avg_dist = cursor.fetchone()
            
            # Get unique tracks involved
            cursor.execute("SELECT COUNT(DISTINCT track1_id) FROM co_occurrence_pairs")
            unique_track1 = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT track2_id) FROM co_occurrence_pairs")
            unique_track2 = cursor.fetchone()[0]
            
            # Get unique playlists
            cursor.execute("SELECT COUNT(DISTINCT playlist_id) FROM co_occurrence_pairs")
            unique_playlists = cursor.fetchone()[0]
            
            # Distance distribution
            cursor.execute("""
                SELECT 
                    distance,
                    COUNT(*) as count
                FROM co_occurrence_pairs 
                GROUP BY distance
                ORDER BY distance
            """)
            distance_dist = cursor.fetchall()
            
            logger.info("📊 Co-occurrence Statistics:")
            logger.info(f"   Total pairs: {total_pairs:,}")
            logger.info(f"   Unique track1: {unique_track1:,}")
            logger.info(f"   Unique track2: {unique_track2:,}")
            logger.info(f"   Unique playlists: {unique_playlists:,}")
            logger.info(f"   Distance range: {min_dist} to {max_dist}")
            logger.info(f"   Average distance: {avg_dist:.1f}")
            
            logger.info("📊 Distance Distribution:")
            for distance, count in distance_dist:
                percentage = (count / total_pairs) * 100
                logger.info(f"   Distance {distance}: {count:>8,} pairs ({percentage:>5.1f}%)")
            
            # Show some sample pairs
            cursor.execute("""
                SELECT track1_id, track2_id, distance, playlist_id 
                FROM co_occurrence_pairs 
                LIMIT 10
            """)
            sample_pairs = cursor.fetchall()
            
            logger.info("🔗 Sample Co-occurrence Pairs:")
            for i, (track1, track2, dist, playlist) in enumerate(sample_pairs, 1):
                logger.info(f"   {i:>2}. {track1[:8]}... + {track2[:8]}... (dist: {dist}, playlist: {playlist[:8]}...)")
            
        except Exception as e:
            logger.error(f"❌ Failed to verify co-occurrence data: {e}")
            raise
    
    def build_co_occurrence_data(self):
        """Main method to build co-occurrence pairs with checkpointing"""
        try:
            start_time = time.time()
            logger.info("🚀 Starting co-occurrence pair generation...")
            logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🎯 Context window: ±{self.context_window} positions")
            
            # Step 1: Connect to database
            self.connect_database()
            
            # Step 2: Load checkpoint and determine resume point
            checkpoint = self.load_checkpoint()
            resume_offset, processed_count = self.get_resume_point()
            
            if resume_offset is None:
                logger.info("🎉 All playlists have been processed!")
                self.verify_co_occurrence_data()
                return
            
            # Step 3: Analyze playlist data
            total_playlists, avg_tracks, min_tracks, max_tracks = self.get_playlist_stats()
            
            # Step 4: Process remaining playlists in batches
            all_pairs = []
            current_processed = processed_count
            
            # Calculate starting offset for remaining playlists
            # Convert resume_offset to integer if it's a string playlist ID
            if isinstance(resume_offset, str):
                # Find the offset in the playlist sequence
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT DISTINCT playlist_id 
                        FROM mpd_tracks 
                        WHERE track_uri LIKE 'spotify:track:%'
                        ORDER BY playlist_id
                        LIMIT ?
                    )
                """, (resume_offset,))
                start_offset = cursor.fetchone()[0]
            else:
                start_offset = resume_offset if resume_offset > 0 else 0
            
            remaining_playlists = total_playlists - processed_count
            logger.info(f"🔄 Resuming from playlist offset {start_offset} ({remaining_playlists:,} playlists remaining)")
            
            for batch_offset in range(start_offset, total_playlists, self.batch_size):
                batch_limit = min(self.batch_size, total_playlists - batch_offset)
                
                logger.info(f"🔄 Processing playlist batch: {batch_offset:,} to {batch_offset + batch_limit:,}")
                
                # Get playlist batch
                playlists = self.get_playlist_batch(batch_offset, batch_limit)
                
                # Generate pairs for this batch
                batch_pairs = self.generate_co_occurrence_pairs(playlists)
                all_pairs.extend(batch_pairs)
                
                current_processed += len(playlists)
                
                # Save checkpoint after each batch
                if playlists:
                    last_playlist_id = max(playlists.keys())
                    self.save_checkpoint(last_playlist_id, current_processed, len(all_pairs))
                
                logger.info(f"✅ Batch complete: {len(batch_pairs):,} pairs from {len(playlists):,} playlists")
                logger.info(f"📊 Progress: {current_processed:,}/{total_playlists:,} playlists ({current_processed/total_playlists*100:.1f}%)")
                
                # Insert pairs periodically to avoid memory buildup
                if len(all_pairs) >= 1000000:  # Insert every 1M pairs
                    logger.info(f"💾 Inserting {len(all_pairs):,} pairs to database...")
                    self.populate_co_occurrence_table(all_pairs)
                    all_pairs = []  # Clear memory
            
            # Insert any remaining pairs
            if all_pairs:
                logger.info(f"💾 Inserting final {len(all_pairs):,} pairs...")
                self.populate_co_occurrence_table(all_pairs)
            
            # Step 5: Verify the results
            self.verify_co_occurrence_data()
            
            # Clear checkpoint file when complete
            if Path(self.checkpoint_file).exists():
                Path(self.checkpoint_file).unlink()
                logger.info("🧹 Checkpoint file cleared - processing complete")
            
            total_elapsed = time.time() - start_time
            logger.info("🎉 Co-occurrence pair generation completed successfully!")
            logger.info(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ Total time: {total_elapsed/60:.1f} minutes")
            logger.info(f"📊 Final stats: {current_processed:,} playlists processed")
            
        except Exception as e:
            logger.error(f"❌ Co-occurrence generation failed: {e}")
            logger.info("💡 You can resume from the last checkpoint by running the script again")
            raise
        finally:
            if self.db_conn:
                self.db_conn.close()

def main():
    """Main function to run co-occurrence generation"""
    builder = CoOccurrenceBuilder()
    builder.build_co_occurrence_data()

if __name__ == "__main__":
    main()
