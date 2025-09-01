#!/usr/bin/env python3
"""
Build Track Vocabulary - Extract unique tracks from MPD data and populate track_vocab table

This script:
1. Extracts unique track IDs from mpd_tracks table
2. Counts frequency of each track across playlists
3. Assigns sequential embedding indices (0, 1, 2, ...)
4. Populates the track_vocab table

Optimized for large datasets with:
- Bulk operations and batch processing
- Progress checkpoints and detailed logging
- Memory-efficient processing
- Performance monitoring
"""

import sqlite3
import logging
import time
from pathlib import Path
from datetime import datetime

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TrackVocabBuilder:
    """Builds track vocabulary from existing MPD data with optimizations"""
    
    def __init__(self, db_path: str = "data/MPD/mpd_database.db"):
        self.db_path = db_path
        self.db_conn = None
        self.batch_size = 10000  # Process in batches for memory efficiency
        
    def connect_database(self):
        """Connect to database with aggressive optimizations for bulk operations"""
        try:
            self.db_conn = sqlite3.connect(self.db_path, timeout=60.0)
            
            # Aggressive optimizations for bulk operations
            self.db_conn.execute("PRAGMA journal_mode = WAL")
            self.db_conn.execute("PRAGMA synchronous = OFF")  # Faster for bulk operations
            self.db_conn.execute("PRAGMA cache_size = 50000")  # Larger cache
            self.db_conn.execute("PRAGMA temp_store = MEMORY")
            self.db_conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping
            self.db_conn.execute("PRAGMA page_size = 65536")  # Larger page size
            
            logger.info("✅ Connected to database with bulk operation optimizations")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def get_dataset_stats(self):
        """Get overview of the dataset before processing"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("📊 Analyzing dataset size...")
            
            # Get total track occurrences
            cursor.execute("SELECT COUNT(*) FROM mpd_tracks WHERE track_uri LIKE 'spotify:track:%'")
            total_occurrences = cursor.fetchone()[0]
            
            # Get total playlists
            cursor.execute("SELECT COUNT(*) FROM mpd_playlists")
            total_playlists = cursor.fetchone()[0]
            
            # Estimate unique tracks (rough estimate)
            cursor.execute("""
                SELECT COUNT(DISTINCT track_uri) 
                FROM mpd_tracks 
                WHERE track_uri LIKE 'spotify:track:%'
                LIMIT 1000000
            """)
            sample_unique = cursor.fetchone()[0]
            
            logger.info(f"📊 Dataset Overview:")
            logger.info(f"   Total track occurrences: {total_occurrences:,}")
            logger.info(f"   Total playlists: {total_playlists:,}")
            logger.info(f"   Estimated unique tracks: {sample_unique:,}+ (sample)")
            
            return total_occurrences, total_playlists, sample_unique
            
        except Exception as e:
            logger.error(f"❌ Failed to get dataset stats: {e}")
            raise
    
    def get_unique_tracks_optimized(self):
        """Extract unique tracks with frequency counts using optimized queries"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("🔍 Extracting unique tracks with frequency counts...")
            start_time = time.time()
            
            # Use optimized query with proper indexing
            cursor.execute("""
                SELECT 
                    track_uri,
                    COUNT(*) as frequency
                FROM mpd_tracks 
                WHERE track_uri LIKE 'spotify:track:%'
                GROUP BY track_uri
                ORDER BY frequency DESC
            """)
            
            tracks = cursor.fetchall()
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Extracted {len(tracks):,} unique tracks in {elapsed:.1f}s")
            logger.info(f"   Processing rate: {len(tracks)/elapsed:.0f} tracks/sec")
            
            # Convert to list of (track_id, frequency) tuples
            track_data = []
            for track_uri, frequency in tracks:
                track_id = track_uri.replace('spotify:track:', '')
                track_data.append((track_id, frequency))
            
            return track_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract unique tracks: {e}")
            raise
    
    def populate_track_vocab_optimized(self, track_data):
        """Populate track_vocab table using optimized bulk operations"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("📝 Populating track_vocab table with optimized bulk operations...")
            
            # Clear existing data (in case of re-run)
            cursor.execute("DELETE FROM track_vocab")
            logger.info("🧹 Cleared existing track_vocab data")
            
            start_time = time.time()
            total_tracks = len(track_data)
            
            # Process in batches for memory efficiency
            for batch_start in range(0, total_tracks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_tracks)
                batch = track_data[batch_start:batch_end]
                
                # Prepare batch data
                batch_data = []
                for i, (track_id, frequency) in enumerate(batch):
                    embedding_index = batch_start + i
                    batch_data.append((track_id, embedding_index, frequency))
                
                # Bulk insert batch
                cursor.executemany("""
                    INSERT INTO track_vocab (track_id, embedding_index, frequency)
                    VALUES (?, ?, ?)
                """, batch_data)
                
                # Progress update every batch
                elapsed = time.time() - start_time
                processed = batch_end
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_tracks - processed) / rate if rate > 0 else 0
                
                logger.info(f"📝 Batch {batch_start//self.batch_size + 1}: "
                          f"Processed {processed:,}/{total_tracks:,} tracks "
                          f"({processed/total_tracks*100:.1f}%) "
                          f"Rate: {rate:.0f} tracks/sec "
                          f"ETA: {eta/60:.1f} minutes")
                
                # Commit every 10 batches to avoid memory buildup
                if (batch_start // self.batch_size + 1) % 10 == 0:
                    self.db_conn.commit()
                    logger.info(f"💾 Committed batch {batch_start//self.batch_size + 1}")
            
            # Final commit
            self.db_conn.commit()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Populated track_vocab with {total_tracks:,} tracks in {elapsed:.1f}s")
            logger.info(f"   Overall rate: {total_tracks/elapsed:.0f} tracks/sec")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate track_vocab: {e}")
            raise
    
    def verify_population_detailed(self):
        """Comprehensive verification of the track_vocab table"""
        try:
            cursor = self.db_conn.cursor()
            
            logger.info("🔍 Verifying track_vocab population...")
            
            # Get comprehensive stats
            cursor.execute("SELECT COUNT(*) FROM track_vocab")
            total_tracks = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(embedding_index), MAX(embedding_index) FROM track_vocab")
            min_idx, max_idx = cursor.fetchone()
            
            cursor.execute("SELECT MIN(frequency), MAX(frequency), AVG(frequency) FROM track_vocab")
            min_freq, max_freq, avg_freq = cursor.fetchone()
            
            # Check for gaps in indices
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT embedding_index FROM track_vocab 
                    WHERE embedding_index NOT IN (
                        SELECT embedding_index FROM track_vocab 
                        ORDER BY embedding_index 
                        LIMIT (SELECT MAX(embedding_index) + 1 FROM track_vocab)
                    )
                )
            """)
            gaps = cursor.fetchone()[0]
            
            # Frequency distribution analysis
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN frequency = 1 THEN '1'
                        WHEN frequency BETWEEN 2 AND 5 THEN '2-5'
                        WHEN frequency BETWEEN 6 AND 20 THEN '6-20'
                        WHEN frequency BETWEEN 21 AND 100 THEN '21-100'
                        ELSE '100+'
                    END as freq_range,
                    COUNT(*) as count
                FROM track_vocab 
                GROUP BY freq_range
                ORDER BY MIN(frequency)
            """)
            freq_distribution = cursor.fetchall()
            
            logger.info("📊 Track Vocabulary Statistics:")
            logger.info(f"   Total tracks: {total_tracks:,}")
            logger.info(f"   Index range: {min_idx} to {max_idx}")
            logger.info(f"   Frequency range: {min_freq} to {max_freq}")
            logger.info(f"   Average frequency: {avg_freq:.1f}")
            
            if gaps == 0:
                logger.info("✅ Embedding indices are sequential (no gaps)")
            else:
                logger.warning(f"⚠️ Found {gaps} gaps in embedding indices")
            
            logger.info("📊 Frequency Distribution:")
            for freq_range, count in freq_distribution:
                percentage = (count / total_tracks) * 100
                logger.info(f"   {freq_range:>6}: {count:>8,} tracks ({percentage:>5.1f}%)")
            
            # Show top tracks
            cursor.execute("SELECT track_id, embedding_index, frequency FROM track_vocab ORDER BY frequency DESC LIMIT 10")
            top_tracks = cursor.fetchall()
            
            logger.info("🏆 Top 10 most frequent tracks:")
            for i, (track_id, idx, freq) in enumerate(top_tracks, 1):
                logger.info(f"   {i:>2}. {track_id[:8]}... (index: {idx:>6}, frequency: {freq:>4})")
            
        except Exception as e:
            logger.error(f"❌ Failed to verify population: {e}")
            raise
    
    def build_vocabulary(self):
        """Main method to build the complete track vocabulary with progress tracking"""
        try:
            start_time = time.time()
            logger.info("🚀 Starting optimized track vocabulary build...")
            logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Step 1: Connect to database
            self.connect_database()
            
            # Step 2: Analyze dataset size
            total_occurrences, total_playlists, estimated_unique = self.get_dataset_stats()
            
            # Step 3: Extract unique tracks
            track_data = self.get_unique_tracks_optimized()
            
            # Step 4: Populate track_vocab table
            self.populate_track_vocab_optimized(track_data)
            
            # Step 5: Verify the results
            self.verify_population_detailed()
            
            total_elapsed = time.time() - start_time
            logger.info("🎉 Track vocabulary build completed successfully!")
            logger.info(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ Total time: {total_elapsed/60:.1f} minutes")
            logger.info(f"📊 Final stats: {len(track_data):,} unique tracks processed")
            
        except Exception as e:
            logger.error(f"❌ Track vocabulary build failed: {e}")
            raise
        finally:
            if self.db_conn:
                self.db_conn.close()

def main():
    """Main function to run the vocabulary build"""
    builder = TrackVocabBuilder()
    builder.build_vocabulary()

if __name__ == "__main__":
    main()
