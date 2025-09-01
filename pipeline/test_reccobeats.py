"""
Test script for ReccoBeats client and feature extraction.

Run this to verify your API key and basic functionality before
processing the full MPD dataset.
"""

import os
import sys
from pathlib import Path

# Add pipeline directory to path
sys.path.append(str(Path(__file__).parent))

from reccobeats_client import ReccoBeatsClient
from feature_extractor import MPDFeatureExtractor


def test_client():
    """Test basic ReccoBeats client functionality."""
    print("🧪 Testing ReccoBeats Client...")
    
    # Create client (no API key needed for open source API)
    client = ReccoBeatsClient()
    
    # Test connection
    print("   Testing API connection...")
    if client.test_connection():
        print("   ✅ API connection successful!")
    else:
        print("   ❌ API connection failed!")
        return False
    
    # Test with a few sample track IDs
    test_tracks = [
        "4iV5W9uYEdYUVa79Axb7Rh",  # Example Spotify track ID
        "1301WleyT98MSxVHPZCA6M"   # Another example
    ]
    
    print(f"   Testing feature extraction for {len(test_tracks)} tracks...")
    try:
        features = client.get_audio_features(test_tracks)
        print(f"   ✅ Successfully fetched features for {len(features)} tracks")
        
        # Show all features for each track
        if features:
            print(f"   📊 Features for {len(features)} tracks:")
            for i, feature in enumerate(features):
                print(f"      Track {i+1} ({feature.spotify_id}):")
                print(f"         Energy: {feature.energy:.3f}")
                print(f"         Valence: {feature.valence:.3f}")
                print(f"         Danceability: {feature.danceability:.3f}")
                print(f"         Tempo: {feature.tempo:.1f} BPM")
                print(f"         Loudness: {feature.loudness:.1f} dB")
                print(f"         Speechiness: {feature.speechiness:.3f}")
                print(f"         Instrumentalness: {feature.instrumentalness:.3f}")
                print(f"         Liveness: {feature.liveness:.3f}")
                print(f"         Key: {feature.key}")
                print(f"         Mode: {feature.mode}")
                print(f"         Acousticness: {feature.acousticness:.3f}")
                if i < len(features) - 1:  # Don't print separator after last track
                    print("         " + "-" * 30)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return False


def test_database_connection():
    """Test database connection and basic queries."""
    print("\n🗄️  Testing Database Connection...")
    
    db_path = "data/MPD/mpd_database.db"
    
    if not os.path.exists(db_path):
        print(f"   ❌ Database not found at {db_path}")
        return False
    
    try:
        from feature_extractor import MPDFeatureExtractor
        
        # Create extractor (no API key needed)
        extractor = MPDFeatureExtractor(db_path)
        
        # Test unique track ID extraction
        print("   Testing track ID extraction...")
        unique_ids = extractor.get_unique_track_ids()
        print(f"   ✅ Found {len(unique_ids)} unique track IDs")
        
        # Test existing features check
        print("   Testing existing features check...")
        existing_features = extractor.get_existing_features()
        print(f"   ✅ Found {len(existing_features)} existing features")
        
        # Calculate tracks needing features
        tracks_needing_features = [tid for tid in unique_ids if tid not in existing_features]
        print(f"   📊 Tracks needing features: {len(tracks_needing_features)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 ReccoBeats Integration Test Suite\n")
    
    # Test 1: Client functionality
    client_ok = test_client()
    
    # Test 2: Database connection
    db_ok = test_database_connection()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    if client_ok:
        print("✅ ReccoBeats Client: PASSED")
    else:
        print("❌ ReccoBeats Client: FAILED")
    
    if db_ok:
        print("✅ Database Connection: PASSED")
    else:
        print("❌ Database Connection: FAILED")
    
    if client_ok and db_ok:
        print("\n🎉 All tests passed! Ready to run feature extraction.")
        print("\nNext steps:")
        print("1. Run: python pipeline/feature_extractor.py")
        print("2. Monitor progress in feature_extraction.log")
        print("3. Consider using parallel processing for faster extraction")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
    
    print("="*50)


if __name__ == "__main__":
    main()
