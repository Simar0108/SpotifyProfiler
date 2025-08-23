# 🚀 Sonic Sync Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Copy `env_example.txt` to `.env` and fill in your credentials:
```bash
cp env_example.txt .env
```

Edit `.env` with your actual Spotify API credentials:
- `SPOTIFY_CLIENT_ID`: Your Spotify App Client ID
- `SPOTIFY_CLIENT_SECRET`: Your Spotify App Client Secret
- `SPOTIFY_REDIRECT_URI`: http://localhost:8000/callback (or your preferred URI)

### 3. Get Spotify API Credentials
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add `http://localhost:8000/callback` to Redirect URIs
4. Copy Client ID and Client Secret to your `.env` file

### 4. Run Data Collection

**Option A: One-time collection**
```bash
python scripts/collect_data.py
```

**Option B: Continuous scheduler (recommended)**
```bash
python scripts/scheduler.py
```

The scheduler will:
- Run initial data collection immediately
- Continue collecting data every 24 hours
- Log all activity to `sonic_sync.log`

## Database Structure

The system creates these tables:
- `users`: Spotify user information and tokens
- `tracks`: Track metadata (name, artist, album, etc.)
- `listening_history`: When you listened to each track
- `audio_features`: Spotify's audio analysis features

## Data Collection Details

- **Recent Tracks**: Fetches your last 50 played tracks
- **Audio Features**: Automatically gets mood/energy data for each track
- **Time Segmentation**: Categorizes listening by morning/afternoon/night
- **Deduplication**: Won't store duplicate listening records

## Troubleshooting

**OAuth Issues**: 
- Make sure your redirect URI matches exactly
- Check that your Spotify app is properly configured

**Database Issues**:
- SQLite database will be created automatically
- Check file permissions in your project directory

**Rate Limiting**:
- Spotify has API limits, but this shouldn't be an issue for personal use
- The system includes error handling for rate limits

## Next Steps

Once you have data collecting, we can build:
1. Time-of-day mood analysis
2. Clustering algorithms
3. GPT personality summaries
4. Visualization dashboard 