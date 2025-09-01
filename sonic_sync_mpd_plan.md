# Sonic Sync — MPD + ReccoBeats Neural Network Plan

## Overview
The goal of this project is to create a **custom neural network (NN)** that learns meaningful music “vibe” representations from **all audio features**. These embeddings are then aggregated over groups of tracks (e.g., a user’s recent 50 plays) and summarized by an **LLM narrative generator**.

This approach uses the **Million Playlist Dataset (MPD)** for large-scale training data and the **ReccoBeats API** for extracting Spotify-style audio features.

---

## Why Build an NN?
Averages of features like energy and valence are simple, but they don’t show off ML depth.  
The NN solves three problems:
1. **Uses all features** — captures nonlinear relationships (e.g., “danceable but low valence with high tempo”).
2. **Learns from playlist context** — tracks co-occurring in playlists should be embedded nearby.
3. **Produces embeddings** — clusterable into “quadrants” or archetypes that generalize across playlists and apply to user data.

---

## Data Sources

### 1. Million Playlist Dataset (MPD)
- ~1M playlists sampled from 2010–2017 US Spotify data.
- Each playlist contains:
  - Playlist title + metadata (num_tracks, duration, edits, etc.).
  - List of tracks with `track_uri`, `track_name`, `artist_name`, `album_name`, `duration_ms`.
- Example:
  ```json
  {
    "name": "musical",
    "pid": 5,
    "num_tracks": 12,
    "tracks": [
      {
        "artist_name": "Degiheugi",
        "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
        "track_name": "Finalement",
        "album_name": "Dancing Chords and Fireflies",
        "duration_ms": 166264
      }
    ]
  }
  ```

### 2. ReccoBeats API
- Provides Spotify-style audio features for track IDs:
  - `energy, valence, danceability, tempo, loudness, speechiness, instrumentalness, liveness, key, mode, acousticness`.
- Supports batch fetch (≤40 IDs).
- These features form the **input vectors** for the NN.

---

## Pipeline Architecture

### Step 1: Extract IDs
- Parse MPD → extract **Spotify track IDs** from `track_uri`.
- Deduplicate IDs.

### Step 2: Fetch Features
- Call ReccoBeats in batches.
- Cache responses in a local DB table:
  - `spotify_id`, `energy`, `valence`, `danceability`, `tempo`, `loudness`, `speechiness`, `instrumentalness`, `liveness`, `key`, `mode`, `acousticness`.

### Step 3: Build Training Pairs
- **Positive pairs**: tracks from the same playlist (within ±2 positions).
- **Negative pairs**: tracks from different playlists.
- This forms the self-supervised signal for training.

### Step 4: Train the NN
- **Input**: full feature vector per track (standardized).
  - Example: `[energy, valence, danceability, tempo, loudness, speechiness, instrumentalness, liveness, acousticness, mode, key_sin, key_cos]`.
- **Model**: small MLP encoder.
  - `in → 128 (ReLU) → 64 (ReLU) → z(32)`
  - Optional projection head for contrastive loss.
- **Loss**: InfoNCE (contrastive).  
  - Pull positive pairs together, push negatives apart.
- **Output**: 32-dim “vibe embedding” per track.

### Step 5: Cluster Embeddings
- Run **KMeans** (K=4) on embeddings → define “quadrants” (data-driven).
- Post-hoc name clusters based on centroids:
  - C1: “bright / kinetic”
  - C2: “warm / laid-back”
  - C3: “dark / moody”
  - C4: “airy / spacious”

### Step 6: Runtime (User Pipeline)
1. Collect user’s **recent 50 track IDs**.
2. Fetch audio features via ReccoBeats (cache if possible).
3. Embed each track with trained NN → `z_i`.
4. Aggregate:
   - `z_group = mean(z_i)`
   - `cluster_histogram = fraction of tracks in each cluster`
   - `feature_means = average raw features`
   - `top_artists`, `top_genres` (from metadata)
5. Build summary JSON:
   ```json
   {
     "n_tracks": 50,
     "feature_means": {
       "energy": 0.61,
       "valence": 0.55,
       "danceability": 0.58,
       "tempo_bpm": 124.2,
       "loudness_db": -7.1
     },
     "quadrant": "bright / kinetic",
     "cluster_histogram": {"C1":0.66,"C2":0.18,"C3":0.10,"C4":0.06},
     "top_artists": ["Radiohead","Phoenix","Two Door Cinema Club"],
     "top_genres": ["indie rock","alt dance","new wave"]
   }
   ```

### Step 7: LLM Narrative
- Prompt template:
  > “You are generating a short narrative (120–160 words) about a user’s music listening. Use the JSON input. Emphasize the quadrant label and cluster mix. Mention a few top artists/genres. Keep it evocative, not technical.”

---

## Why This Works
- **Scalable training**: MPD playlists provide millions of track-pair signals.
- **All features**: NN learns nonlinear combinations beyond energy/valence.
- **Data-driven quadrants**: no need to hand-label; clusters emerge from embeddings.
- **Personalization**: Same ReccoBeats feature space applies to user’s cron-job tracks.
- **Narrative output**: Group embeddings + feature stats → LLM storytelling.

---

## Deliverables
1. **Feature cache**: MPD track IDs with ReccoBeats features.
2. **NN training script** (`train_contrastive.py`).
3. **Embedding + clustering script** (`embed_and_cluster.py`).
4. **Group aggregator** (`aggregate_group.py`).
5. **Narrative generator** (`narrative.py` with prompt template).
6. **Demo app**: Run pipeline on “recent 50” tracks and output narrative.
