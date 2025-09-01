# Sonic Sync — Shifted Project Plan (MPD Embeddings + User Personalization)

## Why the Shift?
- **Problem with ReccoBeats coverage:** Only ~10% of MPD tracks return audio features. Not enough to train a robust NN at scale.  
- **New plan:** Train the NN **only on MPD playlist structure** (co-occurrence) → learn **track embeddings**.  
- **ReccoBeats role changes:** Only used at *serving time* (for a user’s recent 50 tracks) to enrich group summaries with feature averages.  
- **Outcome:** Still a high-level project with large-scale self-supervised learning, embeddings, clustering, and LLM personalization.

---

## Updated Architecture

### 1. Training Data (MPD)
- 150k playlists, 15.5M track occurrences.
- Each playlist is a sequence of `spotify_id`s.
- Positive signal: **tracks that co-occur in playlists** (within ±3 positions).
- Negative signal: **tracks from other playlists**.

### 2. Neural Network (Embedding Model)
- **Input:** one-hot (track ID) → embedding layer (size 64).  
- **Model:** Skip-gram with Negative Sampling (item2vec).  
- **Training objective:**  
  - Pull embeddings of co-occurring tracks closer.  
  - Push embeddings of unrelated tracks apart.  
- **Output:** learned 64-d embedding for each track ID.

### 3. Clustering
- After training, cluster embeddings (KMeans, K=4 or K=6).  
- Define “quadrants” / “archetypes” post-hoc:  
  - e.g., “bright/kinetic”, “warm/laid-back”, “dark/moody”, “airy/spacious”.

### 4. Runtime Personalization (User Pipeline)
1. Collect user’s **recent 50 Spotify track IDs**.
2. For each:
   - Look up its NN embedding (if in MPD vocab).  
   - If missing → fallback (artist average or RB feature projection).  
3. Aggregate:
   - **Embedding mean** (`z_group`)  
   - **Cluster histogram** (fraction in each quadrant)  
   - **Top artists/genres** from cron job metadata  
   - **ReccoBeats features** for those 50 tracks → compute feature averages (energy, valence, tempo, etc.).
4. Build summary JSON:
   ```json
   {
     "n_tracks": 50,
     "embedding_mean": [...],
     "cluster_histogram": {"C1":0.65,"C2":0.20,"C3":0.10,"C4":0.05},
     "feature_means": {"energy":0.61,"valence":0.55,"tempo_bpm":124.2},
     "top_artists": ["Radiohead","Phoenix","Bon Iver"],
     "top_genres": ["indie rock","alt dance","folk"]
   }
   ```

### 5. LLM Narrative Generation
- Prompt template:
  > “You are generating a short narrative (120–160 words) about a user’s music listening. Use the JSON input. Emphasize the quadrant distribution, feature means, and top artists/genres. Keep the tone evocative, not technical.”

---

## Why This is Still High-Level
- **Scale:** Training on 15M track occurrences is industrial-size recommender learning.  
- **Technique:** Self-supervised item2vec (skip-gram) → core method behind Spotify/YouTube embeddings.  
- **System:** Multi-stage pipeline — embeddings + clustering + ReccoBeats enrichment + LLM.  
- **Resume-ready framing:**  
  - *“Trained self-supervised embedding model on 15M+ track occurrences from Spotify’s MPD to learn vibe-aware music representations. Clustered embeddings into high-level quadrants and integrated real-time user personalization by combining group embeddings with audio features and LLM narrative generation.”*

---

## Deliverables
1. **ETL:** SQLite tables for playlists/tracks, plus `track_vocab` for embedding indices.  
2. **Training:** item2vec (skip-gram) model → embedding matrix.  
3. **Clustering:** centroids + assignment per track.  
4. **Runtime:** aggregator that:
   - looks up embeddings for recent 50,  
   - computes cluster histogram + embedding mean,  
   - calls ReccoBeats for feature averages,  
   - outputs summary JSON.  
5. **Narrative:** LLM prompt template + generator.  
6. **Demo app:** show embedding quadrant distribution + narrative for a session.

---
