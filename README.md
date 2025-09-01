# Spotify Profiler

A machine learning pipeline for analyzing Spotify playlist data and generating music recommendations using collaborative filtering and embedding-based approaches.

## Overview

This project processes the Million Playlist Dataset (MPD) to build music recommendation systems using various machine learning techniques including:

- Collaborative filtering with co-occurrence matrices
- Item2Vec embeddings for track representations
- Hyperparameter tuning and experimentation
- Evaluation metrics for recommendation quality

## Project Structure

```
spotifyprofiler/
├── data/MPD/                 # Million Playlist Dataset
├── pipeline/                 # Core processing scripts
│   ├── mpd_processor.py      # MPD data processing
│   ├── build_co_occurrence.py # Co-occurrence matrix builder
│   ├── build_track_vocab.py  # Track vocabulary builder
│   ├── item2vec_trainer.py   # Item2Vec model trainer
│   └── reccobeats_client.py  # Recommendation client
├── tuning/                   # Hyperparameter tuning results
│   ├── checkpoints/          # Model checkpoints
│   ├── embeddings/           # Trained embeddings
│   └── experiment_results/   # Experiment results
├── requirements.txt          # Python dependencies
└── run_*.py                 # Execution scripts
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd spotifyprofiler
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env with your configuration
   ```

## Usage

### Data Processing

1. **Process MPD data**
   ```bash
   python pipeline/mpd_processor.py
   ```

2. **Build co-occurrence matrix**
   ```bash
   python pipeline/build_co_occurrence.py
   ```

3. **Build track vocabulary**
   ```bash
   python pipeline/build_track_vocab.py
   ```

### Training Models

1. **Train Item2Vec embeddings**
   ```bash
   python pipeline/item2vec_trainer.py
   ```

2. **Run hyperparameter tuning**
   ```bash
   python run_second_round.py
   ```

### Making Recommendations

```python
from pipeline.reccobeats_client import RecCoBeatsClient

client = RecCoBeatsClient()
recommendations = client.get_recommendations(playlist_tracks)
```

## Configuration

The project uses JSON configuration files for different experiments:

- `best_second_round_config_working.json` - Best performing configuration
- `second_round_checkpoint_working.json` - Training checkpoint
- Various experiment configs in `tuning/experiment_results/`

## Results

The project includes extensive hyperparameter tuning results stored in `tuning/experiment_results/` with metrics including:

- Precision@K
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Million Playlist Dataset (MPD) for providing the training data
- Item2Vec paper for the embedding approach
- Various open-source libraries used in this project
