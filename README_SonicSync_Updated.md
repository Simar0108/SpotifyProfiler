# 🎧 Sonic Sync – Custom ML-Powered Spotify Mood Profiler

**Sonic Sync** is a full-stack machine learning project that transforms Spotify listening behavior into mood-based personality insights using a custom-trained neural network, automated data pipelines, and GPT-powered narrative generation. This version pivots from relying on Spotify’s deprecated audio features to instead training a mood classifier from public datasets, creating a more robust and scalable system.

---

## 🧠 Project Objectives

- Train a **custom neural network** to classify music mood using public datasets (e.g., DEAM, Million Song Dataset)
- Apply the classifier to **Spotify listening history** collected every 12 hours
- Segment listening behavior by **time of day** and **day of week**
- Use **GPT** to generate descriptive summaries of user listening moods and musical personality
- Deploy the classifier as a **cloud-hosted inference API**
- Track model performance with **MLflow** or **Weights & Biases**
- Deliver outputs through a simple **frontend interface** or report

---

## 🔧 Key Features

- 📥 **Spotify Listening Logger**  
  Cron-based pipeline stores played tracks (every 12 hours) with timestamps in a structured database.

- 🧠 **Neural Network Mood Classifier**  
  Custom feedforward NN trained from scratch using open datasets to predict mood labels.

- 🎯 **Time Segmentation Logic**  
  Assigns tracks to segments like Morning, Afternoon, Night across each weekday.

- 📊 **MLOps + Experiment Tracking**  
  Training runs tracked with MLflow or WandB, with logs for metrics and hyperparameters.

- 🚀 **Cloud Deployment**  
  Model hosted via FastAPI + Docker and deployed on GCP, Render, or Railway.

- ✨ **LLM Integration (GPT)**  
  GPT generates narrative insights and personality profiles from mood predictions.

- 🖼️ **Optional Frontend**  
  Streamlit or custom frontend displays mood clusters and summaries.

---

## 📂 Project Structure

```
spotify-mood-profiler/
├── data/
│   ├── deam/                   # Public mood datasets
│   ├── embeddings/             # Genre/title embeddings
│   └── listening_logs.db       # Spotify listening history (timestamped)
│
├── model/
│   ├── train.py                # Training script for NN
│   ├── model.py                # Neural network architecture
│   ├── inference.py            # Model loading + prediction
│   └── mlflow_logs/            # Experiment tracking logs
│
├── api/
│   ├── app.py                  # FastAPI app for inference
│   ├── Dockerfile              # Containerization
│   └── requirements.txt
│
├── pipeline/
│   ├── fetch_recent.py         # Cron script to log recent tracks
│   ├── segment_tracks.py       # Time-of-day/day-of-week segmentation
│   └── runner.py               # End-to-end processor
│
├── gpt/
│   ├── generate_summary.py     # GPT calls for narrative mood output
│   └── prompts/
│
├── streamlit_app/ (optional)
│   └── app.py
│
├── README.md
└── .env.example
```

---

## 🛠️ Tech Stack & Tools

| Area | Tools |
|------|-------|
| Data Collection | Spotify API, Python, cron |
| Model Training | PyTorch / Keras, Scikit-learn |
| MLOps | MLflow or Weights & Biases |
| Inference API | FastAPI, Docker |
| Cloud | GCP, Railway, Render |
| LLM Integration | OpenAI GPT-4 / GPT-3.5 |
| Frontend (Optional) | Streamlit, React/Svelte |

---

## 🔄 Data Pipeline

1. **Cron script** scrapes `/recently-played` tracks every 12 hours
2. Stores data in a local or cloud-hosted DB with timestamps
3. When ready, runs segmentation logic to categorize tracks by time of day
4. Neural network model classifies mood of each track
5. GPT summarizes each mood/time cluster into a narrative
6. Outputs stored or visualized via API/frontend

---

## 🧪 ML Model Overview

- Inputs:
  - Track name
  - Artist
  - Genre
  - Tempo, key, mode (if available)
  - Embeddings (lyrics/title optional)

- Model:
  - Feedforward neural net (2–3 hidden layers)
  - Output: softmax mood classes or 2D valence/arousal regression
  - Training dataset: DEAM, MSD, Last.fm mood-labeled datasets

- Training Tools:
  - MLflow for tracking accuracy, loss, hyperparameters

---

## ✨ GPT Personality Summary Example

> "On weekday mornings, your music leans mellow and introspective — a space for clarity before chaos. By Friday night, your taste shifts toward emotional high-energy bangers, suggesting a cathartic release at the end of your week."

---

## 📅 Development Timeline (Suggested)

| Week | Focus |
|------|-------|
| 1 | Finalize data ingestion and tracking |
| 2 | Source + preprocess training data |
| 3 | Build + train neural network |
| 4 | Deploy model as inference API |
| 5 | Integrate GPT summarization |
| 6 | Build frontend / report interface |
| 7 | Polish, deploy, and document portfolio post |

---

## 🙌 Author

Simarpal Singh — Data scientist and ML engineer building behavioral insights from audio and language.

---

## 📄 License

MIT License
