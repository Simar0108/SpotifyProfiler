# 🎧 Sonic Sync – Neural Network Backend Plan

This document outlines the design and implementation of a custom NumPy-based neural network that classifies music mood based on DEAM audio features. It also covers how the model will later be used to classify mood segments from real-world Spotify listening data and generate personality-driven summaries using LLMs.

---

## 🎯 Project Goal

Train a custom neural network from scratch (using NumPy) to classify the mood of songs based on DEAM dataset features. This model will later be used to infer the mood of **time-segmented listening behavior** (e.g., “Tuesday Morning”) from the user’s personal Spotify history, and feed that into a GPT model for narrative-driven personality summaries.

---

## 🔁 Two-Phase Plan

---

### 🧱 Phase 1: Build and Train the Neural Network (NumPy Only)

#### ✅ Step 1: Prepare the Dataset
- Load `final_preprocessed_dataset.csv`
- Encode `mood` (categorical) into integers
- Standardize 50 feature columns
- Split into train/test sets (e.g., 80/20)

#### ✅ Step 2: Define Neural Network Architecture
- Input: 50 features  
- Hidden Layer 1: 128 neurons, ReLU  
- Hidden Layer 2: 64 neurons, ReLU  
- Output: 4 neurons, softmax activation (for mood classes)

#### ✅ Step 3: Implement from Scratch
- Forward pass for each layer
- Backpropagation using chain rule and gradient descent
- Use:
  - Softmax + cross-entropy loss
  - ReLU and its derivative
  - Learning rate: start with `0.01`
- Train across `n_epochs` and record:
  - Training loss
  - Accuracy per epoch

#### ✅ Step 4: Evaluate Model
- Final accuracy on test set
- Optional: confusion matrix, plots of loss/accuracy over time

---

### 🧠 Phase 2: Apply Neural Network to User Listening Segments

#### ✅ Step 1: Extract a Segment (e.g., Tuesday Morning)
- Pull tracks from your Spotify logging database based on timestamp filters

#### ❗ Challenge: No Audio Features Available

Your listening database only contains:
- `track_name`
- `artist_name`
- `genre` (if derived)
- No valence/energy/audio metrics (Spotify API deprecated them)

---

### 🔧 Solutions to Handle Missing Audio Features

| Option | Strategy | Pros | Cons |
|--------|----------|------|------|
| **1. Metadata Embedding** | Use word embeddings (`track_name`, `artist`, `genre`) as feature vectors (e.g., word2vec, SBERT) | Simple to implement, keeps model logic | May lose fidelity compared to true audio features |
| **2. Train Metadata-Based Classifier** | Train a second model (shallow NN or logistic regression) on just metadata → mood | Aligned with real-world input | Less expressive than full NN |
| **3. Metadata → Feature Approximation** | Train a regression model to predict DEAM-style features from track/artist/genre | Allows reuse of main NN for inference | Requires extra model and training step |

✅ Plan: Implement **option 1 or 2 after core model is trained**

---

### ✅ Step 2: Run Inference on a Segment
- Predict mood label for each track
- Aggregate moods into:
  - Total count
  - Most frequent
  - Example songs per class

#### ✅ Step 3: Format for LLM
Prepare GPT input prompt like:

```text
During Tuesday Morning, the listener’s mood breakdown is:
- 60% Chill
- 25% Energetic
- 15% Melancholic

Example songs: “Motion Sickness – Phoebe Bridgers”, “Midnight City – M83”

Write a poetic summary of the user’s personality and emotional tendencies during this time segment.
```

---

## ✅ Deliverables

| File | Purpose |
|------|---------|
| `neural_network.py` | Custom NumPy-based classifier |
| `train.py` | Training script for NN |
| `predict.py` | Run inference on segment |
| `segment_analysis.py` | Apply to Spotify history |
| `gpt_summarizer.py` | Format and send GPT prompt |
| `README.md` | Project instructions |
