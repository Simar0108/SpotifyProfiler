#!/usr/bin/env python3
"""
Second Round Hyperparameter Experiments - Comprehensive Optimization

This script runs a strategic second round of experiments based on first round insights:
- Phase 1: Fine-tune winners (Learning Rate, Negative Samples, Hidden Dims)
- Phase 2: Architecture refinement (Context Window, Weight Decay, Batch Size)
- Phase 3: Advanced techniques (Schedulers, Optimizers)

Features:
- Progress tracking with detailed logging
- Checkpoint/resume capability
- Optimized execution order
- Comprehensive result analysis
- Automatic best configuration selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import logging
import json
import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
import random
import itertools
from tqdm import tqdm
import warnings
import pickle
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Set up logging with timestamps and progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CoOccurrenceDataset(Dataset):
    """Enhanced dataset for co-occurrence pairs with negative sampling"""
    
    def __init__(self, db_path, max_samples=None, negative_samples=5, random_seed=42):
        self.db_path = db_path
        self.negative_samples = negative_samples
        self.pairs = []
        self.track_vocab = {}
        self.vocab_size = 0
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self._load_data(max_samples)
    
    def _load_data(self, max_samples):
        """Load co-occurrence pairs and build vocabulary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load track vocabulary
                cursor.execute("SELECT track_id, embedding_index FROM track_vocab ORDER BY embedding_index")
                for track_id, idx in cursor.fetchall():
                    self.track_vocab[track_id] = idx
                
                self.vocab_size = len(self.track_vocab)
                logger.info(f"📚 Vocabulary size: {self.vocab_size:,}")
                
                # Load co-occurrence pairs
                query = "SELECT track1_id, track2_id FROM co_occurrence_pairs"
                if max_samples:
                    query += f" LIMIT {max_samples}"
                
                cursor.execute(query)
                pairs = cursor.fetchall()
                
                # Convert to indices
                for track1_id, track2_id in pairs:
                    if track1_id in self.track_vocab and track2_id in self.track_vocab:
                        idx1 = self.track_vocab[track1_id]
                        idx2 = self.track_vocab[track2_id]
                        self.pairs.append((idx1, idx2))
                
                logger.info(f"🔗 Loaded {len(self.pairs):,} co-occurrence pairs")
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get positive pair and negative samples"""
        pos_track, context_track = self.pairs[idx]
        
        # Generate negative samples
        neg_tracks = []
        for _ in range(self.negative_samples):
            neg_track = random.randint(0, self.vocab_size - 1)
            while neg_track == pos_track or neg_track == context_track:
                neg_track = random.randint(0, self.vocab_size - 1)
            neg_tracks.append(neg_track)
        
        return {
            'pos_track': torch.tensor(pos_track, dtype=torch.long),
            'context_track': torch.tensor(context_track, dtype=torch.long),
            'neg_tracks': torch.tensor(neg_tracks, dtype=torch.long)
        }

class Item2VecModel(nn.Module):
    """Enhanced Item2Vec model with configurable architecture"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.1):
        super(Item2VecModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.track_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layers
        self.hidden_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
    
    def forward(self, pos_track, context_track, neg_tracks):
        """Forward pass with positive and negative samples"""
        # Get embeddings
        pos_emb = self.track_embeddings(pos_track)
        context_emb = self.context_embeddings(context_track)
        neg_embs = self.track_embeddings(neg_tracks)
        
        # Concatenate positive and context embeddings
        pos_context = torch.cat([pos_emb, context_emb], dim=1)
        
        # Process through hidden layers
        hidden = self.hidden_layer(pos_context)
        
        # Positive score
        pos_score = self.output_layer(hidden)
        
        # Negative scores
        neg_scores = []
        for i in range(neg_tracks.size(1)):
            neg_context = torch.cat([pos_emb, neg_embs[:, i, :]], dim=1)
            neg_hidden = self.hidden_layer(neg_context)
            neg_score = self.output_layer(neg_hidden)
            neg_scores.append(neg_score)
        
        neg_scores = torch.cat(neg_scores, dim=1)
        
        return pos_score, neg_scores

class ExperimentManager:
    """Manages experiment execution with checkpointing and resume capability"""
    
    def __init__(self, checkpoint_file="second_round_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.completed_experiments = set()
        self.current_phase = 1
        self.current_experiment = 0
        self.start_time = time.time()
        self.total_experiments = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load progress from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                self.completed_experiments = set(checkpoint.get('completed_experiments', []))
                self.current_phase = checkpoint.get('current_phase', 1)
                self.current_experiment = checkpoint.get('current_experiment', 0)
                logger.info(f"📂 Loaded checkpoint: {len(self.completed_experiments)} experiments completed")
                logger.info(f"🔄 Resuming from Phase {self.current_phase}, Experiment {self.current_experiment}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        checkpoint = {
            'completed_experiments': list(self.completed_experiments),
            'current_phase': self.current_phase,
            'current_experiment': self.current_experiment,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': self.total_experiments
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def mark_completed(self, experiment_name):
        """Mark an experiment as completed"""
        self.completed_experiments.add(experiment_name)
        self._save_checkpoint()
        logger.info(f"✅ Completed: {experiment_name}")
    
    def is_completed(self, experiment_name):
        """Check if experiment is already completed"""
        return experiment_name in self.completed_experiments
    
    def get_progress(self):
        """Get current progress information"""
        completed = len(self.completed_experiments)
        total = self.total_experiments
        progress = (completed / total * 100) if total > 0 else 0
        elapsed = time.time() - self.start_time
        
        return {
            'completed': completed,
            'total': total,
            'progress': progress,
            'elapsed_time': elapsed,
            'current_phase': self.current_phase
        }

def create_learning_rate_experiments():
    """Phase 1: Learning rate fine-tuning experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'batch_size': 1024,
        'negative_samples': 20,  # Use best from first round
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    learning_rates = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
    
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        experiments.append({
            'name': f'lr_{lr:.3f}_64d',
            'config': config,
            'phase': 1,
            'priority': 'high'
        })
    
    return experiments

def create_negative_sampling_experiments():
    """Phase 1: Negative sampling optimization experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    negative_samples = [15, 18, 22, 25, 30]
    
    for neg in negative_samples:
        config = base_config.copy()
        config['negative_samples'] = neg
        experiments.append({
            'name': f'neg_{neg}_64d',
            'config': config,
            'phase': 1,
            'priority': 'high'
        })
    
    return experiments

def create_hidden_dimension_experiments():
    """Phase 1: Hidden dimension tuning experiments"""
    base_config = {
        'embedding_dim': 64,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'negative_samples': 20,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    hidden_dims = [64, 96, 128, 192, 256]
    
    for hidden_dim in hidden_dims:
        config = base_config.copy()
        config['hidden_dim'] = hidden_dim
        experiments.append({
            'name': f'hidden_{hidden_dim}_64d',
            'config': config,
            'phase': 1,
            'priority': 'high'
        })
    
    return experiments

def create_context_window_experiments():
    """Phase 2: Context window optimization experiments"""
    # Note: This would require modifying the data loading to support different context windows
    # For now, we'll create placeholder experiments
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'negative_samples': 20,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    context_windows = [2, 3, 4, 5]
    
    for window in context_windows:
        config = base_config.copy()
        config['context_window'] = window
        experiments.append({
            'name': f'context_{window}_64d',
            'config': config,
            'phase': 2,
            'priority': 'medium'
        })
    
    return experiments

def create_weight_decay_experiments():
    """Phase 2: Weight decay fine-tuning experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'negative_samples': 20,
        'optimizer': 'adam',
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    weight_decays = [0.0, 0.0001, 0.0005, 0.001]
    
    for wd in weight_decays:
        config = base_config.copy()
        config['weight_decay'] = wd
        experiments.append({
            'name': f'wd_{wd:.4f}_64d',
            'config': config,
            'phase': 2,
            'priority': 'medium'
        })
    
    return experiments

def create_batch_size_experiments():
    """Phase 2: Batch size optimization experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'negative_samples': 20,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    batch_sizes = [512, 1024, 2048, 4096]
    
    for batch_size in batch_sizes:
        config = base_config.copy()
        config['batch_size'] = batch_size
        experiments.append({
            'name': f'batch_{batch_size}_64d',
            'config': config,
            'phase': 2,
            'priority': 'medium'
        })
    
    return experiments

def create_scheduler_experiments():
    """Phase 3: Learning rate scheduler comparison experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'negative_samples': 20,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    
    # Cosine scheduler
    config = base_config.copy()
    config['lr_scheduler'] = 'cosine'
    config['lr_t_max'] = 1000
    experiments.append({
        'name': 'scheduler_cosine_64d',
        'config': config,
        'phase': 3,
        'priority': 'low'
    })
    
    # Step scheduler
    config = base_config.copy()
    config['lr_scheduler'] = 'step'
    config['lr_step_size'] = 1000
    config['lr_gamma'] = 0.9
    experiments.append({
        'name': 'scheduler_step_64d',
        'config': config,
        'phase': 3,
        'priority': 'low'
    })
    
    # ReduceLROnPlateau
    config = base_config.copy()
    config['lr_scheduler'] = 'plateau'
    config['lr_patience'] = 3
    config['lr_factor'] = 0.5
    experiments.append({
        'name': 'scheduler_plateau_64d',
        'config': config,
        'phase': 3,
        'priority': 'low'
    })
    
    return experiments

def create_optimizer_experiments():
    """Phase 3: Optimizer variant experiments"""
    base_config = {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 0.01,
        'batch_size': 1024,
        'negative_samples': 20,
        'weight_decay': 0.0,
        'dropout_rate': 0.1,
        'lr_scheduler': None,
        'epochs': 15,
        'patience': 5,
        'checkpoint_freq': 5
    }
    
    experiments = []
    
    # AdamW
    config = base_config.copy()
    config['optimizer'] = 'adamw'
    experiments.append({
        'name': 'optimizer_adamw_64d',
        'config': config,
        'phase': 3,
        'priority': 'low'
    })
    
    # RAdam
    config = base_config.copy()
    config['optimizer'] = 'radam'
    experiments.append({
        'name': 'optimizer_radam_64d',
        'config': config,
        'phase': 3,
        'priority': 'low'
    })
    
    return experiments

def get_all_experiments():
    """Get all experiments organized by phase"""
    experiments = []
    
    # Phase 1: High Impact, Low Risk
    experiments.extend(create_learning_rate_experiments())
    experiments.extend(create_negative_sampling_experiments())
    experiments.extend(create_hidden_dimension_experiments())
    
    # Phase 2: Medium Impact, Medium Risk
    experiments.extend(create_context_window_experiments())
    experiments.extend(create_weight_decay_experiments())
    experiments.extend(create_batch_size_experiments())
    
    # Phase 3: High Risk, High Reward
    experiments.extend(create_scheduler_experiments())
    experiments.extend(create_optimizer_experiments())
    
    return experiments

def run_single_experiment(config, experiment_name, db_path, max_samples=100000, k_folds=3):
    """Run a single experiment with k-fold cross-validation"""
    logger.info(f"🧪 Running experiment: {experiment_name}")
    logger.info(f"⚙️ Config: {json.dumps(config, indent=2)}")
    
    # Initialize results storage
    fold_results = []
    total_training_time = 0
    
    # K-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold in range(1, k_folds + 1):
        logger.info(f"🔄 Fold {fold}/{k_folds}")
        
        # Create dataset
        dataset = CoOccurrenceDataset(db_path, max_samples, config['negative_samples'])
        
        # Split data
        train_indices, val_indices = list(kf.split(dataset))[fold-1]
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = Item2VecModel(
            vocab_size=dataset.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            dropout_rate=config['dropout_rate']
        )
        
        # Setup optimizer
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'radam':
            # Note: RAdam might not be available in all PyTorch versions
            try:
                from torch.optim import RAdam
                optimizer = RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            except ImportError:
                logger.warning("RAdam not available, falling back to Adam")
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        # Setup learning rate scheduler
        scheduler = None
        if config['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('lr_t_max', 1000))
        elif config['lr_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step_size', 1000), gamma=config.get('lr_gamma', 0.9))
        elif config['lr_scheduler'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config.get('lr_patience', 3), factor=config.get('lr_factor', 0.5)
            )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                pos_score, neg_scores = model(
                    batch['pos_track'], 
                    batch['context_track'], 
                    batch['neg_tracks']
                )
                
                # Binary cross-entropy loss
                pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
                neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                
                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pos_score, neg_scores = model(
                        batch['pos_track'], 
                        batch['context_track'], 
                        batch['neg_tracks']
                    )
                    
                    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
                    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                    
                    loss = pos_loss + neg_loss
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
                break
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{config['epochs']}: "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        fold_time = time.time() - start_time
        total_training_time += fold_time
        
        # Store fold results
        fold_results.append({
            'fold': fold,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': fold_time,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': best_val_loss
        })
        
        logger.info(f"✅ Fold {fold} completed: Best Val Loss: {best_val_loss:.4f}, Time: {fold_time/60:.1f} min")
    
    # Calculate aggregate statistics
    best_val_losses = [fold['best_val_loss'] for fold in fold_results]
    avg_best_val_loss = np.mean(best_val_losses)
    std_best_val_loss = np.std(best_val_losses)
    
    # Create results summary
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'k_folds': k_folds,
        'fold_results': fold_results,
        'total_training_time': total_training_time,
        'avg_best_val_loss': avg_best_val_loss,
        'std_best_val_loss': std_best_val_loss,
        'embeddings_shape': [dataset.vocab_size, config['embedding_dim']],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save embeddings
    embeddings_file = f"embeddings_{experiment_name}.npy"
    embeddings = model.track_embeddings.weight.detach().numpy()
    np.save(embeddings_file, embeddings)
    logger.info(f"💾 Embeddings saved: {embeddings_file}")
    
    # Save results
    results_file = f"experiment_results_{experiment_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved: {results_file}")
    
    return results, embeddings

def main():
    """Main execution function"""
    logger.info("🚀 Starting Second Round Hyperparameter Experiments")
    logger.info("=" * 60)
    
    # Configuration
    db_path = "data/MPD/mpd_database.db"
    max_samples = 100000  # 100K pairs for tuning
    k_folds = 3
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        logger.info("Please ensure the MPD database is built first")
        return
    
    # Get all experiments
    all_experiments = get_all_experiments()
    logger.info(f"📋 Total experiments planned: {len(all_experiments)}")
    
    # Initialize experiment manager
    manager = ExperimentManager()
    manager.total_experiments = len(all_experiments)
    
    # Group experiments by phase
    phases = defaultdict(list)
    for exp in all_experiments:
        phases[exp['phase']].append(exp)
    
    # Run experiments by phase
    all_results = []
    
    for phase_num in sorted(phases.keys()):
        logger.info(f"🎯 Starting Phase {phase_num}")
        logger.info(f"📊 Phase {phase_num} has {len(phases[phase_num])} experiments")
        
        phase_experiments = phases[phase_num]
        
        for i, experiment in enumerate(phase_experiments):
            experiment_name = experiment['name']
            
            # Skip if already completed
            if manager.is_completed(experiment_name):
                logger.info(f"⏭️ Skipping completed experiment: {experiment_name}")
                continue
            
            try:
                logger.info(f"🧪 Phase {phase_num} - Experiment {i+1}/{len(phase_experiments)}: {experiment_name}")
                
                # Run experiment
                results, embeddings = run_single_experiment(
                    experiment['config'],
                    experiment_name,
                    db_path,
                    max_samples,
                    k_folds
                )
                
                all_results.append(results)
                manager.mark_completed(experiment_name)
                
                # Progress update
                progress = manager.get_progress()
                logger.info(f"📈 Progress: {progress['completed']}/{progress['total']} "
                          f"({progress['progress']:.1f}%) - "
                          f"Elapsed: {progress['elapsed_time']/3600:.1f}h")
                
            except Exception as e:
                logger.error(f"❌ Experiment {experiment_name} failed: {e}")
                continue
    
    # Comprehensive summary
    logger.info("🎉 All experiments completed!")
    logger.info("📊 Final Results Summary:")
    
    if all_results:
        # Sort by performance
        all_results.sort(key=lambda x: x['avg_best_val_loss'])
        
        for i, result in enumerate(all_results):
            logger.info(f"   {i+1:2d}. {result['experiment_name']}: "
                       f"Val Loss: {result['avg_best_val_loss']:.4f} ± {result['std_best_val_loss']:.4f}, "
                       f"Time: {result['total_training_time']/60:.1f} min")
        
        # Find best configuration
        best_result = all_results[0]
        logger.info(f"🏆 Best configuration: {best_result['experiment_name']}")
        logger.info(f"   Best val loss: {best_result['avg_best_val_loss']:.4f} ± {best_result['std_best_val_loss']:.4f}")
        logger.info(f"   Config: {json.dumps(best_result['config'], indent=2)}")
        
        # Save best configuration for production training
        best_config_file = "best_second_round_config.json"
        with open(best_config_file, 'w') as f:
            json.dump(best_result['config'], f, indent=2)
        logger.info(f"💾 Best configuration saved: {best_config_file}")
        
        # Generate comprehensive analysis
        logger.info("📈 Generating comprehensive analysis...")
        
        # Create summary plot
        experiment_names = [result['experiment_name'] for result in all_results]
        val_losses = [result['avg_best_val_loss'] for result in all_results]
        val_stds = [result['std_best_val_loss'] for result in all_results]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(experiment_names)), val_losses, yerr=val_stds, 
                      capsize=5, alpha=0.7, color='lightcoral')
        plt.xlabel('Experiment')
        plt.ylabel('Validation Loss (lower is better)')
        plt.title('Second Round Hyperparameter Tuning Results')
        plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Highlight best result
        best_idx = 0
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(0.8)
        
        plt.savefig('second_round_results.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Summary plot saved: second_round_results.png")
        
        # Compare with first round
        logger.info("🔄 Comparison with First Round:")
        logger.info("   First round best: ~0.6936 (More Negatives)")
        logger.info(f"   Second round best: {best_result['avg_best_val_loss']:.4f} ({best_result['experiment_name']})")
        
        improvement = (0.6936 - best_result['avg_best_val_loss']) / 0.6936 * 100
        if improvement > 0:
            logger.info(f"   🎉 Improvement: {improvement:.2f}%")
        else:
            logger.info(f"   📉 No improvement: {improvement:.2f}%")
    
    else:
        logger.warning("⚠️ No experiments completed successfully")
    
    # Final progress report
    final_progress = manager.get_progress()
    total_time = final_progress['elapsed_time'] / 3600
    logger.info(f"⏱️ Total execution time: {total_time:.1f} hours")
    logger.info(f"📊 Final completion: {final_progress['completed']}/{final_progress['total']} experiments")

if __name__ == "__main__":
    main()
