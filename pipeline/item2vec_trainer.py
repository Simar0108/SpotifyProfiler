#!/usr/bin/env python3
"""
Item2Vec Trainer - Comprehensive hyperparameter tuning for music track embeddings

This script:
1. Implements item2vec (skip-gram) architecture with extensive hyperparameter search
2. Supports systematic hyperparameter tuning with grid search and random exploration
3. Includes comprehensive monitoring, checkpointing, and validation
4. Scales from small tuning experiments to full production training

Key features:
- Grid search across multiple hyperparameter dimensions
- Real-time progress monitoring and memory tracking
- K-fold cross-validation for robust evaluation
- Adaptive experiment selection and early stopping
- Comprehensive result analysis and visualization
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
warnings.filterwarnings('ignore')

# Set up logging with timestamps
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
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get positive pair and generate negative samples"""
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
    """Enhanced Item2Vec neural network with configurable architecture"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, dropout_rate=0.1):
        super(Item2VecModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.track_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layers with configurable dropout
        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier initialization"""
        nn.init.xavier_uniform_(self.track_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        
        # Initialize hidden layers
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, pos_track, context_track, neg_tracks):
        """Forward pass with positive and negative samples"""
        
        # Get embeddings
        pos_emb = self.track_embeddings(pos_track)
        context_emb = self.context_embeddings(context_track)
        neg_emb = self.track_embeddings(neg_tracks)
        
        # Apply hidden layers
        pos_emb = self.hidden(pos_emb)
        context_emb = self.hidden(context_emb)
        neg_emb = self.hidden(neg_emb)
        
        # Calculate similarities
        pos_sim = torch.sum(pos_emb * context_emb, dim=1)
        neg_sim = torch.sum(pos_emb.unsqueeze(1) * neg_emb, dim=2)
        
        return pos_sim, neg_sim
    
    def get_embeddings(self):
        """Get final embeddings for all tracks"""
        return self.track_embeddings.weight.detach().cpu().numpy()

class Item2VecTrainer:
    """Enhanced trainer class with comprehensive hyperparameter tuning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.memory_usage = []
        self.training_speeds = []
        
        # Performance tracking
        self.start_time = None
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info(f"🚀 Initializing trainer on {self.device}")
        logger.info(f"⚙️ Configuration: {json.dumps(config, indent=2)}")
    
    def setup_model(self, vocab_size):
        """Initialize model with current hyperparameters"""
        self.model = Item2VecModel(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config.get('hidden_dim', 128),
            dropout_rate=self.config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # Setup optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0),
                betas=(0.9, 0.999)
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0.0),
                nesterov=True
            )
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif self.config['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0),
                momentum=0.9
            )
        
        # Setup scheduler
        if self.config.get('lr_scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 1000),
                gamma=self.config.get('lr_gamma', 0.9)
            )
        elif self.config.get('lr_scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('lr_t_max', 1000)
            )
        elif self.config.get('lr_scheduler') == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.get('lr_gamma', 0.95)
            )
        elif self.config.get('lr_scheduler') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with enhanced monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            pos_track = batch['pos_track'].to(self.device)
            context_track = batch['context_track'].to(self.device)
            neg_tracks = batch['neg_tracks'].to(self.device)
            
            # Forward pass
            pos_sim, neg_sim = self.model(pos_track, context_track, neg_tracks)
            
            # Calculate loss (negative log likelihood)
            pos_loss = -F.logsigmoid(pos_sim).mean()
            neg_loss = -F.logsigmoid(-neg_sim).mean()
            loss = pos_loss + neg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update learning rate history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.6f}',
                'Memory': f'{self.get_memory_usage():.1f}MB'
            })
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                pos_track = batch['pos_track'].to(self.device)
                context_track = batch['context_track'].to(self.device)
                neg_tracks = batch['neg_tracks'].to(self.device)
                
                pos_sim, neg_sim = self.model(pos_track, context_track, neg_tracks)
                
                pos_loss = -F.logsigmoid(pos_sim).mean()
                neg_loss = -F.logsigmoid(-neg_sim).mean()
                loss = pos_loss + neg_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataloader, val_dataloader=None, epochs=10):
        """Enhanced training loop with comprehensive monitoring"""
        logger.info(f"🎯 Starting training for {epochs} epochs")
        self.start_time = time.time()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 5)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # Update scheduler if using ReduceLROnPlateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                    break
            
            epoch_time = time.time() - epoch_start_time
            
            # Memory tracking
            memory_mb = self.get_memory_usage()
            self.memory_usage.append(memory_mb)
            
            # Training speed
            if len(self.train_losses) > 1:
                speed = (epoch + 1) / (time.time() - self.start_time) * 60  # epochs per minute
                self.training_speeds.append(speed)
            
            # Logging
            log_msg = f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}"
            if val_loss:
                log_msg += f", Val Loss: {val_loss:.4f}"
            log_msg += f", Time: {epoch_time:.1f}s, Memory: {memory_mb:.1f}MB"
            
            logger.info(log_msg)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_freq', 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Restore best model if early stopping occurred
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("🔄 Restored best model from early stopping")
        
        total_time = time.time() - self.start_time
        logger.info(f"✅ Training completed in {total_time/60:.1f} minutes!")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename):
        """Save model checkpoint with enhanced metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'memory_usage': self.memory_usage,
            'training_speeds': self.training_speeds,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint with enhanced metadata"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.memory_usage = checkpoint.get('memory_usage', [])
        self.training_speeds = checkpoint.get('training_speeds', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"🔄 Checkpoint loaded: {filename}")
    
    def get_embeddings(self):
        """Extract final embeddings"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.get_embeddings()
    
    def plot_training_curves(self, save_path=None):
        """Enhanced training curves with comprehensive metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Results - {self.config.get("experiment_name", "Item2Vec")}', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory usage
        if self.memory_usage:
            axes[0, 2].plot(self.memory_usage, color='purple', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Memory (MB)')
            axes[0, 2].set_title('Memory Usage')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Training speed
        if self.training_speeds:
            axes[1, 0].plot(self.training_speeds, color='orange', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Speed (epochs/min)')
            axes[1, 0].set_title('Training Speed')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[1, 1].hist(self.train_losses, bins=20, alpha=0.7, color='blue', label='Train')
        if self.val_losses:
            axes[1, 1].hist(self.val_losses, bins=20, alpha=0.7, color='red', label='Val')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Configuration summary
        config_text = f"""Configuration:
Embedding Dim: {self.config.get('embedding_dim', 'N/A')}
Hidden Dim: {self.config.get('hidden_dim', 'N/A')}
Learning Rate: {self.config.get('learning_rate', 'N/A')}
Batch Size: {self.config.get('batch_size', 'N/A')}
Optimizer: {self.config.get('optimizer', 'N/A')}
Negative Samples: {self.config.get('negative_samples', 'N/A')}
Final Train Loss: {self.train_losses[-1]:.4f if self.train_losses else 'N/A'}
Best Val Loss: {min(self.val_losses) if self.val_losses else 'N/A'}"""
        
        axes[1, 2].text(0.1, 0.5, config_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Configuration Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training curves saved: {save_path}")
        
        plt.show()

def run_hyperparameter_experiment(config, experiment_name, max_samples=100000, k_folds=3):
    """Run a single hyperparameter experiment with K-fold cross-validation"""
    logger.info(f"🧪 Starting experiment: {experiment_name}")
    logger.info(f"⚙️ Config: {json.dumps(config, indent=2)}")
    
    # Setup data
    dataset = CoOccurrenceDataset(
        db_path="data/MPD/mpd_database.db",
        max_samples=max_samples,
        negative_samples=config['negative_samples']
    )
    
    # K-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"🔄 Fold {fold + 1}/{k_folds}")
        
        # Split data
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Setup trainer
        trainer = Item2VecTrainer(config)
        trainer.setup_model(dataset.vocab_size)
        
        # Train
        start_time = time.time()
        train_losses, val_losses = trainer.train(
            train_loader,
            val_loader,
            epochs=config['epochs']
        )
        training_time = time.time() - start_time
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': training_time,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None
        }
        fold_results.append(fold_result)
        
        logger.info(f"✅ Fold {fold + 1} completed: "
                   f"Best Val Loss: {fold_result['best_val_loss']:.4f}, "
                   f"Time: {training_time/60:.1f} min")
    
    # Aggregate results across folds
    best_val_losses = [fold['best_val_loss'] for fold in fold_results if fold['best_val_loss']]
    avg_best_val_loss = np.mean(best_val_losses) if best_val_losses else float('inf')
    std_best_val_loss = np.std(best_val_losses) if best_val_losses else 0
    
    total_training_time = sum(fold['training_time'] for fold in fold_results)
    
    # Get final embeddings from last fold
    embeddings = trainer.get_embeddings()
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'k_folds': k_folds,
        'fold_results': fold_results,
        'total_training_time': total_training_time,
        'avg_best_val_loss': avg_best_val_loss,
        'std_best_val_loss': std_best_val_loss,
        'embeddings_shape': embeddings.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save experiment results
    results_file = f"experiment_results_{experiment_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save embeddings
    embeddings_file = f"embeddings_{experiment_name}.npy"
    np.save(embeddings_file, embeddings)
    
    # Plot training curves from last fold
    plot_file = f"training_curves_{experiment_name}.png"
    trainer.plot_training_curves(save_path=plot_file)
    
    logger.info(f"✅ Experiment {experiment_name} completed!")
    logger.info(f"📊 Results: {results_file}")
    logger.info(f"💾 Embeddings: {embeddings_file}")
    logger.info(f"📈 Total training time: {total_training_time/60:.1f} minutes")
    logger.info(f"📊 Cross-validation: {avg_best_val_loss:.4f} ± {std_best_val_loss:.4f}")
    
    return results, embeddings

def generate_hyperparameter_grid():
    """Generate comprehensive hyperparameter search space"""
    
    # Define hyperparameter ranges
    hyperparams = {
        'embedding_dim': [64, 128, 256],
        'hidden_dim': [64, 128, 256, 512],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
        'batch_size': [512, 1024, 2048, 4096, 8192],
        'negative_samples': [5, 10, 15, 20, 25],
        'optimizer': ['adam', 'sgd', 'adagrad', 'rmsprop'],
        'weight_decay': [0.0, 0.0001, 0.001, 0.01],
        'dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'lr_scheduler': [None, 'step', 'cosine', 'exponential', 'plateau'],
        'epochs': [10, 15, 20, 25],
        'patience': [3, 5, 7, 10]
    }
    
    # Generate all combinations
    keys = list(hyperparams.keys())
    combinations = list(itertools.product(*[hyperparams[key] for key in keys]))
    
    # Convert to list of configs
    configs = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        # Add scheduler-specific parameters
        if config['lr_scheduler'] == 'step':
            config['lr_step_size'] = 1000
            config['lr_gamma'] = 0.9
        elif config['lr_scheduler'] == 'cosine':
            config['lr_t_max'] = 1000
        elif config['lr_scheduler'] == 'exponential':
            config['lr_gamma'] = 0.95
        
        configs.append(config)
    
    return configs

def main():
    """Main function to run comprehensive hyperparameter experiments"""
    
    # Define focused hyperparameter experiments for systematic tuning
    experiments = [
        {
            'name': 'baseline_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.01,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'lr_scheduler': None,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': 'high_lr_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.1,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'lr_scheduler': None,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': 'large_batch_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.01,
                'batch_size': 4096,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'lr_scheduler': None,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': 'more_negatives_64d',
            'config': {
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
        },
        {
            'name': 'sgd_optimizer_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.01,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'sgd',
                'weight_decay': 0.0001,
                'dropout_rate': 0.1,
                'lr_scheduler': 'step',
                'lr_step_size': 1000,
                'lr_gamma': 0.9,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': '128d_embeddings',
            'config': {
                'embedding_dim': 128,
                'hidden_dim': 256,
                'learning_rate': 0.01,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'lr_scheduler': None,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': 'high_dropout_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.01,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.3,
                'lr_scheduler': None,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        },
        {
            'name': 'cosine_scheduler_64d',
            'config': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'learning_rate': 0.01,
                'batch_size': 1024,
                'negative_samples': 10,
                'optimizer': 'adam',
                'weight_decay': 0.0,
                'dropout_rate': 0.1,
                'lr_scheduler': 'cosine',
                'lr_t_max': 1000,
                'epochs': 15,
                'patience': 5,
                'checkpoint_freq': 5
            }
        }
    ]
    
    logger.info(f"🎯 Running {len(experiments)} hyperparameter experiments")
    logger.info(f"📊 Each experiment uses 100K pairs with 3-fold cross-validation")
    
    # Run experiments
    all_results = []
    
    for i, experiment in enumerate(experiments, 1):
        try:
            logger.info(f"🧪 Experiment {i}/{len(experiments)}: {experiment['name']}")
            
            results, embeddings = run_hyperparameter_experiment(
                experiment['config'],
                experiment['name'],
                max_samples=100000,  # 100K pairs for tuning
                k_folds=3  # 3-fold cross-validation
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"❌ Experiment {experiment['name']} failed: {e}")
            continue
    
    # Comprehensive summary of results
    logger.info("🎉 All experiments completed!")
    logger.info("📊 Experiment Summary:")
    
    for result in all_results:
        logger.info(f"   {result['experiment_name']}: "
                   f"Val Loss: {result['avg_best_val_loss']:.4f} ± {result['std_best_val_loss']:.4f}, "
                   f"Time: {result['total_training_time']/60:.1f} min")
    
    # Find best configuration
    if all_results:
        best_result = min(all_results, key=lambda x: x['avg_best_val_loss'])
        logger.info(f"🏆 Best configuration: {best_result['experiment_name']}")
        logger.info(f"   Best val loss: {best_result['avg_best_val_loss']:.4f} ± {best_result['std_best_val_loss']:.4f}")
        logger.info(f"   Config: {json.dumps(best_result['config'], indent=2)}")
        
        # Save best configuration for production training
        best_config_file = "best_hyperparameters.json"
        with open(best_config_file, 'w') as f:
            json.dump(best_result['config'], f, indent=2)
        logger.info(f"💾 Best configuration saved: {best_config_file}")
    
    # Generate comprehensive analysis
    logger.info("📈 Generating comprehensive analysis...")
    
    # Create summary plot
    if all_results:
        experiment_names = [result['experiment_name'] for result in all_results]
        val_losses = [result['avg_best_val_loss'] for result in all_results]
        val_stds = [result['std_best_val_loss'] for result in all_results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(experiment_names)), val_losses, yerr=val_stds, 
                      capsize=5, alpha=0.7, color='skyblue')
        plt.xlabel('Experiment')
        plt.ylabel('Validation Loss (lower is better)')
        plt.title('Hyperparameter Tuning Results')
        plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Highlight best result
        best_idx = np.argmin(val_losses)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(0.8)
        
        plt.savefig('hyperparameter_tuning_summary.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Summary plot saved: hyperparameter_tuning_summary.png")
        plt.show()

if __name__ == "__main__":
    main()