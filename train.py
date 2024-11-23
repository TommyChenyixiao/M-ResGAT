# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import CoraDataset
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT
from models.resgat import ResGAT
from models.multihop_resgat import MultiHopResGAT
from utils.metrics import evaluate, plot_roc_curves

class Config:
    # General settings
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    HIDDEN_CHANNELS = 256
    DROPOUT = 0.3
    NUM_LAYERS = 3
    GAT_HEADS = 8
    NUM_HOPS = 2
    COMBINE_METHOD = 'attention'  # ['concat', 'sum', 'attention']
    
    # Training parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 500
    PATIENCE = 100
    MIN_IMPROVEMENT = 1e-4
    
    # Data parameters
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    
    # Paths
    DATA_ROOT = '/tmp/CitationFull'
    OUTPUT_DIR = 'outputs'
    
    @classmethod
    def create_output_dirs(cls):
        """Create necessary output directories"""
        dirs = ['models', 'plots', 'results']
        for dir_name in dirs:
            path = os.path.join(cls.OUTPUT_DIR, dir_name)
            os.makedirs(path, exist_ok=True)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set random seeds
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.SEED)
        
        # Create output directories
        config.create_output_dirs()
        
        # Load dataset
        self.load_data()
        
        # Initialize models
        self.initialize_models()

    def load_data(self):
        """Load and preprocess dataset"""
        dataset = CoraDataset(self.config.DATA_ROOT)
        self.data = dataset.get_data()
        self.stats = dataset.get_dataset_stats()
        
        # Move data to device
        self.data.x = self.data.x.to(self.device)
        self.data.edge_index = self.data.edge_index.to(self.device)
        self.data.y = self.data.y.to(self.device)
        
        # Get split indices
        indices = torch.randperm(self.stats['num_nodes'])
        train_size = int(self.stats['num_nodes'] * self.config.TRAIN_RATIO)
        val_size = int(self.stats['num_nodes'] * self.config.VAL_RATIO)
        
        self.train_idx = indices[:train_size]
        self.val_idx = indices[train_size:train_size + val_size]
        self.test_idx = indices[train_size + val_size:]

    def initialize_models(self):
        """Initialize all models"""
        self.models = {
            'GCN': GCN(
                self.stats['num_features'],
                self.config.HIDDEN_CHANNELS,
                self.stats['num_classes'],
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT
            ),
            'GraphSAGE': GraphSAGE(
                self.stats['num_features'],
                self.config.HIDDEN_CHANNELS,
                self.stats['num_classes'],
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT
            ),
            'GAT': GAT(
                self.stats['num_features'],
                self.config.HIDDEN_CHANNELS,
                self.stats['num_classes'],
                num_layers=self.config.NUM_LAYERS,
                heads=self.config.GAT_HEADS,
                dropout=self.config.DROPOUT
            ),
            'ResGAT': ResGAT(
                self.stats['num_features'],
                self.config.HIDDEN_CHANNELS,
                self.stats['num_classes'],
                num_layers=self.config.NUM_LAYERS,
                heads=self.config.GAT_HEADS,
                dropout=self.config.DROPOUT,
                residual=True
            ),
            'MultiHopResGAT': MultiHopResGAT(
                self.stats['num_features'],
                self.config.HIDDEN_CHANNELS,
                self.stats['num_classes'],
                num_layers=self.config.NUM_LAYERS,
                heads=self.config.GAT_HEADS,
                dropout=self.config.DROPOUT,
                residual=True,
                num_hops=self.config.NUM_HOPS,
                combine=self.config.COMBINE_METHOD
            )
        }
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)

    def train_model(self, model, model_name):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=1,
            eta_min=1e-6
        )
        
        best_val_acc = 0
        best_model_state = None
        early_stop_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_roc_auc': [],
            'learning_rates': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            # Training
            model.train()
            optimizer.zero_grad()
            out = model(self.data.x, self.data.edge_index)
            
            loss = F.cross_entropy(out[self.train_idx], self.data.y[self.train_idx])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                # Training metrics
                train_acc = accuracy_score(
                    self.data.y[self.train_idx].cpu(),
                    out[self.train_idx].argmax(dim=1).cpu()
                )
                
                # Validation metrics
                val_acc, val_f1, val_roc_data = evaluate(
                    model, self.data, self.val_idx, return_roc=True
                )
                
                # Store metrics
                history['train_loss'].append(loss.item())
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                history['val_roc_auc'].append(val_roc_data['roc_auc']['macro'])
                history['learning_rates'].append(scheduler.get_last_lr()[0])
                
                # Early stopping check
                if val_acc > best_val_acc + self.config.MIN_IMPROVEMENT:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1:03d}: "
                          f"Loss = {loss.item():.4f}, "
                          f"Train Acc = {train_acc:.4f}, "
                          f"Val Acc = {val_acc:.4f}, "
                          f"Val ROC = {val_roc_data['roc_auc']['macro']:.4f}")
        
        training_time = time.time() - start_time
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_acc, test_f1, test_roc_data = evaluate(model, self.data, self.test_idx, return_roc=True)
        
        # Save model and results
        self.save_results(model, model_name, history, {
            'training_time': training_time,
            'epochs': epoch + 1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_data['roc_auc']['macro']
        })
        
        return history, test_roc_data

    def save_results(self, model, model_name, history, metrics):
        """Save model, plots, and metrics"""
        base_path = os.path.join(self.config.OUTPUT_DIR, model_name + '_' + self.timestamp)
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(self.config.OUTPUT_DIR, 'models', f'{model_name}_model.pt'))
        
        # Save metrics
        with open(os.path.join(self.config.OUTPUT_DIR, 'results', f'{model_name}_metrics.json'), 'w') as f:
            json.dump({**metrics, 'history': history}, f, indent=4)
        
        # Generate and save plots
        self.plot_training_curves(history, model_name)

    def plot_training_curves(self, history, model_name):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, history['train_acc'], label='Train')
        ax2.plot(epochs, history['val_acc'], label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # ROC AUC
        ax3.plot(epochs, history['val_roc_auc'])
        ax3.set_title('Validation ROC AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ROC AUC')
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(epochs, history['learning_rates'])
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'plots', f'{model_name}_training.png'))
        plt.close()

    def train_all_models(self):
        """Train all models and compare results"""
        results = {}
        
        for model_name, model in self.models.items():
            history, test_roc_data = self.train_model(model, model_name)
            results[model_name] = {
                'history': history,
                'test_roc_data': test_roc_data
            }
            
            # Plot ROC curves
            plot_roc_curves(
                test_roc_data,
                model_name,
                os.path.join(self.config.OUTPUT_DIR, 'plots', f'{model_name}_roc.png')
            )
        
        # Generate comparison plot
        self.plot_model_comparison(results)
        
        return results

    def plot_model_comparison(self, results):
        """Plot comparison of model performances"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Compare validation accuracy
        for model_name, result in results.items():
            ax1.plot(result['history']['val_acc'], label=model_name)
        ax1.set_title('Validation Accuracy Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Compare ROC AUC
        for model_name, result in results.items():
            ax2.plot(result['history']['val_roc_auc'], label=model_name)
        ax2.set_title('Validation ROC AUC Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ROC AUC')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'plots', 'model_comparison.png'))
        plt.close()

def main():
    trainer = Trainer(Config)
    results = trainer.train_all_models()
    
    # Print final results
    print("\nFinal Results:")
    print("=" * 100)
    print(f"{'Model':<15} {'Test Acc':<10} {'Test F1':<10} {'Test ROC AUC':<10}")
    print("-" * 100)
    
    for model_name, result in results.items():
        history = result['history']
        test_roc_data = result['test_roc_data']
        print(f"{model_name:<15} "
              f"{max(history['val_acc']):.4f}    "
              f"{max(history['val_f1']):.4f}    "
              f"{test_roc_data['roc_auc']['macro']:.4f}")

if __name__ == "__main__":
    main()