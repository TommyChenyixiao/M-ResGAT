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

from data.dataset import DatasetHandler
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT
from models.resgat import ResGAT
from models.mresgat import MultiHopResGAT
from models.M_resgat import MGAT
from models.M_mresgat import MultiHopMGAT
from utils.metrics import evaluate, plot_roc_curves

class Config:
    # General settings
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Common model parameters
    NUM_HOPS = 2
    COMBINE_METHOD = 'attention'  # ['concat', 'sum', 'attention']
    EPOCHS = 1000
    PATIENCE = 100
    MIN_IMPROVEMENT = 1e-4
    
    # Dataset settings
    DATASET_CONFIGS = {
        'Cora_ML': {
            'hidden_channels': 256,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gat_heads': 8,
            'num_layers': 3,
            'num_hops': 2,
            'combine_method': 'attention',
            'beta': 0.5  # Balance between first-order and motif attention
        },
        'Cora': {
            'hidden_channels': 256,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gat_heads': 8,
            'num_layers': 3,
            'num_hops': 2,
            'combine_method': 'attention',
            'beta': 0.5
        },
        'CiteSeer': {
            'hidden_channels': 256,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gat_heads': 8,
            'num_layers': 3,
            'num_hops': 2,
            'combine_method': 'attention',
            'beta': 0.5
        }
    }
    
    def __init__(self, dataset_name='Cora_ML'):
        # Dataset name
        self.DATASET_NAME = dataset_name
        
        # Training parameters (common across all datasets)
        self.EPOCHS = 1000
        self.PATIENCE = 100
        self.MIN_IMPROVEMENT = 1e-4
        
        # Load dataset-specific configurations
        self._load_dataset_config()
        
        # Setup paths
        self._setup_paths()
    
    def _load_dataset_config(self):
        """Load dataset-specific configurations"""
        if self.DATASET_NAME not in self.DATASET_CONFIGS:
            raise ValueError(f"No configuration for dataset {self.DATASET_NAME}")
            
        config = self.DATASET_CONFIGS[self.DATASET_NAME]
        
        # Model architecture parameters
        self.HIDDEN_CHANNELS = config['hidden_channels']
        self.NUM_LAYERS = config['num_layers']
        self.DROPOUT = config['dropout']
        self.GAT_HEADS = config['gat_heads']
        self.NUM_HOPS = config['num_hops']
        self.COMBINE_METHOD = config['combine_method']
        self.MGAT_BETA = config['beta']

        # Optimization parameters
        self.LEARNING_RATE = config['learning_rate']
        self.WEIGHT_DECAY = config['weight_decay']
    
    def _setup_paths(self):
        """Setup paths for dataset"""
        self.DATA_ROOT = '/tmp/CitationFull'
        self.OUTPUT_DIR = f'outputs/{self.DATASET_NAME.lower()}'
        
        # Create output directories
        for dir_name in ['models', 'plots', 'results']:
            path = os.path.join(self.OUTPUT_DIR, dir_name)
            os.makedirs(path, exist_ok=True)
    
    def __str__(self):
        """String representation of configuration"""
        return (
            f"Configuration for {self.DATASET_NAME}:\n"
            f"  Model Parameters:\n"
            f"    Hidden Channels: {self.HIDDEN_CHANNELS}\n"
            f"    Number of Layers: {self.NUM_LAYERS}\n"
            f"    Dropout: {self.DROPOUT}\n"
            f"    GAT Heads: {self.GAT_HEADS}\n"
            f"    Number of Hops: {self.NUM_HOPS}\n"
            f"    Combine Method: {self.COMBINE_METHOD}\n"
            f"    MGAT Beta: {self.MGAT_BETA}\n"
            f"  Training Parameters:\n"
            f"    Learning Rate: {self.LEARNING_RATE}\n"
            f"    Weight Decay: {self.WEIGHT_DECAY}\n"
            f"    Epochs: {self.EPOCHS}\n"
            f"    Patience: {self.PATIENCE}\n"
            f"    Min Improvement: {self.MIN_IMPROVEMENT}\n"
            f"  Paths:\n"
            f"    Data Root: {self.DATA_ROOT}\n"
            f"    Output Directory: {self.OUTPUT_DIR}"
        )

class TransductiveTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_environment()
    
    def _setup_environment(self):
        # Set random seeds
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.SEED)
        
        # Load dataset
        self.dataset_handler = DatasetHandler(
            self.config.DATASET_NAME,
            self.config.DATA_ROOT
        )
        self.data = self.dataset_handler.get_data()
        self.stats = self.dataset_handler.get_dataset_stats()
        
        # Print dataset statistics
        self.dataset_handler.print_stats()
        
        # Create masks if they don't exist
        if not hasattr(self.data, 'train_mask'):
            self._create_masks()
        
        # Move data to device
        self.data = self.data.to(self.device)
        
        # Initialize models
        self._initialize_models()
        
    def _create_masks(self):
        """Create stratified masks for transductive learning"""
        num_nodes = self.data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Stratified split for each class
        for c in range(self.data.y.max().item() + 1):
            idx = (self.data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))]
            
            n_train = int(0.8 * idx.size(0))
            n_val = int(0.1 * idx.size(0))
            
            train_mask[idx[:n_train]] = True
            val_mask[idx[n_train:n_train + n_val]] = True
            test_mask[idx[n_train + n_val:]] = True
        
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask
        
    def _print_data_stats(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print(f"Nodes: {self.data.num_nodes}")
        print(f"Edges: {self.data.num_edges}")
        print(f"Features: {self.data.num_features}")
        print(f"Classes: {len(self.data.y.unique())}")
        print(f"Training nodes: {self.data.train_mask.sum().item()}")
        print(f"Validation nodes: {self.data.val_mask.sum().item()}")
        print(f"Test nodes: {self.data.test_mask.sum().item()}")
        
    def _initialize_models(self):
        """Initialize all model architectures"""
        self.models = {
            # 'GCN': GCN(
            #     self.stats['num_features'],
            #     self.config.HIDDEN_CHANNELS,
            #     self.stats['num_classes'],
            #     num_layers=self.config.NUM_LAYERS,
            #     dropout=self.config.DROPOUT
            # ),
            # 'GraphSAGE': GraphSAGE(
            #     self.stats['num_features'],
            #     self.config.HIDDEN_CHANNELS,
            #     self.stats['num_classes'],
            #     num_layers=self.config.NUM_LAYERS,
            #     dropout=self.config.DROPOUT
            # ),
            # 'GAT': GAT(
            #     self.stats['num_features'],
            #     self.config.HIDDEN_CHANNELS,
            #     self.stats['num_classes'],
            #     num_layers=self.config.NUM_LAYERS,
            #     heads=self.config.GAT_HEADS,
            #     dropout=self.config.DROPOUT
            # ),
            # 'ResGAT': ResGAT(
            #     self.stats['num_features'],
            #     self.config.HIDDEN_CHANNELS,
            #     self.stats['num_classes'],
            #     num_layers=self.config.NUM_LAYERS,
            #     heads=self.config.GAT_HEADS,
            #     dropout=self.config.DROPOUT,
            #     residual=True
            # ),
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
            ),
            'M-resGAT': MGAT(
            self.stats['num_features'],
            self.config.HIDDEN_CHANNELS,
            self.stats['num_classes'],
            num_layers=self.config.NUM_LAYERS,
            heads=self.config.GAT_HEADS,
            dropout=self.config.DROPOUT,
            beta=self.config.MGAT_BETA,
            residual=True
            ),
            'MultiHopM-ResGAT': MultiHopMGAT(
            self.stats['num_features'],
            self.config.HIDDEN_CHANNELS,
            self.stats['num_classes'],
            num_layers=self.config.NUM_LAYERS,
            heads=self.config.GAT_HEADS,
            dropout=self.config.DROPOUT,
            beta=self.config.MGAT_BETA,
            num_hops=self.config.NUM_HOPS,
            residual=True
            )
        }
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
            
    def _train_epoch(self, model, optimizer):
        """Train for one epoch in transductive setting"""
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on entire graph
        out = model(self.data.x, self.data.edge_index)
        
        # Compute loss only on training nodes
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute metrics
        pred = out[self.data.train_mask].argmax(dim=1)
        acc = (pred == self.data.y[self.data.train_mask]).float().mean().item()
        
        return loss.item(), acc
        
    @torch.no_grad()
    def _evaluate(self, model, mask):
        """Evaluate model on specific mask"""
        model.eval()
        out = model(self.data.x, self.data.edge_index)
        pred = out[mask].argmax(dim=1)
        
        acc = (pred == self.data.y[mask]).float().mean().item()
        f1 = f1_score(
            self.data.y[mask].cpu().numpy(),
            pred.cpu().numpy(),
            average='macro'
        )
        
        return acc, f1
        
    def train_model(self, model, model_name):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-6
        )
        
        best_val_acc = 0
        best_model_state = None
        early_stop_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': [],  # Add AUC tracking
            'learning_rates': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            # Training
            train_loss, train_acc = self._train_epoch(model, optimizer)
            scheduler.step()
            
            # Validation with ROC
            val_acc, val_f1, val_roc_data = evaluate(
                model, self.data, self.data.val_mask, return_roc=True
            )
            val_auc = val_roc_data['roc_auc']['macro']
            
            # Store metrics (avoid storing non-serializable ROC data)
            history['train_loss'].append(float(train_loss))  # Convert to Python float
            history['train_acc'].append(float(train_acc))
            history['val_acc'].append(float(val_acc))
            history['val_f1'].append(float(val_f1))
            history['val_auc'].append(float(val_auc))
            history['learning_rates'].append(float(scheduler.get_last_lr()[0]))
            
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
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:03d}: "
                    f"Loss = {train_loss:.4f}, "
                    f"Train Acc = {train_acc:.4f}, "
                    f"Val Acc = {val_acc:.4f}, "
                    f"Val AUC = {val_auc:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate best model
        model.load_state_dict(best_model_state)
        test_acc, test_f1, test_roc_data = evaluate(
            model, self.data, self.data.test_mask, return_roc=True
        )
        test_auc = test_roc_data['roc_auc']['macro']
        
        metrics = {
            'training_time': training_time,
            'epochs': epoch + 1,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc
        }
        
        # Store ROC curves
        plot_roc_curves(
            test_roc_data,
            model_name,
            os.path.join(self.config.OUTPUT_DIR, 'plots', f'{model_name}_roc.png')
        )
        
        # Save results (without ROC data)
        self._save_results(model, model_name, history, metrics)
        
        return history, metrics
        
    def _save_results(self, model, model_name, history, metrics):
        """Save training results"""
        # Save model
        model_path = os.path.join(self.config.OUTPUT_DIR, 'models', f'{model_name}_model.pt')
        torch.save(model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.config.OUTPUT_DIR, 'results', f'{model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({**metrics, 'history': history}, f, indent=4)
        
        # Generate plots
        self._plot_training_curves(history, model_name)
        
    def _plot_training_curves(self, history, model_name):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training curves
        ax1.plot(epochs, history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(epochs, history['train_acc'], label='Train')
        ax2.plot(epochs, history['val_acc'], label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(epochs, history['val_f1'])
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.grid(True)
        
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
            history, metrics = self.train_model(model, model_name)
            results[model_name] = {
                'history': history,
                'metrics': metrics
            }
        
        self._plot_model_comparison(results)
        self._print_final_results(results)
        
        return results
        
    def _plot_model_comparison(self, results):
        """Plot comparison of model performances"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        for model_name, result in results.items():
            history = result['history']
            ax1.plot(history['train_acc'], label=f'{model_name} (Train)')
            ax1.plot(history['val_acc'], label=f'{model_name} (Val)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # F1 Score comparison
        for model_name, result in results.items():
            ax2.plot(result['history']['val_f1'], label=model_name)
        ax2.set_title('Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        # AUC comparison
        for model_name, result in results.items():
            ax3.plot(result['history']['val_auc'], label=model_name)
        ax3.set_title('Validation Macro AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate
        for model_name, result in results.items():
            ax4.plot(result['history']['learning_rates'], label=model_name)
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'plots', 'model_comparison.png'))
        plt.close()
        
    def _print_final_results(self, results):
        """Print final comparison results"""
        print("\nFinal Results:")
        print("=" * 120)
        print(f"{'Model':<15} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} "
            f"{'Test F1':<10} {'Test AUC':<10} {'Time(s)':<10} {'Epochs'}")
        print("-" * 120)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            history = result['history']
            
            print(f"{model_name:<15} "
                f"{max(history['train_acc']):.4f}    "
                f"{metrics['best_val_acc']:.4f}    "
                f"{metrics['test_acc']:.4f}    "
                f"{metrics['test_f1']:.4f}    "
                f"{metrics['test_auc']:.4f}    "
                f"{metrics['training_time']:.1f}      "
                f"{metrics['epochs']}")
        
        # Save results to file
        results_path = os.path.join(self.config.OUTPUT_DIR, 'results', 'final_results.txt')
        with open(results_path, 'w') as f:
            f.write("Final Results Summary\n")
            f.write("=" * 120 + "\n")
            headers = ['Model', 'Train Acc', 'Val Acc', 'Test Acc', 
                    'Test F1', 'Test AUC', 'Time(s)', 'Epochs']
            f.write(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} "
                f"{headers[3]:<10} {headers[4]:<10} {headers[5]:<10} "
                f"{headers[6]:<10} {headers[7]}\n")
            f.write("-" * 120 + "\n")
            
            for model_name, result in results.items():
                metrics = result['metrics']
                history = result['history']
                
                f.write(f"{model_name:<15} "
                    f"{max(history['train_acc']):.4f}    "
                    f"{metrics['best_val_acc']:.4f}    "
                    f"{metrics['test_acc']:.4f}    "
                    f"{metrics['test_f1']:.4f}    "
                    f"{metrics['test_auc']:.4f}    "
                    f"{metrics['training_time']:.1f}      "
                    f"{metrics['epochs']}\n")
            
            # Add detailed statistics
            f.write("\nTraining Details:\n")
            f.write("-" * 60 + "\n")
            for model_name, result in results.items():
                metrics = result['metrics']
                f.write(f"\n{model_name}:\n")
                f.write(f"  Training Time: {metrics['training_time']:.2f} seconds\n")
                f.write(f"  Total Epochs: {metrics['epochs']}\n")
                f.write(f"  Best Validation Accuracy: {metrics['best_val_acc']:.4f}\n")
                f.write(f"  Final Test Accuracy: {metrics['test_acc']:.4f}\n")
                f.write(f"  Final Test F1 Score: {metrics['test_f1']:.4f}\n")
                f.write(f"  Final Test Macro AUC: {metrics['test_auc']:.4f}\n")

def main():
    """Main execution function"""
    # Available datasets
    datasets = ['Cora_ML', 'CiteSeer']
    
    for dataset_name in datasets:
        print(f"\nTraining on {dataset_name} dataset")
        print("=" * 50)
        
        config = Config(dataset_name)
        trainer = TransductiveTrainer(config)
        results = trainer.train_all_models()
        
        print(f"\nCompleted training on {dataset_name}")
        print(f"Results saved in: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise