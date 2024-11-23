# test_models.py
import unittest
import torch
import torch.nn.functional as F
from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import NormalizeFeatures
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam

# Import our implementations
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT
from models.resgat import ResGAT
from models.multihop_resgat import MultiHopResGAT
from data.dataset import CoraDataset
from utils.train_utils import get_split_idx, train_model
from utils.metrics import evaluate, plot_roc_curves

class ModelTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set up dataset
        cls.dataset = CoraDataset()
        cls.data = cls.dataset.get_data()
        cls.stats = cls.dataset.get_dataset_stats()
        
        # Modified split ratios for better performance
        cls.train_ratio = 0.8  # Increased training data
        cls.val_ratio = 0.1
        
        # Get split indices
        cls.train_idx, cls.val_idx, cls.test_idx = get_split_idx(
            cls.stats['num_nodes'], 
            cls.train_ratio, 
            cls.val_ratio
        )
        
        # Create directory for test outputs
        cls.output_dir = 'test_outputs'
        os.makedirs(cls.output_dir, exist_ok=True)

    def setUp(self):
        """Initialize models before each test"""
        # Adjusted hyperparameters
        self.hidden_channels = 256
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4

        self.models = {
            'GCN': GCN(
                self.stats['num_features'], 
                self.hidden_channels, 
                self.stats['num_classes'],
                dropout=self.dropout,
                num_layers=3
            ),
            'GraphSAGE': GraphSAGE(
                self.stats['num_features'], 
                self.hidden_channels, 
                self.stats['num_classes'],
                dropout=self.dropout,
                num_layers=3
            ),
            'GAT': GAT(
                self.stats['num_features'], 
                self.hidden_channels, 
                self.stats['num_classes'],
                num_layers=3,
                heads=8,
                dropout=self.dropout
            ),
            'ResGAT': ResGAT(
                self.stats['num_features'],
                self.hidden_channels,
                self.stats['num_classes'],
                num_layers=3,
                heads=8,
                dropout=self.dropout,
                residual=True
            ),
            'MultiHopResGAT': MultiHopResGAT(
                self.stats['num_features'],
                self.hidden_channels,
                self.stats['num_classes'],
                num_layers=3,
                heads=8,
                dropout=self.dropout,
                residual=True,
                num_hops=2,
                combine='attention'
            )
        }
    
    def test_resgat_architecture(self):
        """Test ResGAT architecture"""
        model = self.models['ResGAT']
        
        # Test input layer
        self.assertTrue(hasattr(model, 'input_layer'))
        self.assertTrue(hasattr(model.input_layer, 'gat'))
        self.assertTrue(hasattr(model.input_layer, 'norm'))
        
        # Test residual connections
        self.assertTrue(hasattr(model.input_layer, 'residual'))
        
        # Test output dimensions
        output = model(self.data.x, self.data.edge_index)
        self.assertEqual(output.shape, (self.stats['num_nodes'], self.stats['num_classes']))
    
    def test_multihop_resgat_architecture(self):
        """Test Multi-hop ResGAT architecture"""
        model = self.models['MultiHopResGAT']
        
        # Test multi-hop structure
        self.assertTrue(hasattr(model.input_layer, 'gat_layers'))
        self.assertEqual(len(model.input_layer.gat_layers), 2)  # num_hops=2
        
        # Test attention mechanism
        self.assertTrue(hasattr(model.input_layer, 'attention'))
        
        # Test output dimensions
        output = model(self.data.x, self.data.edge_index)
        self.assertEqual(output.shape, (self.stats['num_nodes'], self.stats['num_classes']))

    def train_model(self, model, model_name, num_epochs=1000, patience=100):  # Increased epochs and patience
        """Train a model and return performance metrics"""
        optimizer = Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=20, 
            min_lr=1e-5
        )
        
        # Initialize tracking variables
        best_val_acc = 0
        best_model_state = None
        early_stop_counter = 0
        training_history = {
            'train_loss': [],
            'val_acc': [],
            'val_roc_auc': []
        }
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            output = model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(output[self.train_idx], self.data.y[self.train_idx])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_acc, _, val_roc_data = evaluate(model, self.data, self.val_idx, return_roc=True)
                
                # Update scheduler
                scheduler.step(val_acc)
                
                # Store metrics
                training_history['train_loss'].append(loss.item())
                training_history['val_acc'].append(val_acc)
                training_history['val_roc_auc'].append(val_roc_data['roc_auc']['macro'])
                
                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    print(f"{model_name} - Early stopping at epoch {epoch}")
                    break
                
                # Print progress
                if (epoch + 1) % 100 == 0:
                    print(f"{model_name} - Epoch {epoch+1}: Loss = {loss.item():.4f}, "
                          f"Val Acc = {val_acc:.4f}, Val ROC = {val_roc_data['roc_auc']['macro']:.4f}")
        
        training_time = time.time() - start_time
        
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        test_acc, test_f1, test_roc_data = evaluate(model, self.data, self.test_idx, return_roc=True)
        
        # Plot training curves
        self.plot_training_curves(training_history, model_name)
        
        return {
            'training_time': training_time,
            'epochs': len(training_history['train_loss']),
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_data['roc_auc']['macro'],
            'training_history': training_history,
            'test_roc_data': test_roc_data
        }

    def plot_training_curves(self, history, model_name):
        """Plot training curves"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'])
        ax1.set_title(f'{model_name} - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot validation accuracy
        ax2.plot(epochs, history['val_acc'])
        ax2.set_title(f'{model_name} - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Plot ROC AUC
        ax3.plot(epochs, history['val_roc_auc'])
        ax3.set_title(f'{model_name} - Validation ROC AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ROC AUC')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_training.png'))
        plt.close()

    def test_model_performance(self):
        """Test and compare model performance"""
        results = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining and evaluating {name}...")
            results[name] = self.train_model(model, name)
            
            # Plot ROC curves
            plot_roc_curves(
                results[name]['test_roc_data'],
                name,
                os.path.join(self.output_dir, f'{name.lower()}_roc.png')
            )
        
        # Print comparative results
        print("\nModel Performance Comparison:")
        print("=" * 100)
        print(f"{'Model':<10} {'Time(s)':<10} {'Epochs':<8} {'Val Acc':<10} "
              f"{'Test Acc':<10} {'Test F1':<10} {'Test ROC':<10}")
        print("-" * 100)
        
        for name, result in results.items():
            print(f"{name:<10} "
                  f"{result['training_time']:<10.2f} "
                  f"{result['epochs']:<8} "
                  f"{result['best_val_acc']:<10.4f} "
                  f"{result['test_acc']:<10.4f} "
                  f"{result['test_f1']:<10.4f} "
                  f"{result['test_roc_auc']:<10.4f}")
        
        # Save results to file
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write("Model Performance Results\n")
            f.write("=" * 100 + "\n")
            for name, result in results.items():
                f.write(f"\n{name} Results:\n")
                for metric, value in result.items():
                    if metric not in ['training_history', 'test_roc_data']:
                        f.write(f"{metric}: {value}\n")
        
        # Assert performance requirements with adjusted thresholds
        for name, result in results.items():
            with self.subTest(model=name):
                self.assertGreater(result['test_acc'], 0.65,  # Adjusted threshold
                                 f"{name} accuracy below minimum threshold")
                self.assertGreater(result['test_f1'], 0.65,   # Adjusted threshold
                                 f"{name} F1-score below minimum threshold")
                self.assertGreater(result['test_roc_auc'], 0.65,  # Adjusted threshold
                                 f"{name} ROC AUC below minimum threshold")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        pass

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelTester)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()