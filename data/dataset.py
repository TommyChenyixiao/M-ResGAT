from torch_geometric.datasets import CitationFull, Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch

class DatasetHandler:
    AVAILABLE_DATASETS = {
        'Cora_ML': CitationFull,
        'Cora': CitationFull,
        'CiteSeer': CitationFull,
        'PubMed': CitationFull,
    }

    def __init__(self, dataset_name, root='/tmp/CitationFull'):
        self.dataset_name = dataset_name
        self.root = root
        self.dataset = self._load_dataset()
        self.data = self.dataset[0]
        
    def _load_dataset(self):
        """Load the specified dataset with appropriate loader"""
        if self.dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {self.dataset_name} not supported. "
                           f"Available datasets: {list(self.AVAILABLE_DATASETS.keys())}")
            
        dataset_class = self.AVAILABLE_DATASETS[self.dataset_name]
        
        if dataset_class == CitationFull:
            return dataset_class(
                root=self.root,
                name=self.dataset_name,
                transform=NormalizeFeatures()
            )
        else:
            return dataset_class(
                root=self.root,
                name=self.dataset_name,
                transform=NormalizeFeatures()
            )
        
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics"""
        num_nodes = self.data.num_nodes
        num_edges = self.data.num_edges // 2  # Divide by 2 for undirected graphs
        num_features = self.dataset.num_features
        num_classes = len(torch.unique(self.data.y))
        
        # Calculate class distribution
        class_distribution = {}
        for c in range(num_classes):
            count = (self.data.y == c).sum().item()
            class_distribution[f'class_{c}'] = {
                'count': count,
                'percentage': count/num_nodes * 100
            }
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'num_classes': num_classes,
            'class_distribution': class_distribution,
            'avg_degree': (2 * num_edges) / num_nodes,
            'feature_dim': self.data.x.size(1)
        }
    
    def get_data(self):
        """Get the processed PyG data object"""
        return self.data

    def print_stats(self):
        """Print detailed dataset statistics"""
        stats = self.get_dataset_stats()
        print(f"\n{self.dataset_name} Dataset Statistics:")
        print("=" * 50)
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Features: {stats['num_features']}")
        print(f"Classes: {stats['num_classes']}")
        print(f"Average Degree: {stats['avg_degree']:.2f}")
        
        print("\nClass Distribution:")
        for class_id, info in stats['class_distribution'].items():
            print(f"{class_id}: {info['count']} nodes ({info['percentage']:.2f}%)")