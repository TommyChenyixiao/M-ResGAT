from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import NormalizeFeatures
import torch

class CoraDataset:
    def __init__(self, root='/tmp/CitationFull'):
        self.dataset = CitationFull(root=root, name='Cora_ML', 
                                  transform=NormalizeFeatures())
        self.data = self.dataset[0]
        
    def get_dataset_stats(self):
        num_nodes = self.data.num_nodes
        num_edges = self.data.num_edges // 2
        num_features = self.dataset.num_features
        num_classes = len(torch.unique(self.data.y))
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'num_classes': num_classes
        }
    
    def get_data(self):
        return self.data