import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create a ModuleList to store all convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer: from input features to hidden channels
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Middle layers: hidden to hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer: from hidden channels to num_classes
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        # Process through all layers except the last one
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer without ReLU and dropout
        x = self.convs[-1](x, edge_index)
        return x