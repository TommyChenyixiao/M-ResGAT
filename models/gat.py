import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, 
                 heads=8, dropout=0.6):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create a ModuleList to store all convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer: from input features to hidden channels
        self.convs.append(GATConv(num_features, hidden_channels, 
                                 heads=heads, dropout=dropout))
        
        # Middle layers: hidden to hidden (note the multiplied input dim due to concatenation)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                    heads=heads, dropout=dropout))
            
        # Last layer: from hidden channels to num_classes (no concatenation)
        self.convs.append(GATConv(hidden_channels * heads, num_classes,
                                 heads=1, concat=False, dropout=dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        # Initial dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process through all layers except the last one
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer without ELU
        x = self.convs[-1](x, edge_index)
        return x