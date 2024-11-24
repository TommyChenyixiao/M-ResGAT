import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, 
                 heads=8, dropout=0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout
        ))
        
        # Hidden layers
        hidden_in_channels = hidden_channels * heads  # Account for concatenation
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                in_channels=hidden_in_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout
            ))
        
        # Output layer (no concatenation)
        self.convs.append(GATConv(
            in_channels=hidden_in_channels,
            out_channels=num_classes,
            heads=1,
            concat=False,
            dropout=dropout
        ))
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        # Initial dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process through hidden layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x