import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, ModuleList, LayerNorm

class ResGATLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, residual=True):
        super().__init__()
        self.gat = GATConv(
            in_channels, 
            out_channels // heads,  # Divide out_channels by heads to maintain dimension
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.norm = LayerNorm(out_channels)
        self.residual = residual
        
        if in_channels != out_channels and residual:
            self.res_linear = Linear(in_channels, out_channels)
        else:
            self.res_linear = None

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.norm.reset_parameters()
        if self.res_linear is not None:
            self.res_linear.reset_parameters()

    def forward(self, x, edge_index):
        # GAT transformation
        out = self.gat(x, edge_index)
        
        # Residual connection
        if self.residual:
            if self.res_linear is not None:
                x = self.res_linear(x)
            out = out + x
            
        # Layer normalization
        out = self.norm(out)
        
        return out

class ResGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, 
                 heads=8, dropout=0.6, residual=True):
        super().__init__()
        self.dropout = dropout
        
        # Input layer
        self.input_layer = ResGATLayer(
            num_features, 
            hidden_channels, 
            heads=heads,
            dropout=dropout,
            residual=residual
        )
        
        # Hidden layers
        self.layers = ModuleList([
            ResGATLayer(
                hidden_channels, 
                hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual
            ) for _ in range(num_layers-2)
        ])
        
        # Output layer
        self.output_layer = GATConv(
            hidden_channels, 
            num_classes, 
            heads=1, 
            concat=False,
            dropout=dropout
        )

    def reset_parameters(self):
        self.input_layer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x, edge_index):
        # Input layer
        x = self.input_layer(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output_layer(x, edge_index)
        
        return x
