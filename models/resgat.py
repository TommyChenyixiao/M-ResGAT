import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, ModuleList, LayerNorm

class ResGATLayer(torch.nn.Module):
    """Residual Graph Attention Layer with Layer Normalization.
    
    This layer extends the standard GAT by incorporating:
    1. Residual connections for improved gradient flow
    2. Layer normalization for training stability
    3. Flexible input/output dimension handling
    
    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        heads (int, optional): Number of attention heads. Defaults to 8
        dropout (float, optional): Dropout probability. Defaults to 0.6
        residual (bool, optional): Whether to use residual connection. Defaults to True
        
    Attributes:
        gat (GATConv): Graph attention layer
        norm (LayerNorm): Layer normalization
        res_linear (Linear): Optional transform for residual when dimensions differ
        
    Note:
        - Output channels are divided by number of heads to maintain final dimension
        - Residual connections adapt to different input/output dimensions if needed
        - Layer norm is applied after residual connection
        - Uses concatenation of attention heads
    """
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
    """Residual Graph Attention Network with Layer Normalization.
    
    This model stacks multiple ResGATLayers to create a deep architecture that 
    leverages both attention mechanisms and residual connections. The architecture 
    consists of an input transformation layer, multiple residual GAT layers, and 
    a final prediction layer.
    
    Args:
        num_features (int): Number of input features
        hidden_channels (int): Number of hidden features
        num_classes (int): Number of output classes
        num_layers (int, optional): Number of model layers. Defaults to 3
        heads (int, optional): Number of attention heads. Defaults to 8
        dropout (float, optional): Dropout probability. Defaults to 0.6
        residual (bool, optional): Whether to use residual connections. Defaults to True
        
    Attributes:
        input_layer (ResGATLayer): Initial feature transformation layer
        layers (ModuleList): List of hidden ResGATLayers
        output_layer (GATConv): Final prediction layer
        dropout (float): Dropout probability
        
    Note:
        - Each layer combines GAT with residual connections and normalization
        - Hidden layers maintain constant dimension
        - Each layer is followed by ELU activation and dropout
        - Output layer uses single-head attention without residual connection
    """
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
