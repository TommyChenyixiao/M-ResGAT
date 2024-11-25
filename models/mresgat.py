import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, ModuleList, LayerNorm
from torch_geometric.utils import add_self_loops, remove_self_loops

class MultiHopResGATLayer(torch.nn.Module):
    """Multi-hop Graph Attention Layer with Residual Connections.
    
    This layer extends GAT by incorporating:
    1. Multi-hop attention aggregation
    2. Residual connections with layer normalization
    3. Flexible combination strategies for multi-hop outputs
    
    The layer processes the input graph at different hop distances and combines the 
    results using either concatenation, summation, or learned attention weights.
    
    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        heads (int, optional): Number of attention heads. Defaults to 8
        dropout (float, optional): Dropout probability. Defaults to 0.6
        residual (bool, optional): Whether to use residual connection. Defaults to True
        num_hops (int, optional): Number of hops to process. Defaults to 2
        combine (str, optional): How to combine multi-hop outputs ['concat', 'sum', 'attention']. 
            Defaults to 'concat'
            
    Attributes:
        gat_layers (ModuleList): GAT layers for each hop
        norm (LayerNorm): Layer normalization
        attention (Linear): Attention weights for combining hops (if combine='attention')
        res_linear (Linear): Transform for residual connection if dimensions differ
        
    Note:
        - For 'concat' combination, out_channels is split evenly across hops
        - Each hop uses independent GAT parameters
        - Residual connections use layer normalization for stability
        - Higher hops are computed by iteratively traversing the graph
    """
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, residual=True, 
                 num_hops=2, combine='concat'):
        super().__init__()
        assert combine in ['concat', 'sum', 'attention']
        self.combine = combine
        self.num_hops = num_hops
        
        if combine == 'concat':
            self.per_hop_out = out_channels // num_hops
        else:
            self.per_hop_out = out_channels
            
        # Create GAT layers for each hop
        self.gat_layers = ModuleList([
            GATConv(
                in_channels, 
                self.per_hop_out // heads,
                heads=heads,
                dropout=dropout,
                concat=True
            ) for _ in range(num_hops)
        ])
        
        self.norm = LayerNorm(out_channels)
        self.residual = residual
        
        if combine == 'attention':
            self.attention = Linear(self.per_hop_out, 1)
            
        if in_channels != out_channels and residual:
            self.res_linear = Linear(in_channels, out_channels)
        else:
            self.res_linear = None

    def reset_parameters(self):
        for gat in self.gat_layers:
            gat.reset_parameters()
        self.norm.reset_parameters()
        if self.res_linear is not None:
            self.res_linear.reset_parameters()
        if self.combine == 'attention':
            self.attention.reset_parameters()

    def forward(self, x, edge_index):
        # Compute multi-hop attention
        hop_outputs = []
        curr_edge_index = edge_index
        
        for i in range(self.num_hops):
            # Apply GAT for current hop
            hop_out = self.gat_layers[i](x, curr_edge_index)
            hop_outputs.append(hop_out)
            
            # Prepare edges for next hop
            if i < self.num_hops - 1:
                curr_edge_index = self._get_next_hop_edges(curr_edge_index)
        
        # Combine hop outputs
        if self.combine == 'concat':
            out = torch.cat(hop_outputs, dim=-1)
        elif self.combine == 'sum':
            out = sum(hop_outputs)
        else:  # attention
            # Calculate attention weights for each hop
            attn_weights = [F.softmax(self.attention(ho), dim=-1) for ho in hop_outputs]
            out = sum(w * ho for w, ho in zip(attn_weights, hop_outputs))
        
        # Residual connection
        if self.residual:
            if self.res_linear is not None:
                x = self.res_linear(x)
            out = out + x
        
        # Layer normalization
        out = self.norm(out)
        
        return out

    def _get_next_hop_edges(self, edge_index):
        """Compute edges for the next hop"""
        # Remove self-loops before computing next hop
        edge_index, _ = remove_self_loops(edge_index)
        
        # Compute next hop connections
        row, col = edge_index
        next_row = row[col]
        next_col = col[col]
        next_edge_index = torch.stack([next_row, next_col], dim=0)
        
        # Add self-loops back
        next_edge_index, _ = add_self_loops(next_edge_index)
        
        return next_edge_index

class MultiHopResGAT(torch.nn.Module):
    """Multi-hop Residual Graph Attention Network.
    
    This model stacks multiple MultiHopResGATLayers to create a deep architecture that:
    1. Processes the graph at multiple hop distances
    2. Uses residual connections and layer normalization
    3. Combines multi-hop information flexibly
    
    The architecture consists of an input layer, multiple hidden layers with 
    multi-hop processing, and a standard GAT output layer.
    
    Args:
        num_features (int): Number of input features
        hidden_channels (int): Number of hidden features
        num_classes (int): Number of output classes
        num_layers (int, optional): Number of model layers. Defaults to 3
        heads (int, optional): Number of attention heads. Defaults to 8
        dropout (float, optional): Dropout probability. Defaults to 0.6
        residual (bool, optional): Whether to use residual connections. Defaults to True
        num_hops (int, optional): Number of hops per layer. Defaults to 2
        combine (str, optional): How to combine multi-hop outputs ['concat', 'sum', 'attention']. 
            Defaults to 'concat'
            
    Attributes:
        input_layer (MultiHopResGATLayer): Initial feature transformation layer
        layers (ModuleList): List of hidden MultiHopResGATLayers
        output_layer (GATConv): Final prediction layer
        
    Note:
        - Input and hidden layers use multi-hop processing
        - Each layer is followed by ELU activation and dropout
        - Output layer uses standard single-hop GAT
        - All hidden layers maintain the same dimension
    """
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, 
                 heads=8, dropout=0.6, residual=True, num_hops=2, combine='concat'):
        super().__init__()
        self.dropout = dropout
        
        # Input layer
        self.input_layer = MultiHopResGATLayer(
            num_features, 
            hidden_channels,
            heads=heads,
            dropout=dropout,
            residual=residual,
            num_hops=num_hops,
            combine=combine
        )
        
        # Hidden layers
        self.layers = ModuleList([
            MultiHopResGATLayer(
                hidden_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
                num_hops=num_hops,
                combine=combine
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