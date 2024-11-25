import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros

class MotifGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        beta: float = 0.5,
        bias: bool = True,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.residual = residual

        # Linear transformation for node features
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        # Residual connection
        if residual:
            self.res_linear = Linear(in_channels, heads * out_channels, bias=False) \
                if in_channels != heads * out_channels else None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)
        if self.residual and self.res_linear is not None:
            glorot(self.res_linear.weight)

    def compute_motif_adj(self, edge_index, num_nodes):
        """Compute motif-based adjacency matrix."""
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        motif_adj = torch.matrix_power(adj, 3)  # triangle motifs
        motif_adj = motif_adj / (motif_adj.sum(dim=-1, keepdim=True).clamp(min=1))
        return motif_adj

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Store residual
        residual = x
        
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Add self-loops to edge_index
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Compute attention
        # First-order attention
        alpha_first = self._compute_attention(x, edge_index)
        
        # Motif-based attention
        motif_adj = self.compute_motif_adj(edge_index, num_nodes)
        alpha_motif = self._compute_motif_attention(x, edge_index, motif_adj)
        
        # Combine attentions using beta
        alpha = self.beta * alpha_first + (1 - self.beta) * alpha_motif
        
        # Apply attention dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Propagate messages
        out = self.propagate(edge_index, x=x, alpha=alpha)

        # Concatenate or average multi-head attention results
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Add residual connection
        if self.residual:
            if self.res_linear is not None:
                residual = self.res_linear(residual)
            out = out + residual

        if self.bias is not None:
            out += self.bias

        return out

    def _compute_attention(self, x, edge_index):
        """Compute first-order attention coefficients."""
        x_j = x[edge_index[0]]  # Source node features
        x_i = x[edge_index[1]]  # Target node features

        # Compute attention coefficients
        alpha = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])
        
        return alpha

    def _compute_motif_attention(self, x, edge_index, motif_adj):
        """Compute motif-based attention coefficients."""
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]

        # Compute attention including motif information
        alpha = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)
        alpha = alpha * motif_adj[edge_index[0], edge_index[1]].unsqueeze(1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])
        
        return alpha

    def message(self, x_j, alpha):
        """Message function."""
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class MGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        beta: float = 0.5,
        residual: bool = True
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(
            MotifGATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                beta=beta,
                residual=False  # No residual for first layer
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                MotifGATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    beta=beta,
                    residual=residual
                )
            )
        
        # Output layer
        self.convs.append(
            MotifGATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                beta=beta,
                residual=residual
            )
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x