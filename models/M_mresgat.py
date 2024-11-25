import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros

class MultiHopMotifGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_hops: int = 2,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        beta: float = 0.5,
        residual: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.residual = residual

        # Linear transformations for each hop
        self.lins = ModuleList([
            Linear(in_channels, heads * out_channels, bias=False)
            for _ in range(num_hops)
        ])

        # Attention parameters for each hop
        self.att_src = Parameter(torch.empty(num_hops, heads, out_channels))
        self.att_dst = Parameter(torch.empty(num_hops, heads, out_channels))

        # Hop attention
        self.hop_attention = Parameter(torch.ones(num_hops, requires_grad=True))

        # Residual connection
        if residual:
            self.res_linear = Linear(in_channels, heads * out_channels if concat else out_channels, bias=False) \
                if in_channels != (heads * out_channels if concat else out_channels) else None

        # Bias parameter
        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Layer normalization for residual
        if residual:
            self.layer_norm = torch.nn.LayerNorm(heads * out_channels if concat else out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        for lin in self.lins:
            glorot(lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        self.hop_attention.data.fill_(1.0)
        zeros(self.bias)
        if self.residual and self.res_linear is not None:
            glorot(self.res_linear.weight)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Store residual
        residual = x
        
        out = 0
        motif_adj = None  # Will be computed once and reused

        # Compute hop attention weights
        hop_weights = F.softmax(self.hop_attention, dim=0)

        # Process each hop
        for hop in range(self.num_hops):
            # Get hop-specific edges
            hop_edge_index = self._get_hop_edges(edge_index, hop, num_nodes)
            
            if self.add_self_loops:
                hop_edge_index, _ = remove_self_loops(hop_edge_index)
                hop_edge_index, _ = add_self_loops(hop_edge_index, num_nodes=num_nodes)
            
            # Linear transformation for current hop
            hop_x = self.lins[hop](x).view(-1, self.heads, self.out_channels)

            # Compute motif adjacency matrix (compute once and reuse)
            if motif_adj is None:
                motif_adj = self._compute_motif_adj(edge_index, num_nodes)

            # First-order attention
            alpha_first = self._compute_attention(hop_x, hop_edge_index, hop)
            
            # Motif-based attention
            alpha_motif = self._compute_motif_attention(hop_x, hop_edge_index, motif_adj, hop)
            
            # Combine attentions
            alpha = self.beta * alpha_first + (1 - self.beta) * alpha_motif
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Propagate for current hop
            hop_out = self.propagate(hop_edge_index, x=hop_x, alpha=alpha)
            out = out + hop_weights[hop] * hop_out

        # Concatenate or average the heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Add residual connection
        if self.residual:
            if self.res_linear is not None:
                residual = self.res_linear(residual)
            out = self.layer_norm(out + residual)

        if self.bias is not None:
            out += self.bias

        return out

    def _get_hop_edges(self, edge_index, hop, num_nodes):
        """Efficiently compute k-hop edges."""
        adj = torch.sparse_coo_tensor(
            edge_index, 
            torch.ones(edge_index.size(1), device=edge_index.device),
            (num_nodes, num_nodes)
        )
        hop_adj = torch.matrix_power(adj.to_dense(), hop + 1)
        hop_edges = hop_adj.nonzero(as_tuple=False).t()
        return hop_edges

    def _compute_motif_adj(self, edge_index, num_nodes):
        """Compute motif-based adjacency matrix."""
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        motif_adj = torch.matrix_power(adj, 3)  # triangle motifs
        motif_adj = motif_adj / (motif_adj.sum(dim=-1, keepdim=True).clamp(min=1))
        return motif_adj

    def _compute_attention(self, x, edge_index, hop_idx):
        """Compute first-order attention coefficients."""
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]

        alpha = (x_j * self.att_src[hop_idx]).sum(-1) + \
                (x_i * self.att_dst[hop_idx]).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])
        
        return alpha

    def _compute_motif_attention(self, x, edge_index, motif_adj, hop_idx):
        """Compute motif-based attention coefficients."""
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]

        alpha = (x_j * self.att_src[hop_idx]).sum(-1) + \
                (x_i * self.att_dst[hop_idx]).sum(-1)
        alpha = alpha * motif_adj[edge_index[0], edge_index[1]].unsqueeze(1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])
        
        return alpha

    def message(self, x_j, alpha):
        """Message function."""
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_hops={self.num_hops}, '
                f'heads={self.heads})')

class MultiHopMGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        beta: float = 0.5,
        num_hops: int = 2,
        residual: bool = True
    ):
        super().__init__()

        self.convs = ModuleList()

        # Input layer (no residual)
        self.convs.append(
            MultiHopMotifGATConv(
                in_channels,
                hidden_channels,
                num_hops=num_hops,
                heads=heads,
                dropout=dropout,
                beta=beta,
                residual=False
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                MultiHopMotifGATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    num_hops=num_hops,
                    heads=heads,
                    dropout=dropout,
                    beta=beta,
                    residual=residual
                )
            )

        # Output layer
        self.convs.append(
            MultiHopMotifGATConv(
                hidden_channels * heads,
                out_channels,
                num_hops=num_hops,
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
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x