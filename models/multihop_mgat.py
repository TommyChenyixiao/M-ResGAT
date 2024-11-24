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

        # Bias parameter
        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        for lin in self.lins:
            glorot(lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        self.hop_attention.data.fill_(1.0)  # Initialize hop attention
        zeros(self.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        out = 0

        # Iterate over hops
        hop_weights = F.softmax(self.hop_attention, dim=0)
        for hop in range(self.num_hops):
            hop_edge_index = self._get_hop_edges(edge_index, hop, num_nodes)
            
            if self.add_self_loops:
                hop_edge_index, _ = remove_self_loops(hop_edge_index)
                hop_edge_index, _ = add_self_loops(hop_edge_index, num_nodes=num_nodes)
            
            # Linear transformation for hop
            hop_x = self.lins[hop](x).view(-1, self.heads, self.out_channels)

            # Compute attention coefficients
            alpha = self._compute_attention(hop_x, hop_edge_index, hop)

            # Propagate messages
            hop_out = self.propagate(hop_edge_index, x=hop_x, alpha=alpha)
            out += hop_weights[hop] * hop_out

        # Concatenate or average the heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def _get_hop_edges(self, edge_index, hop, num_nodes):
        """Efficiently compute k-hop edges."""
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=edge_index.device), 
                                      (num_nodes, num_nodes))
        hop_adj = torch.matrix_power(adj.to_dense(), hop + 1)  # Convert to dense only for small hops
        hop_edges = hop_adj.nonzero(as_tuple=False).t()
        return hop_edges

    def _compute_attention(self, x, edge_index, hop_idx):
        """Compute attention coefficients."""
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]

        # Attention computation
        alpha = (x_j * self.att_src[hop_idx]).sum(-1) + (x_i * self.att_dst[hop_idx]).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])

        return F.dropout(alpha, p=self.dropout, training=self.training)

    def message(self, x_j, alpha):
        """Message passing."""
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
        num_hops: int = 2
    ):
        super().__init__()

        self.convs = ModuleList()

        # Input layer
        self.convs.append(
            MultiHopMotifGATConv(
                in_channels,
                hidden_channels,
                num_hops=num_hops,
                heads=heads,
                dropout=dropout,
                beta=beta
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
                    beta=beta
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
                beta=beta
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
