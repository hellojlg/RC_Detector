import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from Transformer_implementation import Transformer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index
import torch
import torch.nn as nn
import torch.nn.functional as F
def transformer_interface(h_t_minus_1, h_t, transformer):
    h_sequence = torch.stack([h_t_minus_1, h_t], dim=1)

    src_mask = None
    output = transformer(h_sequence, src_mask)

    h_t_plus_1 = output[:, -1, :]
    return h_t_plus_1

class VarGru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VarGru, self).__init__()
        self.hidden_size = hidden_size
        self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        self.Wh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.zeros_(weight)
            else:
                nn.init.xavier_uniform_(weight)

    # zt = σ(Uz ht−1 + bz) rt = σ(Ur ht−1 + br)
    # zt = σ(Uz ht−1)  rt = σ(Ur ht−1)
    # zt = σ(bz)  rt = σ(br)
    def forward(self, x, h, type):
        if type == 'gru1':
            r = torch.sigmoid(torch.matmul(h, self.Ur) + self.br)
            z = torch.sigmoid(torch.matmul(h, self.Uz) + self.bz)
        elif type == 'gru2':
            r = torch.sigmoid(torch.matmul(h, self.Ur))
            z = torch.sigmoid(torch.matmul(h, self.Uz))
        elif type == 'gru3':
            r = torch.sigmoid(self.br)
            z = torch.sigmoid(self.bz)
        else:
            raise ValueError(f"Unsupported type '{type}'")
        h_tilde = torch.tanh(torch.matmul(x, self.Wh.T) + torch.matmul(r * h, self.Uh.T) + self.bh)
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class HGTConv(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            heads: int = 8,
            rnn_type='gru1',
            use_skip=False,
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.use_skip = use_skip
        self.rnn_type = rnn_type
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.norm = nn.LayerNorm(out_channels)
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }
        self.dst_node_types = set([key[-1] for key in self.edge_types])

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)
        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        if use_skip:
            self.skip = ParameterDict({
                node_type: Parameter(torch.empty(1))
                for node_type in self.node_types
            })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        if rnn_type == 'gru':
            self.rnn = nn.GRUCell(out_channels, out_channels)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_channels, out_channels, 1, batch_first=True)
        elif rnn_type == 'transformer':
            self.transformer = Transformer(self.out_channels, self.heads, 2048, 1, 0.1)
        else:
            self.rnn = VarGru(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        self.norm.reset_parameters()
        if self.use_skip:
            ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
            self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N


            edge_type_offset = self.edge_types_map[edge_type]

            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])
        type_vec = torch.cat(type_list, dim=1).flatten()
        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]
    ) -> Dict[NodeType, Optional[Tensor]]:

        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)


        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]


        for node_type, out in out_dict.items():
            if out.size(-1) == x_dict[node_type].size(-1):
                if out.size(0) == 0:
                    continue
                if self.use_skip:
                    alpha = self.skip[node_type].sigmoid()
                    out = alpha * out + (1 - alpha) * x_dict[node_type]
                else:
                    if self.rnn_type == 'gru':
                        out = self.norm(self.rnn(x_dict[node_type], out))
                    elif self.rnn_type == 'lstm':
                        node_embeddings = x_dict[node_type].unsqueeze(1)  # (n, 1, m)
                        neighbor_embeddings = out.unsqueeze(1)  # (n, 1, m)
                        x_input = torch.cat((node_embeddings, neighbor_embeddings), dim=1)
                        lstm_out, _ = self.rnn(x_input)
                        out = self.norm(lstm_out[:, -1, :])
                    elif self.rnn_type == 'transformer':
                        out = self.norm(transformer_interface(x_dict[node_type], out, self.transformer))
                    else:
                        out = self.norm(self.rnn(x_dict[node_type], out, self.rnn_type))

            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
