import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

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
        else:
            r = torch.sigmoid(self.br)
            z = torch.sigmoid(self.bz)
        h_tilde = torch.tanh(torch.matmul(x, self.Wh.T) + torch.matmul(r * h, self.Uh.T) + self.bh)
        h_next = (1 - z) * h + z * h_tilde
        return h_next

class HGTConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        rnn_type='gru',
        use_skip=False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        self.W_lin = HeteroDictLinear(in_channels,
                                      out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.use_skip = use_skip
        self.rnn_type = rnn_type
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.norm = nn.LayerNorm(out_channels)
        #为每种边做个编号{('author', 'to', 'paper'): 0, ('paper', 'to', 'author'): 1, ('paper', 'to', 'term'): 2, ('paper', 'to', 'conference'): 3, ('term', 'to', 'paper'): 4, ('conference', 'to', 'paper'): 5}
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }
        #集合存目标节点的类型
        self.dst_node_types = set([key[-1] for key in self.edge_types])
        #kqv放在一起，HeteroDictLinear会对于不同节点类型采用独立的kqv，
        self.negative_slope = 0.01
        dim = out_channels // heads
        num_types = heads * len(self.edge_types)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * dim))
        #self.k_rel和self.v_rel对于每个边的每个头都有一个独立的参数，有num_types个
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                 is_sorted=True)
        #对于每种边采用不同的跳跃连接数
        if use_skip:
            self.skip = ParameterDict({
                 node_type: Parameter(torch.empty(1))
                 for node_type in self.node_types
             })
        #p_rel先验权重
        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))
        if rnn_type == 'gru':
            self.rnn = nn.GRUCell(out_channels, out_channels)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_channels, out_channels, 1, batch_first=True)
        else:
            self.rnn = VarGru(out_channels, out_channels)
        self.attn_linear = nn.Linear(2 * dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.W_lin.reset_parameters()
        self.v_rel.reset_parameters()
        self.rnn.reset_parameters()
        self.norm.reset_parameters()
        if self.use_skip:
            ones(self.skip)
        ones(self.p_rel)
        nn.init.xavier_uniform_(self.attn_linear.weight)
    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}#计算每个key（节点类型）的偏移量，就是它们的起始索引，用来分离节点。
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        #将x_dict的所有节点张量合并成一个大张量，如author（2048,2，768）,paper(200,2,768)等等，合成为（2248，2，768），用offset区别不同类型节点
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            #编码[2,N],[[edge_type_offset],[num_edge_types+edge_type_offset]]]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
               1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])
        type_vec = torch.cat(type_list, dim=1).flatten()
        #ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        #k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        k = torch.cat(ks, dim=0)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)


        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:

        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # 算出 K, Q, V over node types，并用多头:
        W_dict = self.W_lin(x_dict)
        for key, val in W_dict.items():
            k, q, v = val,val,val
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

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:

        # Iterate over node types:
        for node_type, out in out_dict.items():
            if out.size(-1) == x_dict[node_type].size(-1):
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
                        lstm_out, (hn, cn)=self.rnn(x_input)
                        out = self.norm(lstm_out[:, -1, :])
                    else:
                        out = self.norm(self.rnn(x_dict[node_type], out, self.rnn_type))
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (torch.cat([k_j, q_i], dim=-1) * self.att).sum(dim=-1) * edge_attr
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
