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


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rnn_type = rnn_type
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.norm = nn.LayerNorm(out_channels)

        if rnn_type == 'gru':
            self.rnn = nn.GRUCell(out_channels, out_channels)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_channels, out_channels, 1, batch_first=True)
        else:
            self.rnn = VarGru(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.rnn.reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:

        out_dict = x_dict
        for node_type, out in out_dict.items():
            if out.size(-1) == x_dict[node_type].size(-1):
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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
