
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn\

import torch_geometric.transforms as T


from torch_geometric.data import HeteroData
import json
import random


# convert graph from json format to pytorch HeteroData format
def get_graph_data(graph):
    data = HeteroData()
#data中加入空节点特征初始化：初始化各种关系的空边索引：
    data["add_node"].x = torch.zeros((0, 64), dtype=torch.long)
    data["del_node"].x = torch.zeros((0, 64), dtype=torch.long)

    data["add_node"].token_ids = torch.zeros((0, 64), dtype=torch.long)
    data["del_node"].token_ids = torch.zeros((0, 64), dtype=torch.long)

    data["del_node", "line_mapping", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["del_node", "cdfg", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["del_node", "ref", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )

    data["add_node", "line_mapping", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["add_node", "cdfg", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["add_node", "ref", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )

    for node in graph:
        if node["isDel"]:
            data["del_node"].x = torch.cat(
                (data["del_node"].x, torch.tensor([node["token_ids"][:64]])), 0
            )
            data["del_node"].token_ids = torch.cat(
                (data["del_node"].token_ids, torch.tensor([node["token_ids"][:64]])), 0
            )
            for n in node["cfgs"]:
                #[node["nodeIndex"], n]边
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )#view(1, -1)：这部分将张量重塑为 (1, 2) 的形状。
                data["del_node", "cdfg", "del_node"].edge_index = torch.cat(
                    (data["del_node", "cdfg", "del_node"].edge_index, edge), 0
                )
                #张量沿第一个维度（0）连接0是参数

            for n in node["dfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "cdfg", "del_node"].edge_index = torch.cat(
                    (data["del_node", "cdfg", "del_node"].edge_index, edge), 0
                )

            for n in node["fieldParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "ref", "del_node"].edge_index = torch.cat(
                    (data["del_node", "ref", "del_node"].edge_index, edge), 0
                )

            for n in node["methodParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "ref", "del_node"].edge_index = torch.cat(
                    (data["del_node", "ref", "del_node"].edge_index, edge), 0
                )

            if node["lineMapIndex"] != -1:
                edge = torch.tensor(
                    [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
                ).view(1, -1)
                data["del_node", "line_mapping", "add_node"].edge_index = torch.cat(
                    (data["del_node", "line_mapping", "add_node"].edge_index, edge), 0
                )

        else:
            data["add_node"].x = torch.cat(
                (data["add_node"].x, torch.tensor([node["token_ids"][:64]])), 0
            )
            data["add_node"].token_ids = torch.cat(
                (data["add_node"].token_ids, torch.tensor([node["token_ids"][:64]])), 0
            )
            for n in node["cfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data[
                    "add_node", "cdfg", "add_node"
                ].edge_index = torch.cat(
                    (data["add_node", "cdfg", "add_node"].edge_index, edge), 0
                )

            for n in node["dfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "cdfg", "add_node"].edge_index = torch.cat(
                    (data["add_node", "cdfg", "add_node"].edge_index, edge), 0
                )

            for n in node["fieldParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "ref", "add_node"].edge_index = torch.cat(
                    (data["add_node", "ref", "add_node"].edge_index, edge), 0
                )

            for n in node["methodParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "ref", "add_node"].edge_index = torch.cat(
                    (data["add_node", "ref", "add_node"].edge_index, edge), 0
                )

            if node["lineMapIndex"] != -1:
                edge = torch.tensor(
                    [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
                ).view(1, -1)
                data["add_node", "line_mapping", "del_node"].edge_index = torch.cat(
                    (data["add_node", "line_mapping", "del_node"].edge_index, edge), 0
                )

    data["del_node", "line_mapping", "add_node"].edge_index = (
        data["del_node", "line_mapping", "add_node"].edge_index.t().contiguous()
    )
    data["del_node", "cdfg", "del_node"].edge_index = (
        data["del_node", "cdfg", "del_node"].edge_index.t().contiguous()
    )
    data["del_node", "ref", "del_node"].edge_index = (
        data["del_node", "ref", "del_node"].edge_index.t().contiguous()
    )

    data["add_node", "line_mapping", "del_node"].edge_index = (
        data["add_node", "line_mapping", "del_node"].edge_index.t().contiguous()
    )
    data["add_node", "cdfg", "add_node"].edge_index = (
        data["add_node", "cdfg", "add_node"].edge_index.t().contiguous()
    )
    data["add_node", "ref", "add_node"].edge_index = (
        data["add_node", "ref", "add_node"].edge_index.t().contiguous()
    )
    return data


#
# from torch_geometric.data import HeteroData
# import torch
#
# def get_graph_data(graph):
#     data = HeteroData()
#     # Initialize node features for 'add' and 'del' nodes
#     data['add_node'].x = torch.zeros((0, 64), dtype=torch.long)
#     data['del_node'].x = torch.zeros((0, 64), dtype=torch.long)
#
#     # Initialize token_ids for 'add' and 'del' nodes
#     data['add_node'].token_ids = torch.zeros((0, 64), dtype=torch.long)
#     data['del_node'].token_ids = torch.zeros((0, 64), dtype=torch.long)
#
#     # Initialize different types of edges
#     data['del_node', 'cfg', 'del_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['del_node', 'dfg', 'del_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['del_node', 'field_access', 'del_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['del_node', 'method_call', 'del_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#
#     data['add_node', 'cfg', 'add_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['add_node', 'dfg', 'add_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['add_node', 'field_access', 'add_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data['add_node', 'method_call', 'add_node'].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data["add_node", "line_mapping", "del_node"].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     data["del_node", "line_mapping", "add_node"].edge_index = torch.zeros((0, 2), dtype=torch.long)
#     # Example for processing nodes and edges (simplified for brevity)
#     for node in graph:
#         node_type = 'add_node' if not node['isDel'] else 'del_node'
#         data[node_type].x = torch.cat((data[node_type].x, torch.tensor([node['token_ids'][:64]])), 0)
#         data[node_type].token_ids = torch.cat((data[node_type].token_ids, torch.tensor([node['token_ids'][:64]])), 0)
#         if node_type == 'del_node':
#             if node["lineMapIndex"] != -1:
#                 edge = torch.tensor(
#                     [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
#                 ).view(1, -1)
#                 data["del_node", "line_mapping", "add_node"].edge_index = torch.cat(
#                     (data["del_node", "line_mapping", "add_node"].edge_index, edge), 0
#                 )
#
#         else:
#             if node["lineMapIndex"] != -1:
#                 edge = torch.tensor(
#                     [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
#                 ).view(1, -1)
#                 data["add_node", "line_mapping", "del_node"].edge_index = torch.cat(
#                     (data["add_node", "line_mapping", "del_node"].edge_index, edge), 0
#                 )
#         # Process CFG edges
#         for n in node['cfgs']:
#             edge = torch.tensor([[node['nodeIndex'], n]], dtype=torch.long).view(
#                     1, -1
#                 )
#             data[node_type, 'cfg', node_type].edge_index = torch.cat(
#                 (data[node_type, 'cfg', node_type].edge_index, edge), 0)
#
#         # Process DFG edges
#         for n in node['dfgs']:
#             edge = torch.tensor([[node['nodeIndex'], n]], dtype=torch.long).view(
#                     1, -1
#                 )
#             data[node_type, 'dfg', node_type].edge_index = torch.cat(
#                 (data[node_type, 'dfg', node_type].edge_index, edge), 0)
#
#         # Process field access edges
#         for n in node['fieldParents']:
#             edge = torch.tensor([[node['nodeIndex'], n]], dtype=torch.long).view(
#                     1, -1
#                 )
#             data[node_type, 'field_access', node_type].edge_index = torch.cat(
#                 (data[node_type, 'field_access', node_type].edge_index, edge), 0)
#
#         # Process method call edges
#         for n in node['methodParents']:
#             edge = torch.tensor([[node['nodeIndex'], n]], dtype=torch.long).view(
#                     1, -1
#                 )
#             data[node_type, 'method_call', node_type].edge_index = torch.cat(
#                 (data[node_type, 'method_call', node_type].edge_index, edge), 0)
#
#     # Convert all edge indices to contiguous for better performance
#     for key in data.edge_types:
#         data[key].edge_index = (
#             data[key].edge_index.t().contiguous()
#         )
#
#     return data