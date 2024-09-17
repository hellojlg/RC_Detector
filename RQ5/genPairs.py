from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
import json
import random

from genPyG import get_graph_data


class miniGraph:
    def __init__(self, g, pyg, fDir):
        self.g = g
        self.pyg = pyg
        self.fDir = fDir
        self.score = 0.0


# get all train pairs
def get_all_pairs(all_data_map, max_cnt=1000):
    all_pairs = []
    for key, graphs in all_data_map.items():
        cnt = 0
        rootcnt = 0
        #graphs是一个test的内容,一个test下有许多图，graph是一个图，
        graphs1 = []
        #对此test下的图，第一个节点是rootcause的图放在前面
        for graph in graphs:
            if graph[0]["rootcause"]:
                graphs1.append(graph)
                rootcnt += 1

        for graph in graphs:
            if not graph[0]["rootcause"]:#不是gen
                graphs1.append(graph)
#处理graph中的节点
        for i in range(len(graphs1)):
            for j in range(i + 1, len(graphs1)):
                pyg1 = get_graph_data(graphs1[i])
                pyg2 = get_graph_data(graphs1[j])
                #graphs1[i], pyg1, key,key为test数,graphs1[i]当前提交的信息，pyg1为当前提交图，比较那次提交是最rootcause
                minig1 = miniGraph(graphs1[i], pyg1, key)
                minig2 = miniGraph(graphs1[j], pyg2, key)
                if graphs1[i][0]["rootcause"] == graphs1[j][0]["rootcause"]:
                    #将minig1和minig2配对，加个概率0，5
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 0.5})
                    cnt = cnt + 1
                    #rootcause不同的情况
                if graphs1[i][0]["rootcause"] and not graphs1[j][0]["rootcause"]:
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 1.0})
                    cnt = cnt + 1
                if not graphs1[i][0]["rootcause"] and graphs1[j][0]["rootcause"]:
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 0.0})
                    cnt = cnt + 1
                if cnt > max_cnt:
                    break
            if i == rootcnt:
                break
            if cnt > max_cnt:
                break
    return all_pairs


def get_dir_to_minigraphs(all_data_map):
    dir_to_minigraphs = {}
    for key, graphs in all_data_map.items():
        dir_to_minigraphs[key] = []
        for i in range(len(graphs)):
            pyg = get_graph_data(graphs[i])
            minig = miniGraph(graphs[i], pyg, key)
            dir_to_minigraphs[key].append(minig)
    return dir_to_minigraphs
