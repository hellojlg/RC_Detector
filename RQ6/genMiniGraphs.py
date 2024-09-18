
# Use a pipeline as a high-level helper
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

from transformers import RobertaTokenizer, RobertaModel


#读取所有的测试集训练集的数据
def getAllGraph():

    fDirMap = {}
    for i in range(0, 1572):
        testDir = f"test{i}"
        fDir = f"../trainData/test{i}"
        graphPath = f"{fDir}/graph1.json"

        with open(fDir + "/info.json", "r") as f:
            info = json.load(f)

        graph = None
        with open(graphPath, "r") as f:
            graph = json.load(f)
        fDirMap[testDir] = graph
    return fDirMap


#graph当前fDir对应的图，此函数用来建无向边，如某节点的dfgs有e节点,也要将e节点的dfgs添加某节点，来建双向边
def toBidirectional(graph, fDir):
    for node in graph:
        node["fDir"] = fDir
        index = node["nodeIndex"]

        for e in node["cfgs"]:
            if index not in graph[e]["cfgs"]:
                graph[e]["cfgs"].append(index)

        for e in node["dfgs"]:
            if index not in graph[e]["dfgs"]:
                graph[e]["dfgs"].append(index)

        for e in node["fieldParents"]:
            if index not in graph[e]["fieldParents"]:
                graph[e]["fieldParents"].append(index)

        for e in node["methodParents"]:
            if index not in graph[e]["methodParents"]:
                graph[e]["methodParents"].append(index)



def clone(node):
    cnode = {}
    cnode["cfgs"] = [e for e in node["cfgs"]]
    cnode["dfgs"] = [e for e in node["dfgs"]]
    cnode["fieldParents"] = [e for e in node["fieldParents"]]
    cnode["methodParents"] = [e for e in node["methodParents"]]
    cnode["commits"] = [cid for cid in node["commits"]]

    cnode["code"] = node["code"]
    cnode["fName"] = node["fName"]
    cnode["isDel"] = node["isDel"]
    cnode["lineBeg"] = node["lineBeg"]
    cnode["lineEnd"] = node["lineEnd"]
    cnode["lineMapIndex"] = node["lineMapIndex"]
    cnode["nodeIndex"] = node["nodeIndex"]
    cnode["rootcause"] = node["rootcause"]
    cnode["fDir"] = node["fDir"]
    return cnode


#以删除节点为根，建个对这个新图节点数不超过8，只加入节点，应该后续在处理
def dfs(index, depth, graph, newGraph, visited):
    if depth >= 2 or (index in visited) or len(visited) > 8:
        return

    newGraph.append(clone(graph[index]))
    visited.add(index)
    curNode = graph[index]

    for e in curNode["cfgs"][:3]:
        dfs(e, depth + 1, graph, newGraph, visited)

    for e in curNode["dfgs"][:1]:
        dfs(e, depth + 1, graph, newGraph, visited)

    for e in curNode["fieldParents"][:1]:
        dfs(e, depth + 1, graph, newGraph, visited)

    for e in curNode["methodParents"][:1]:
        dfs(e, depth + 1, graph, newGraph, visited)

    if curNode["lineMapIndex"] != -1:
        dfs(curNode["lineMapIndex"], depth + 1, graph, newGraph, visited)


#关键
def adjustIndex(newGraph):
    delIndexMap = {}
    addIndexMap = {}

    delCnt = 0
    addCnt = 0

    for node in newGraph:
        if node["isDel"]:
            delIndexMap[node["nodeIndex"]] = delCnt
            delCnt = delCnt + 1
        else:
            addIndexMap[node["nodeIndex"]] = addCnt
            addCnt = addCnt + 1

    indexMap = None

    for node in newGraph:
        if node["isDel"]:
            indexMap = delIndexMap
        else:
            indexMap = addIndexMap

        tmp = []
        for e in node["cfgs"]:
            if e in indexMap:
                e = indexMap[e]
                tmp.append(e)

        node["cfgs"] = tmp
        tmp = []

        for e in node["dfgs"]:
            if e in indexMap:
                e = indexMap[e]
                tmp.append(e)

        node["dfgs"] = tmp
        tmp = []

        for e in node["fieldParents"]:
            if e in indexMap:
                e = indexMap[e]
                tmp.append(e)

        node["fieldParents"] = tmp
        tmp = []

        for e in node["methodParents"]:
            if e in indexMap:
                e = indexMap[e]
                tmp.append(e)

        node["methodParents"] = tmp
        tmp = []

        if node["lineMapIndex"] != -1 and node["isDel"]:
            if node["lineMapIndex"] in addIndexMap:
                node["lineMapIndex"] = addIndexMap[node["lineMapIndex"]]
            else:
                node["lineMapIndex"] = -1
        elif node["lineMapIndex"] != -1 and not node["isDel"]:
            if node["lineMapIndex"] in delIndexMap:
                node["lineMapIndex"] = delIndexMap[node["lineMapIndex"]]
            else:
                node["lineMapIndex"] = -1

        node["nodeIndex"] = indexMap[node["nodeIndex"]]

    return newGraph


#对每个test对应的图进行处理
def genMiniGraphs(graph, fDir):
    allGraph = []
    toBidirectional(graph, fDir)
    for node in graph:
        if not node["isDel"]:
            continue
        node["fDir"] = fDir#noneed
        indexMap = {}
        newGraph = []
        visited = set()
        #对每个删除节点进行处理，dfs来构建新的图
        dfs(node["nodeIndex"], 0, graph, newGraph, visited)
        #加入调整下标后的新图
        allGraph.append(adjustIndex(newGraph))

    return allGraph


#处理每个test的所有删除节点的所对应的图
def getAllMiniGraphs(fDirMap):
    cnt = 0
    allMiniGraphs = {}
    #fDir实际上是上文中getAllGraph()函数中定义的testDir，fDir, graph是键与值的关系
    for fDir, graph in fDirMap.items():
        miniGraphs = genMiniGraphs(graph, fDir)#每个test可能有很多个删除节点，所以miniGraphs可能包含多个图
        allMiniGraphs[fDir] = miniGraphs
    return allMiniGraphs



def genAllMiniGraphs(model_type):
    fDirMap = getAllGraph()#
    if model_type == "graphcodebert":
        tokenizer = RobertaTokenizer.from_pretrained("../microsoft/graphcodebert-base")
    elif model_type == "codet5":
        tokenizer = RobertaTokenizer.from_pretrained("../microsoft/codet5-base")
    elif model_type == "unixcoder":
        tokenizer = RobertaTokenizer.from_pretrained("../microsoft/unixcoder-base")
    elif model_type == "codebert":
        tokenizer = RobertaTokenizer.from_pretrained("../microsoft/codebert-base")
    else:
        raise ValueError(f"Unsupported model type '{model_type}'.")

    allMiniGraphs = getAllMiniGraphs(fDirMap)
    #遍历test
    for fDir, miniGraphs in allMiniGraphs.items():
        #遍历test中所有的图
        for minig in miniGraphs:
            #遍历所有节点
            for node in minig:
                #为node["code"]编码
                node["token_ids"] = tokenizer.encode_plus(
                    text=node["code"],
                    add_special_tokens=True,
                    max_length=64,
                    padding="max_length",
                )["input_ids"]
                #node["token_ids"] =encode_plus方法返回的字典中提取input_ids键对应的值，编码后的token ID列表
    with open("miniGraphs.json", "w") as f:
        json.dump(allMiniGraphs, f)
