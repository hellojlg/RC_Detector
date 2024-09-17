from dataset import *
from torch.nn.utils.rnn import pad_sequence
from settings import *
from read_data import *
from torch.utils.data import DataLoader, Dataset
from model import bilstmModel
import torch
from torch_geometric.data import HeteroData
import json
import random
from itertools import chain
from functools import cmp_to_key
from settings import *


def divide_lst(lst, n, k):
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):
        if cnt < k - 1:
            all_list.append(lst[i : i + n])
        else:
            all_list.append(lst[i:])
            break
        cnt = cnt + 1
    return all_list


def collate_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    y = torch.tensor(y)
    return x_pad, y


def cmp(n1, n2):
    score1 = n1["prob"]
    score2 = n2["prob"]
    if score1 < score2:
        return 1
    elif score1 > score2:
        return -1
    return 0


def eval(dir_to_minigraphs, all_proba_pairs, test_all_nodes):
    for i, proba_pair in enumerate(all_proba_pairs):
        cur_prob = proba_pair[1]
        test_all_nodes[i]["prob"] = cur_prob
        if test_all_fdirs[i] not in dir_to_minigraphs:
            dir_to_minigraphs[test_all_fdirs[i]] = []
        dir_to_minigraphs[test_all_fdirs[i]].append(test_all_nodes[i])

    for fdir in dir_to_minigraphs.keys():
        dir_to_minigraphs[fdir].sort(key=cmp_to_key(cmp))


def eval_recall_topk(fdirs, dir_to_minigraphs, k):
    root_cause_cnt = 0
    for fdir in set(fdirs):
        f = False
        for node in dir_to_minigraphs[fdir][:k]:
            if node["rootcause"]:
                f = True
        if f == True:
            root_cause_cnt = root_cause_cnt + 1
    return root_cause_cnt / len(set(fdirs))


def eval_mean_first_rank(fdirs, dir_to_minigraphs):
    total_rank_cnt = 0
    for fdir in set(fdirs):
        for i, node in enumerate(dir_to_minigraphs[fdir]):
            if node["rootcause"]:
                total_rank_cnt = total_rank_cnt + i + 1
                break

    return total_rank_cnt / len(set(fdirs))


def do_cross_fold():
    data1 = json.load(open("../dataset1.json"))
    data2 = json.load(open("../dataset2.json"))
    data3 = json.load(open("../dataset3.json"))

    all_data = []
    all_data.extend(data1)
    all_data.extend(data2)
    all_data.extend(data3)

    all_data_list = divide_lst(all_data, int(len(all_data) / 10), 10)
    for i in range(len(all_data_list)):
        test_data = all_data_list[i]
        train_data = []
        for j in range(len(all_data_list)):
            if i != j:
                train_data.extend(all_data_list[j])
        (
            train_all_nodes,
            train_all_codes,
            train_all_labels,
            train_all_fdirs,
            train_all_info,
        ) = get_dataset(train_data)
        (
            test_all_nodes,
            test_all_codes,
            test_all_labels,
            test_all_fdirs,
            test_all_info,
        ) = get_dataset(test_data)
        train_data_set = dl_dataset(
            train_all_nodes,
            train_all_codes,
            train_all_labels,
            train_all_fdirs,
            train_all_info,
        )
        test_data_set = dl_dataset(
            test_all_nodes,
            test_all_codes,
            test_all_labels,
            test_all_fdirs,
            test_all_info,
        )

        train_data_loader = DataLoader(
            train_data_set, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        test_data_loader = DataLoader(
            test_data_set, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        bilstm_model = bilstmModel(300, 150, 5, 0.1)
        optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001)
        citerion = torch.nn.CrossEntropyLoss()

        all_info = []
        for epoch in range(0, 100):
            train_predict = []
            bilstm_model.train()
            total_loss = 0
            for x, y in train_data_loader:
                pred = bilstm_model(x)
                loss = citerion(pred, y)

                loss.backward()
                optimizer.step()
                total_loss = total_loss + loss.detach().item()
                train_predict.extend(pred.argmax(dim=1).tolist())

            bilstm_model.eval()
            test_predict = []
            test_probs_pairs = []
            dir_to_minigraphs = {}
            for x, y in test_data_loader:
                pred = bilstm_model(x)
                loss = citerion(pred, y)

                loss.backward()
                optimizer.step()
                test_predict.extend(pred.argmax(dim=1).tolist())

                for p_list in pred.tolist():
                    test_probs_pairs.append(p_list)

            eval(dir_to_minigraphs, test_all_fdirs, test_all_nodes)
            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall2 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)
            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)

            info = {}
            info["recall@1"] = top_recall1
            info["recall@2"] = top_recall2
            info["recall@3"] = top_recall3
            info["mfr"] = mfr
            all_info.append(info)
            with open(f"./cross_fold/{i}.json", "w") as f:
                json.dump(all_info, f)


def do_cross_project():
    data1 = json.load(open("../dataset1.json"))
    data2 = json.load(open("../dataset2.json"))
    data3 = json.load(open("../dataset3.json"))

    trainset = data1
    testset = []
    testset.append(data2)
    testset.append(data3)
    (
        train_all_nodes,
        train_all_codes,
        train_all_labels,
        train_all_fdirs,
        train_all_info,
    ) = get_dataset(trainset)
    (
        test_all_nodes,
        test_all_codes,
        test_all_labels,
        test_all_fdirs,
        test_all_info,
    ) = get_dataset(testset)
    train_data_set = dl_dataset(
        train_all_nodes,
        train_all_codes,
        train_all_labels,
        train_all_fdirs,
        train_all_info,
    )
    test_data_set = dl_dataset(
        test_all_nodes, test_all_codes, test_all_labels, test_all_fdirs, test_all_info
    )
    train_data_loader = DataLoader(
        train_data_set, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        test_data_set, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    bilstm_model = bilstmModel(300, 150, 5, 0.1)
    optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, 100):
        train_predict = []
        bilstm_model.train()
        total_loss = 0
        for x, y in train_data_loader:
            pred = bilstm_model(x)
            loss = citerion(pred, y)

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.detach().item()
            train_predict.extend(pred.argmax(dim=1).tolist())

        bilstm_model.eval()
        test_predict = []
        test_probs_pairs = []
        dir_to_minigraphs = {}
        for x, y in test_data_loader:
            pred = bilstm_model(x)
            loss = citerion(pred, y)

            loss.backward()
            optimizer.step()

            test_predict.extend(pred.argmax(dim=1))

            for p_list in pred.tolist():
                test_probs_pairs.append(p_list)

        eval(dir_to_minigraphs, test_all_fdirs, test_all_nodes)
        top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
        top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
        top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)
        mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)

        info = {}
        info["recall@1"] = top_recall1
        info["recall@2"] = top_recall2
        info["recall@3"] = top_recall3
        info["mfr"] = mfr
        all_info.append(info)

        with open(f"./cross_project/result.json", "w") as f:
            json.dump(all_info, f)


if __name__ == "__main__":
    do_cross_fold()
