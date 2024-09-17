import csv
import pandas as pd
import numpy as np
import time
import warnings
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from ml_datatset import get_dataset
import os
from itertools import chain
from functools import cmp_to_key
from typing import Dict, List, Union
from settings import *
SEED = 2023

def get_true_cid_map(fdirs):
    true_cid_map = {}
    for fdir in fdirs:
        true_cid_map[fdir] = set()
        with open(f"{DATA_PATH}/{fdir}/info.json") as f:
            info = json.load(f)
            for cid in set(info["induce"]):
                true_cid_map[fdir].add(cid)
    return true_cid_map


def cmp(n1, n2):
    score1 = n1["prob"]
    score2 = n2["prob"]
    if score1 < score2:
        return 1
    elif score1 > score2:
        return -1
    return 0


def eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes):
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
    total_find_cnt = 0
    for fdir in set(fdirs):
        for i, node in enumerate(dir_to_minigraphs[fdir]):
            if node["rootcause"]:
                total_rank_cnt = total_rank_cnt + i + 1
                total_find_cnt = total_find_cnt + 1
                break

    return total_rank_cnt / total_find_cnt


def eval_top(fdirs, dir_to_minigraphs, true_cid_map, k):
    tp = 0
    fp = 0
    total_t = 0

    for fdir in set(fdirs):
        cidSet = set()
        total_t = total_t + len(true_cid_map[fdir])
        for node in dir_to_minigraphs[fdir][:k]:
            f = False
            for cid in node["commits"]:
                if cid not in cidSet and cid in true_cid_map[fdir]:
                    f = True
                    cidSet.add(cid)
            # the node is rootcause
            if f:
                tp = tp + 1
                continue
            # each node should correspond to one commit
            f1 = False
            for cid in node["commits"]:
                if cid not in cidSet:
                    cidSet.add(cid)
                    f1 = True
            if f1:
                fp = fp + 1
    return tp, fp, total_t


def selectFromLinearSVC2(train_content, train_label, test_content):
    lsvc = LinearSVC().fit(train_content, train_label)
    model = SelectFromModel(lsvc, prefit=True)

    new_train = model.transform(train_content)
    new_test = model.transform(test_content)

    return new_train, new_test


def divide_lst(lst, n, k):
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):
        if cnt < k - 1:
            l = lst[i : i + n]
            random.shuffle(l)
            all_list.append(l)
        else:
            l = lst[i:]
            random.shuffle(l)
            all_list.append(l)
            break
        cnt = cnt + 1
    return all_list


def do_cross_fold():
    data1 = json.load(open("../dataset1.json"))
    data2 = json.load(open("../dataset2.json"))
    data3 = json.load(open("../dataset3.json"))
    all_data = []
    all_data.extend(data1)
    all_data.extend(data2)
    all_data.extend(data3)

    all_data_list = divide_lst(all_data, int(len(all_data) / 10), 10)

    learners = ["RF", "SVM", "LR", "KNN", "XGB"]

    rf_rtop1 = []
    rf_rtop2 = []
    rf_rtop3 = []
    rf_mfr = []

    svm_rtop1 = []
    svm_rtop2 = []
    svm_rtop3 = []
    svm_mfr = []

    lr_rtop1 = []
    lr_rtop2 = []
    lr_rtop3 = []
    lr_mfr = []

    knn_rtop1 = []
    knn_rtop2 = []
    knn_rtop3 = []
    knn_mfr = []

    xgb_rtop1 = []
    xgb_rtop2 = []
    xgb_rtop3 = []
    xgb_mfr = []
    for i in range(0, len(all_data_list)):
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

        vectorizer = CountVectorizer(max_features=10000, tokenizer=None, stop_words=None)

        train_content_matrix = vectorizer.fit_transform(train_all_codes)
        test_content_matrix = vectorizer.transform(test_all_codes)
        train_content_matrix, test_content_matrix = selectFromLinearSVC2(
            train_content_matrix, train_all_labels, test_content_matrix
        )

        train_x = train_content_matrix.toarray()
        train_y = train_all_labels
        test_x = test_content_matrix.toarray()
        test_y = test_all_labels
        learners = ["RF", "SVM", "LR", "KNN", "XGB"]
        workers = 16
        true_cid_map = get_true_cid_map(test_data)

        for l in learners:
            if l == "RF":
                clf = RandomForestClassifier(n_jobs=workers, random_state=SEED)
                clf.fit(train_x, train_y)
                proba = clf.predict_proba(test_x)

                all_proba_pairs = clf.predict_proba(test_x).tolist()

                dir_to_minigraphs = {}
                eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

                rf_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
                rf_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
                rf_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
                rf_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))
                

            elif l == "SVM":
                clf = SVC(kernel="rbf", probability=True, random_state=SEED)
                clf.fit(train_x, train_y)
                proba = clf.predict_proba(test_x)

                all_proba_pairs = clf.predict_proba(test_x).tolist()
                dir_to_minigraphs = {}
                eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

                svm_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
                svm_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
                svm_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
                svm_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))
                

            elif l == "LR":
                clf = LogisticRegression(
                    multi_class="ovr", max_iter=1000, n_jobs=workers
                )
                clf.fit(train_x, train_y)
                proba = clf.predict_proba(test_x)

                all_proba_pairs = clf.predict_proba(test_x).tolist()
                dir_to_minigraphs = {}
                eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

                lr_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
                lr_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
                lr_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
                lr_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))
                

            elif l == "KNN":
                clf = KNeighborsClassifier(n_jobs=workers)
                clf.fit(train_x, train_y)
                proba = clf.predict_proba(test_x)

                all_proba_pairs = clf.predict_proba(test_x).tolist()
                dir_to_minigraphs = {}
                eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

                knn_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
                knn_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
                knn_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
                knn_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))


            elif l == "XGB":
                classes = sorted(list(set(train_y)))
                train_y_index = [classes.index(y) for y in train_y]
                clf = XGBClassifier(
                    n_jobs=workers, seed=SEED, eval_metric="logloss", max_depth=10
                )
                clf.fit(train_x, train_y_index)

                all_proba_pairs = clf.predict_proba(test_x).tolist()
                dir_to_minigraphs = {}
                eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

                xgb_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
                xgb_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
                xgb_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
                xgb_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))


    test_result = {}
    test_result["rf"] = {}
    test_result["rf"]["recall@1"] = sum(rf_rtop1) / len(rf_rtop1)
    test_result["rf"]["recall@2"] = sum(rf_rtop2) / len(rf_rtop2)
    test_result["rf"]["recall@3"] = sum(rf_rtop3) / len(rf_rtop3)
    test_result["rf"]["mfr"] = sum(rf_mfr) / len(rf_mfr)
    
    test_result["svm"] = {}
    test_result["svm"]["recall@1"] = sum(svm_rtop1) / len(svm_rtop1)
    test_result["svm"]["recall@2"] = sum(svm_rtop2) / len(svm_rtop2)
    test_result["svm"]["recall@3"] = sum(svm_rtop3) / len(svm_rtop3)
    test_result["svm"]["mfr"] = sum(svm_mfr) / len(svm_mfr)

    test_result["lr"] = {}
    test_result["lr"]["recall@1"] = sum(lr_rtop1) / len(lr_rtop1)
    test_result["lr"]["recall@2"] = sum(lr_rtop2) / len(lr_rtop2)
    test_result["lr"]["recall@3"] = sum(lr_rtop3) / len(lr_rtop3)
    test_result["lr"]["mfr"] = sum(lr_mfr) / len(lr_mfr)

    test_result["knn"] = {}
    test_result["knn"]["recall@1"] = sum(knn_rtop1) / len(knn_rtop1)
    test_result["knn"]["recall@2"] = sum(knn_rtop2) / len(knn_rtop2)
    test_result["knn"]["recall@3"] = sum(knn_rtop3) / len(knn_rtop3)
    test_result["knn"]["mfr"] = sum(knn_mfr) / len(knn_mfr)

    test_result["xgb"] = {}
    test_result["xgb"]["recall@1"] = sum(xgb_rtop1) / len(xgb_rtop1)
    test_result["xgb"]["recall@2"] = sum(xgb_rtop2) / len(xgb_rtop2)
    test_result["xgb"]["recall@3"] = sum(xgb_rtop3) / len(xgb_rtop3)
    test_result["xgb"]["mfr"] = sum(xgb_mfr) / len(xgb_mfr)

    with open("result.json", "w") as f:
        json.dump(test_result, f)


def do_cross_project_predict():
    trainset = []
    with open("../dataset2.json") as f:
        trainset.append(json.load(f))

    with open("../dataset3.json") as f:
        trainset.append(json.load(f))

    with open("dataset1.json") as f:
        testset = json.load(f)

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

    vectorizer = CountVectorizer(max_features=10000, tokenizer=None, stop_words=None)

    train_content_matrix = vectorizer.fit_transform(train_all_codes)
    test_content_matrix = vectorizer.transform(test_all_codes)

    train_x = train_content_matrix.toarray()
    train_y = train_all_labels
    test_x = test_content_matrix.toarray()
    test_y = test_all_labels

    learners = ["RF", "SVM", "LR", "KNN", "XGB"]
    workers = 32

    for l in learners:
        if l == "RF":
            clf = RandomForestClassifier(max_depth=None, n_jobs=workers)
            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            all_proba_pairs = clf.predict_proba(test_x).tolist()
            dir_to_minigraphs = {}
            eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)

            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)
            print(f"RF")
            print(f"recall at top 1 {top_recall1}")
            print(f"recall at top 2 {top_recall3}")
            print(f"recall at top 3 {top_recall5}")
            print(f"mean first rank {mfr}")
        elif l == "SVM":
            clf = SVC(kernel="rbf", probability=True, random_state=SEED)
            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            all_proba_pairs = clf.predict_proba(test_x).tolist()
            dir_to_minigraphs = {}
            eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)

            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)
            print(f"SVM")
            print(f"recall at top 1 {top_recall1}")
            print(f"recall at top 2 {top_recall3}")
            print(f"recall at top 3 {top_recall5}")
            print(f"mean first rank {mfr}")
        elif l == "LR":
            clf = LogisticRegression(multi_class="ovr", max_iter=1000, n_jobs=workers)
            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            all_proba_pairs = clf.predict_proba(test_x).tolist()
            dir_to_minigraphs = {}
            eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)

            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)
            print(f"LR")
            print(f"recall at top 1 {top_recall1}")
            print(f"recall at top 2 {top_recall3}")
            print(f"recall at top 3 {top_recall5}")
            print(f"mean first rank {mfr}")
        elif l == "KNN":
            clf = KNeighborsClassifier(n_jobs=workers)
            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            all_proba_pairs = clf.predict_proba(test_x).tolist()
            dir_to_minigraphs = {}
            eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)

            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)
            print(f"KNN")
            print(f"recall at top 1 {top_recall1}")
            print(f"recall at top 2 {top_recall3}")
            print(f"recall at top 3 {top_recall5}")
            print(f"mean first rank {mfr}")
        elif l == "XGB":
            classes = sorted(list(set(train_y)))
            train_y_index = [classes.index(y) for y in train_y]
            clf = XGBClassifier(
                n_jobs=workers, seed=SEED, eval_metric="logloss", max_depth=10
            )
            clf.fit(train_x, train_y_index)

            all_proba_pairs = clf.predict_proba(test_x).tolist()
            dir_to_minigraphs = {}
            eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)

            top_recall1 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 1)
            top_recall3 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 2)
            top_recall5 = eval_recall_topk(test_all_fdirs, dir_to_minigraphs, 3)

            mfr = eval_mean_first_rank(test_all_fdirs, dir_to_minigraphs)
            print(f"XGB")
            print(f"recall at top 1 {top_recall1}")
            print(f"recall at top 2 {top_recall3}")
            print(f"recall at top 3 {top_recall5}")
            print(f"mean first rank {mfr}")


if __name__ == "__main__":
    do_cross_fold()
    do_cross_project_predict()
