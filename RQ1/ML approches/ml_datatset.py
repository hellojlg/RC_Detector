import json
import os
from settings import *
from tokenizer import *


def get_dataset(all_data, tokenizer_max_length=32):
    all_nodes = []
    all_codes = []
    all_labels = []
    all_fdirs = []
    all_info = []

    for fdir in all_data:
        graph = json.load(open(os.path.join(DATA_PATH, fdir, "graph1.json")))
        info = json.load(open(os.path.join(DATA_PATH, fdir, "info.json")))
        for node in graph:
            if not node["isDel"]:
                continue

            all_nodes.append(node)
            all_codes.append(
                " ".join(tokenize_text(tokenize_by_punctuation(node["code"])))
            )
            if node["rootcause"]:
                all_labels.append(1)
            else:
                all_labels.append(0)

            all_fdirs.append(fdir)
            all_info.append(info)

    return all_nodes, all_codes, all_labels, all_fdirs, all_info
