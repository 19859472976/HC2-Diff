import os
import pickle
import dgl
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch


data_dir = "./ICEWS14"
output_file = "graph_dict.pkl"



def load_data(data_dir):
    files = ["train.txt", "valid.txt", "test.txt"]
    all_quads = []
    for file in files:
        with open(os.path.join(data_dir, file), "r") as f:
            for line in f:
                line = line.strip().split("\t")
                if len(line) == 4:
                    h, r, t, ts = line
                    all_quads.append((h, r, t, ts))
    return all_quads



def group_by_time(quads):
    time_dict = defaultdict(list)
    for h, r, t, ts in quads:
        time_dict[ts].append((h, r, t))
    return time_dict



def build_graph(quads, entity2id, rel2id):
    edges = []
    for h, r, t in quads:
        edges.append((entity2id[h], entity2id[t], rel2id[r]))

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    rel = [e[2] for e in edges]
    graph = dgl.graph((src, dst), num_nodes=len(entity2id))
    graph.edata["rel"] = torch.tensor(rel)
    return graph



if __name__ == "__main__":

    all_quads = load_data(data_dir)


    entities = set()
    relations = set()
    for h, r, t, ts in all_quads:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    entity2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}


    time_dict = group_by_time(all_quads)


    graph_dict = {}
    for ts, quads in time_dict.items():
        graph = build_graph(quads, entity2id, rel2id)
        ts_int = int(ts)
        graph_dict[ts_int] = graph


    with open(output_file, "wb") as f:
        pickle.dump(graph_dict, f)

    print(f"Saved graph_dict.pkl with {len(graph_dict)} timestamps.")