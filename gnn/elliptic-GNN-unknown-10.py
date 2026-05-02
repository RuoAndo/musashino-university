# -*- coding: utf-8 -*-

import time
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, to_networkx

warnings.filterwarnings("ignore")


# ==========================
# 設定
# ==========================
TXS_FEATURES = "./transactions/txs_features.txt"
TXS_CLASSES  = "./transactions/txs_classes.txt"
TXS_EDGES    = "./transactions/txs_edgelist.txt"

SEED = 42
BETWEENNESS_K = 100
OUTPUT_TIME_STEP_SUMMARY = "unknown_outlier_summary_by_time_step.csv"


# ==========================
# ログ
# ==========================
START_TIME = None

def log(msg):
    t = int(time.time() - START_TIME)
    print(f"[{datetime.now().strftime('%H:%M:%S')} +{t}s] {msg}")


# ==========================
# 基本
# ==========================
def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ==========================
# 外れ値スコア
# ==========================
def robust_z(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    med = s.median()
    mad = (s - med).abs().median()

    if mad == 0:
        return (s - s.mean()) / (s.std() + 1e-9)

    return 0.6745 * (s - med) / mad


# ==========================
# main
# ==========================
def main():
    global START_TIME
    START_TIME = time.time()

    log("START")
    set_seed()

    # ======================
    # 読み込み
    # ======================
    df_f = pd.read_csv(TXS_FEATURES)
    df_c = pd.read_csv(TXS_CLASSES)
    df_e = pd.read_csv(TXS_EDGES)

    log(f"features={df_f.shape} classes={df_c.shape} edges={df_e.shape}")

    # ======================
    # ノード
    # ======================
    ids = df_f.iloc[:, 0].values
    id2idx = {int(v): i for i, v in enumerate(ids)}

    # class: 3 → unknown
    y = df_c["class"].map(lambda x: -1 if x == 3 else 0).values

    # ======================
    # エッジ（高速）
    # ======================
    col0, col1 = df_e.columns[:2]

    edges = [
        [id2idx[int(u)], id2idx[int(v)]]
        for u, v in zip(df_e[col0], df_e[col1])
        if int(u) in id2idx and int(v) in id2idx
    ]

    log(f"edge構築完了: {len(edges)}")

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    # ======================
    # Data
    # ======================
    data = Data()
    data.edge_index = edge_index
    data.y = torch.tensor(y)
    data.time_steps = torch.tensor(df_f["Time step"].values)
    data.node_ids = torch.tensor(ids)

    # ======================
    # グラフ
    # ======================
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    log(f"Graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # ======================
    # 中心性
    # ======================
    log("中心性計算")

    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, k=BETWEENNESS_K, seed=SEED)
    pr = nx.pagerank(G)

    # ======================
    # unknown抽出
    # ======================
    rows = []
    ts = data.time_steps.numpy()
    node_ids = data.node_ids.numpy()

    for i in range(len(y)):
        if y[i] != -1:
            continue

        rows.append({
            "txId": int(node_ids[i]),
            "time_step": int(ts[i]),
            "degree": deg.get(i, 0),
            "betweenness": bet.get(i, 0),
            "pagerank": pr.get(i, 0)
        })

    df = pd.DataFrame(rows)

    log(f"unknown数={len(df)}")

    # ======================
    # 外れ値スコア
    # ======================
    df["d_z"] = robust_z(df["degree"])
    df["b_z"] = robust_z(df["betweenness"])
    df["p_z"] = robust_z(df["pagerank"])

    df["score"] = df[["d_z", "b_z", "p_z"]].abs().max(axis=1)

    # ======================
    # Top10表示
    # ======================
    log("外れ値 Top10 ノード")

    df_top10 = df.sort_values("score", ascending=False).head(10)

    for rank, (_, row) in enumerate(df_top10.iterrows(), start=1):
        print(f"\n--- rank {rank} ---")
        print(f"txId        : {int(row['txId'])}")
        print(f"time_step   : {int(row['time_step'])}")
        print(f"degree      : {int(row['degree'])}")
        print(f"betweenness : {row['betweenness']:.6e}")
        print(f"pagerank    : {row['pagerank']:.6e}")
        print(f"score       : {row['score']:.6f}")

    # ======================
    # time_step集計（表示なし）
    # ======================
    summary = df.groupby("time_step").agg(
        count=("score", "count"),
        avg_degree=("degree", "mean"),
        max_score=("score", "max"),
        avg_score=("score", "mean")
    ).reset_index()

    log(f"time_step集計 rows={len(summary)}")

    summary.to_csv(OUTPUT_TIME_STEP_SUMMARY, index=False, encoding="utf-8-sig")

    log("CSV保存完了")
    log("DONE")


if __name__ == "__main__":
    main()