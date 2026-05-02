# -*- coding: utf-8 -*-

import time
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, to_networkx
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")


# ==========================
# 設定
# ==========================
TXS_FEATURES = "./transactions/txs_features.txt"
TXS_CLASSES  = "./transactions/txs_classes.txt"
TXS_EDGES    = "./transactions/txs_edgelist.txt"

SEED = 42

TRAIN_END_STEP = 34
VAL_END_STEP = 39

EPOCHS = 150
LR = 0.001
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
DROPOUT = 0.5

BETWEENNESS_K = 100

OUTPUT_UNKNOWN_DETAIL = "unknown_outlier_with_gcn_score.csv"
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
# GCN
# ==========================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        return self.lin(x)


def tune_threshold(y_true, prob_illicit):
    best_th = 0.5
    best_f1 = -1

    for th in np.arange(0.05, 0.96, 0.05):
        pred = (prob_illicit >= th).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1


# ==========================
# main
# ==========================
def main():
    global START_TIME
    START_TIME = time.time()

    log("START")
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # ======================
    # 読み込み
    # ======================
    log("CSV読み込み開始")

    df_f = pd.read_csv(TXS_FEATURES)
    df_c = pd.read_csv(TXS_CLASSES)
    df_e = pd.read_csv(TXS_EDGES)

    log(f"features={df_f.shape} classes={df_c.shape} edges={df_e.shape}")

    # ======================
    # txIdでclassを結合
    # ======================
    id_col_f = df_f.columns[0]
    id_col_c = df_c.columns[0]

    df = df_f.merge(df_c, left_on=id_col_f, right_on=id_col_c, how="left")

    if df["class"].isna().any():
        log("警告: classが欠損している行があります。unknown扱いにします。")
        df["class"] = df["class"].fillna(3)

    ids = df[id_col_f].astype(int).values
    id2idx = {int(v): i for i, v in enumerate(ids)}

    # class変換
    # 1: illicit → 0
    # 2: licit   → 1
    # 3: unknown → -1
    y_np = df["class"].map(
        lambda x: 0 if int(x) == 1 else 1 if int(x) == 2 else -1
    ).values

    time_steps_np = df["Time step"].astype(int).values

    # ======================
    # 特徴量
    # ======================
    exclude_cols = {
        id_col_f,
        id_col_c,
        "class",
        "Time step"
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    log(f"特徴量数={len(feature_cols)}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    train_mask_np = (
        (y_np != -1) &
        (time_steps_np <= TRAIN_END_STEP)
    )

    val_mask_np = (
        (y_np != -1) &
        (time_steps_np > TRAIN_END_STEP) &
        (time_steps_np <= VAL_END_STEP)
    )

    test_mask_np = (
        (y_np != -1) &
        (time_steps_np > VAL_END_STEP)
    )

    scaler = StandardScaler()
    X_scaled = X.values.astype(np.float32)
    X_scaled[train_mask_np] = scaler.fit_transform(X_scaled[train_mask_np])
    X_scaled[~train_mask_np] = scaler.transform(X_scaled[~train_mask_np])

    # ======================
    # エッジ
    # ======================
    log("edge構築開始")

    col0, col1 = df_e.columns[:2]

    edges = []
    skipped = 0

    for u, v in zip(df_e[col0], df_e[col1]):
        try:
            u = int(u)
            v = int(v)
        except Exception:
            skipped += 1
            continue

        if u in id2idx and v in id2idx:
            edges.append([id2idx[u], id2idx[v]])
        else:
            skipped += 1

    log(f"edge構築完了: edges={len(edges)} skipped={skipped}")

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(ids))

    # ======================
    # Data
    # ======================
    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_np, dtype=torch.long),
        time_steps=torch.tensor(time_steps_np, dtype=torch.long),
        node_ids=torch.tensor(ids, dtype=torch.long),
        train_mask=torch.tensor(train_mask_np, dtype=torch.bool),
        val_mask=torch.tensor(val_mask_np, dtype=torch.bool),
        test_mask=torch.tensor(test_mask_np, dtype=torch.bool)
    )

    data = data.to(device)

    log(f"train={int(data.train_mask.sum())} val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}")
    log(f"unknown={(y_np == -1).sum()}")

    # ======================
    # GCN学習
    # ======================
    log("GCN学習開始")

    model = GCN(
        in_dim=data.x.size(1),
        hidden_dim=HIDDEN_DIM,
        out_dim=2
    ).to(device)

    train_y = data.y[data.train_mask]
    counts = torch.bincount(train_y, minlength=2).float()
    weights = counts.sum() / (counts + 1e-9)
    weights = weights / weights.mean()
    weights = weights.to(device)

    log(f"class weights={weights.detach().cpu().numpy()}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask],
            weight=weights
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            prob = F.softmax(out, dim=1)

            val_prob_illicit = prob[data.val_mask, 0].detach().cpu().numpy()
            val_true_raw = data.y[data.val_mask].detach().cpu().numpy()

            # illicitを1として評価
            val_true = (val_true_raw == 0).astype(int)

            th, val_f1 = tune_threshold(val_true, val_prob_illicit)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "model": model.state_dict(),
                "threshold": th,
                "epoch": epoch
            }

        if epoch == 1 or epoch % 10 == 0:
            log(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"val_illicit_f1={val_f1:.4f} "
                f"best={best_val_f1:.4f} "
                f"th={th:.2f}"
            )

    log("GCN学習完了")

    model.load_state_dict(best_state["model"])
    best_th = best_state["threshold"]

    log(f"best_epoch={best_state['epoch']} best_threshold={best_th:.2f}")

    # ======================
    # test評価
    # ======================
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        prob = F.softmax(out, dim=1)

    test_prob_illicit = prob[data.test_mask, 0].detach().cpu().numpy()
    test_true_raw = data.y[data.test_mask].detach().cpu().numpy()

    test_true = (test_true_raw == 0).astype(int)
    test_pred = (test_prob_illicit >= best_th).astype(int)

    print("\n=== Test Confusion Matrix ===")
    print(confusion_matrix(test_true, test_pred))

    print("\n=== Test Classification Report ===")
    print(classification_report(
        test_true,
        test_pred,
        target_names=["licit", "illicit"],
        zero_division=0
    ))

    # ======================
    # 中心性計算
    # ======================
    log("中心性計算開始")

    data_cpu = data.cpu()
    G = to_networkx(data_cpu, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    log(f"Graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    deg = dict(G.degree())

    log("betweenness計算開始")
    bet = nx.betweenness_centrality(G, k=BETWEENNESS_K, seed=SEED)

    log("pagerank計算開始")
    pr = nx.pagerank(G)

    log("中心性計算完了")

    # ======================
    # unknown抽出
    # ======================
    prob_cpu = prob.detach().cpu().numpy()
    illicit_prob_all = prob_cpu[:, 0]

    rows = []

    for i in range(len(y_np)):
        if y_np[i] != -1:
            continue

        rows.append({
            "txId": int(ids[i]),
            "time_step": int(time_steps_np[i]),
            "degree": deg.get(i, 0),
            "betweenness": bet.get(i, 0),
            "pagerank": pr.get(i, 0),
            "gcn_illicit_prob": float(illicit_prob_all[i])
        })

    df_u = pd.DataFrame(rows)

    log(f"unknown数={len(df_u)}")

    # ======================
    # 外れ値スコア
    # ======================
    df_u["d_z"] = robust_z(df_u["degree"])
    df_u["b_z"] = robust_z(df_u["betweenness"])
    df_u["p_z"] = robust_z(df_u["pagerank"])

    df_u["centrality_score"] = df_u[["d_z", "b_z", "p_z"]].abs().max(axis=1)

    # 中心性外れ値 × GCN疑わしさ
    df_u["combined_score"] = (
        df_u["centrality_score"].rank(pct=True) * 0.5
        + df_u["gcn_illicit_prob"].rank(pct=True) * 0.5
    )

    # ======================
    # Top10表示
    # ======================
    log("unknown 外れ値 Top10 ノード")

    df_top10 = df_u.sort_values("combined_score", ascending=False).head(10)

    for rank, (_, row) in enumerate(df_top10.iterrows(), start=1):
        print(f"\n--- rank {rank} ---")
        print(f"txId             : {int(row['txId'])}")
        print(f"time_step        : {int(row['time_step'])}")
        print(f"degree           : {int(row['degree'])}")
        print(f"betweenness      : {row['betweenness']:.6e}")
        print(f"pagerank         : {row['pagerank']:.6e}")
        print(f"centrality_score : {row['centrality_score']:.6f}")
        print(f"gcn_illicit_prob : {row['gcn_illicit_prob']:.6f}")
        print(f"combined_score   : {row['combined_score']:.6f}")

    # ======================
    # time_step集計
    # ======================
    summary = df_u.groupby("time_step").agg(
        unknown_count=("combined_score", "count"),
        avg_degree=("degree", "mean"),
        avg_betweenness=("betweenness", "mean"),
        avg_pagerank=("pagerank", "mean"),
        max_centrality_score=("centrality_score", "max"),
        avg_centrality_score=("centrality_score", "mean"),
        avg_gcn_illicit_prob=("gcn_illicit_prob", "mean"),
        max_gcn_illicit_prob=("gcn_illicit_prob", "max"),
        avg_combined_score=("combined_score", "mean"),
        max_combined_score=("combined_score", "max")
    ).reset_index()

    log(f"time_step集計 rows={len(summary)}")

    # ======================
    # CSV保存
    # ======================
    df_u.to_csv(OUTPUT_UNKNOWN_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_TIME_STEP_SUMMARY, index=False, encoding="utf-8-sig")

    log(f"CSV保存: {OUTPUT_UNKNOWN_DETAIL}")
    log(f"CSV保存: {OUTPUT_TIME_STEP_SUMMARY}")
    log("DONE")


if __name__ == "__main__":
    main()