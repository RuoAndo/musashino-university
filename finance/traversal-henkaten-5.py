# -*- coding: utf-8 -*-
"""
変化点検知 完全版
対応 method:
- pelt
- binseg
- bottomup
- window
- dynp
"""

import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.tsa.ar_model import AutoReg


# =========================================================
# 0. パラメータ
# =========================================================

DATA_DIR = Path(".")
FILE_PATTERN = "*.csv"

OUTPUT_PREFIX = "change_point"

# -------------------------
# 変化点検知設定
# -------------------------
CP_METHOD = "bottomup"      # pelt / binseg / bottomup / window / dynp
CHANGE_MODEL = "l1"        # l1 / l2 / rbf / normal / linear など

CHANGE_N_BKPS = 10         # binseg / bottomup / window / dynp 用
CHANGE_PEN_BASE = 2.0      # pelt 用

CHANGE_MIN_SIZE = 3
CHANGE_JUMP = 3
WINDOW_WIDTH = 20          # window 用

MIN_RETURNS_FOR_CP = 20
MAX_POINTS_FOR_CP = 2000

# -------------------------
# ARモデル設定
# -------------------------
AR_LAGS = 5
DO_AR = True

# -------------------------
# 出力設定
# -------------------------
CSV_ENCODING = "cp932"

# -------------------------
# 再現性
# -------------------------
USE_FIXED_SEED = False
RANDOM_SEED = 42


# =========================================================
# 1. ログ関数
# =========================================================

GLOBAL_START = time.time()

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - GLOBAL_START
    print(f"[{now} | +{elapsed:9.2f}s] {msg}", flush=True)


# =========================================================
# 2. CSV列の自動判定
# =========================================================

PRICE_CANDIDATES = [
    "price", "Price",
    "close", "Close",
    "market_price",
    "Market Price",
    "value", "Value"
]

TIME_CANDIDATES = [
    "timestamp", "Timestamp",
    "datetime", "Datetime",
    "date", "Date",
    "time", "Time"
]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# 3. CSV読み込み
# =========================================================

def read_csv_safely(path):
    encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "iso-8859-1"]

    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e

    raise last_err


# =========================================================
# 4. 前処理
# =========================================================

def prepare_price_series(path):
    df = read_csv_safely(path)

    time_col = find_col(df, TIME_CANDIDATES)
    price_col = find_col(df, PRICE_CANDIDATES)

    if price_col is None:
        raise ValueError(f"価格列が見つかりません: {path.name}")

    if time_col is None:
        df["_time"] = np.arange(len(df))
        time_col = "_time"
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]

    if time_col != "_time":
        df = df.dropna(subset=[time_col])
        df = df.sort_values(time_col)
        df = df.drop_duplicates(subset=[time_col])

    prices = df[price_col].to_numpy