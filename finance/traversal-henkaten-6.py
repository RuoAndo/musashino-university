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

# DATA_DIR = Path(".")
DATA_DIR = Path(r"D:\musashino-university\finance\coingecko_by_coin")
FILE_PATTERN = "*.csv"

OUTPUT_PREFIX = "change_point"

CP_METHOD = "bottomup"      # pelt / binseg / bottomup / window / dynp
CHANGE_MODEL = "l1"        # l1 / l2 / rbf / normal / linear

CHANGE_N_BKPS = 10         # binseg / bottomup / window / dynp 用
CHANGE_PEN_BASE = 2.0      # pelt 用

CHANGE_MIN_SIZE = 3
CHANGE_JUMP = 3
WINDOW_WIDTH = 20

MIN_RETURNS_FOR_CP = 20
MAX_POINTS_FOR_CP = 2000

AR_LAGS = 5
DO_AR = True

CSV_ENCODING = "cp932"

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
    "market_price", "Market Price",
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

    prices = df[price_col].to_numpy(dtype=float)

    if len(prices) < 2:
        raise ValueError("価格データが少なすぎます")

    returns = np.diff(np.log(prices))

    if time_col == "_time":
        times = np.arange(1, len(prices))
    else:
        times = df[time_col].iloc[1:].to_numpy()

    mask = np.isfinite(returns)
    returns = returns[mask]
    times = times[mask]

    return df, returns, times


# =========================================================
# 5. ARモデル
# =========================================================

def run_ar_model(returns):
    if not DO_AR:
        return np.nan

    if len(returns) <= AR_LAGS + 5:
        return np.nan

    try:
        model = AutoReg(returns, lags=AR_LAGS, old_names=False)
        res = model.fit()
        pred = res.predict(start=AR_LAGS, end=len(returns) - 1)
        actual = returns[AR_LAGS:]

        mse = np.mean((actual - pred) ** 2)
        rmse = float(np.sqrt(mse))
        return rmse

    except Exception:
        return np.nan


# =========================================================
# 6. 変化点検知
# =========================================================

def detect_change_points(returns):
    method = CP_METHOD.lower()

    if len(returns) < MIN_RETURNS_FOR_CP:
        return []

    signal = returns.copy()

    if len(signal) > MAX_POINTS_FOR_CP:
        idx = np.linspace(0, len(signal) - 1, MAX_POINTS_FOR_CP).astype(int)
        signal_small = signal[idx]
        scale_idx = idx
    else:
        signal_small = signal
        scale_idx = None

    signal_2d = signal_small.reshape(-1, 1)
    n = len(signal_2d)

    max_possible_bkps = max(1, n // CHANGE_MIN_SIZE - 1)
    n_bkps = min(CHANGE_N_BKPS, max_possible_bkps)

    if n_bkps <= 0:
        return []

    log(
        f"CP開始: method={method}, model={CHANGE_MODEL}, "
        f"len={n}, min_size={CHANGE_MIN_SIZE}, jump={CHANGE_JUMP}"
    )

    if method == "pelt":
        algo = rpt.Pelt(
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(pen=CHANGE_PEN_BASE)

    elif method == "binseg":
        algo = rpt.Binseg(
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "bottomup":
        algo = rpt.BottomUp(
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "window":
        algo = rpt.Window(
            width=WINDOW_WIDTH,
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "dynp":
        algo = rpt.Dynp(
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    else:
        raise ValueError(f"未対応のCP_METHODです: {CP_METHOD}")

    # ruptures は最後の点 len(signal) を含めるので除外
    cps = [b for b in bkps if b < n]

    if scale_idx is not None:
        cps = [int(scale_idx[min(cp, len(scale_idx) - 1)]) for cp in cps]

    cps = sorted(set(cps))

    log(f"CP終了: n_change_points={len(cps)}")

    return cps


# =========================================================
# 7. 1ファイル処理
# =========================================================

def process_one_file(path, index, total):
    name = path.stem
    start = time.time()

    log(f"--- {name} 処理開始: {index}/{total} ---")

    summary = {
        "file": path.name,
        "symbol": name,
        "rows": 0,
        "return_rows": 0,
        "rmse": np.nan,
        "n_change_points": 0,
        "status": "ok",
        "error": ""
    }

    pairs = []

    try:
        df, returns, times = prepare_price_series(path)

        summary["rows"] = len(df)
        summary["return_rows"] = len(returns)

        log(f"[{name}] 読込完了: rows={len(df)}, return_rows={len(returns)}")

        log(f"[{name}] AR開始")
        ar_start = time.time()
        rmse = run_ar_model(returns)
        summary["rmse"] = rmse
        log(f"[{name}] AR終了: RMSE={rmse}, time={time.time() - ar_start:.2f}s")

        cp_start = time.time()
        cps = detect_change_points(returns)
        summary["n_change_points"] = len(cps)

        for cp in cps:
            if cp < len(times):
                cp_time = times[cp]
            else:
                cp_time = ""

            left = max(0, cp - 5)
            right = min(len(returns), cp + 5)

            strength = float(np.mean(np.abs(returns[left:right])))

            pairs.append({
                "symbol": name,
                "file": path.name,
                "cp_index": int(cp),
                "datetime": cp_time,
                "strength": strength
            })

        log(
            f"[{name}] CP終了: n={len(cps)}, "
            f"time={time.time() - cp_start:.2f}s"
        )

    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        log(f"[{name}] 処理失敗: {e}")

    progress = index / total * 100
    log(
        f"--- {name} 処理終了: total_time={time.time() - start:.2f}s, "
        f"progress={progress:.1f}% ---"
    )

    log(
        f"[{name}] summary: rows={summary['rows']}, "
        f"return_rows={summary['return_rows']}, "
        f"rmse={summary['rmse']}, "
        f"n_change_points={summary['n_change_points']}"
    )

    return summary, pairs


# =========================================================
# 8. メイン処理
# =========================================================

def main():
    if USE_FIXED_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    log("処理開始")

    files = sorted(DATA_DIR.glob(FILE_PATTERN))

    # 出力済みファイルを再処理しない
    files = [
        f for f in files
        if not f.name.startswith(OUTPUT_PREFIX)
    ]

    if not files:
        log("対象CSVが見つかりません")
        return

    log(f"対象ファイル数: {len(files)}")

    all_summary = []
    all_pairs = []

    total = len(files)

    for i, path in enumerate(files, start=1):
        summary, pairs = process_one_file(path, i, total)
        all_summary.append(summary)
        all_pairs.extend(pairs)

    log("DataFrame化開始")

    summary_df = pd.DataFrame(all_summary)
    pairs_df = pd.DataFrame(all_pairs)

    log(
        f"DataFrame化完了: "
        f"summary_rows={len(summary_df)}, pairs_rows={len(pairs_df)}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = (
        f"{OUTPUT_PREFIX}_summary_"
        f"{CP_METHOD}_{CHANGE_MODEL}_{timestamp}.csv"
    )

    pairs_path = (
        f"{OUTPUT_PREFIX}_pairs_"
        f"{CP_METHOD}_{CHANGE_MODEL}_{timestamp}.csv"
    )

    summary_df.to_csv(summary_path, index=False, encoding=CSV_ENCODING)
    pairs_df.to_csv(pairs_path, index=False, encoding=CSV_ENCODING)

    log(f"summary保存: {summary_path}")
    log(f"pairs保存: {pairs_path}")

    log("処理完了")


if __name__ == "__main__":
    main()