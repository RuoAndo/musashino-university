# -*- coding: utf-8 -*-
"""
変化点検知 完全版
25組み合わせ実行 + 累計変化点数表示版
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

DATA_DIR = Path(r"D:\musashino-university\finance\coingecko_by_coin")
FILE_PATTERN = "*.csv"

OUTPUT_PREFIX = "change_point"

# 5 × 5 = 25通り
CP_METHODS = ["pelt", "binseg", "bottomup", "window", "dynp"]
CHANGE_MODELS = ["l1", "l2", "rbf", "normal", "linear"]

CHANGE_N_BKPS = 30
CHANGE_PEN_BASE = 0.2

CHANGE_MIN_SIZE = 2
CHANGE_JUMP = 1
WINDOW_WIDTH = 10

MIN_RETURNS_FOR_CP = 10
MAX_POINTS_FOR_CP = 2000

RETURN_SCALE = 100.0

AR_LAGS = 5
DO_AR = True

CSV_ENCODING = "cp932"

USE_FIXED_SEED = False
RANDOM_SEED = 42

CP_PREVIEW_N = 10
CP_TOP_STRENGTH_N = 5


# =========================================================
# 1. ログ関数
# =========================================================

GLOBAL_START = time.time()

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - GLOBAL_START
    print(f"[{now} | +{elapsed:9.2f}s] {msg}", flush=True)


def log_combo(combo_index, total_combos, method, model, msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - GLOBAL_START
    print(
        f"[{now} | +{elapsed:9.2f}s] "
        f"[COMBO {combo_index}/{total_combos}] "
        f"[method={method} | model={model}] {msg}",
        flush=True
    )


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
        return float(np.sqrt(mse))

    except Exception:
        return np.nan


# =========================================================
# 6. 変化点強度
# =========================================================

def calc_cp_strength(returns, cp, width=5):
    before = returns[max(0, cp - width):cp]
    after = returns[cp:min(len(returns), cp + width)]

    if len(before) == 0 or len(after) == 0:
        return np.nan

    return float(abs(np.mean(after) - np.mean(before)))


# =========================================================
# 7. 変化点検知
# =========================================================

def detect_change_points(returns, method, change_model, combo_index, total_combos):
    method = method.lower()

    if len(returns) < MIN_RETURNS_FOR_CP:
        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"CPスキップ: returns={len(returns)} < {MIN_RETURNS_FOR_CP}"
        )
        return []

    signal = returns.copy() * RETURN_SCALE

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

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"CP開始: len={n}, n_bkps={n_bkps}, pen={CHANGE_PEN_BASE}, "
        f"min_size={CHANGE_MIN_SIZE}, jump={CHANGE_JUMP}, scale={RETURN_SCALE}"
    )

    if method == "pelt":
        algo = rpt.Pelt(
            model=change_model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(pen=CHANGE_PEN_BASE)

    elif method == "binseg":
        algo = rpt.Binseg(
            model=change_model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "bottomup":
        algo = rpt.BottomUp(
            model=change_model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "window":
        algo = rpt.Window(
            width=WINDOW_WIDTH,
            model=change_model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    elif method == "dynp":
        algo = rpt.Dynp(
            model=change_model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        )
        bkps = algo.fit(signal_2d).predict(n_bkps=n_bkps)

    else:
        raise ValueError(f"未対応のmethodです: {method}")

    cps = [b for b in bkps if b < n]

    if scale_idx is not None:
        cps = [int(scale_idx[min(cp, len(scale_idx) - 1)]) for cp in cps]

    cps = sorted(set(cps))

    if len(cps) == 0:
        log_combo(combo_index, total_combos, method, change_model, "CP終了: 変化点なし")
    else:
        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"CP終了: n_change_points={len(cps)}, sample={cps[:CP_PREVIEW_N]}"
        )

    return cps


# =========================================================
# 8. 変化点詳細ログ
# =========================================================

def log_change_point_details(
    name,
    returns,
    times,
    cps,
    method,
    change_model,
    combo_index,
    total_combos
):
    if len(cps) == 0:
        log_combo(combo_index, total_combos, method, change_model, f"[{name}] 変化点なし")
        return

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"[{name}] 変化点サンプル先頭{min(CP_PREVIEW_N, len(cps))}件"
    )

    for cp in cps[:CP_PREVIEW_N]:
        cp_time = times[cp] if cp < len(times) else ""
        strength = calc_cp_strength(returns, cp)

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] CP index={cp}, datetime={cp_time}, strength={strength}"
        )

    strength_rows = []

    for cp in cps:
        cp_time = times[cp] if cp < len(times) else ""
        strength = calc_cp_strength(returns, cp)

        strength_rows.append({
            "cp_index": int(cp),
            "datetime": cp_time,
            "strength": strength
        })

    strength_rows = sorted(
        strength_rows,
        key=lambda x: -999999 if pd.isna(x["strength"]) else x["strength"],
        reverse=True
    )

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"[{name}] 強度上位{min(CP_TOP_STRENGTH_N, len(strength_rows))}件"
    )

    for row in strength_rows[:CP_TOP_STRENGTH_N]:
        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] TOP CP index={row['cp_index']}, datetime={row['datetime']}, strength={row['strength']}"
        )


# =========================================================
# 9. 1ファイル処理
# =========================================================

def process_one_file(path, index, total, method, change_model, combo_index, total_combos):
    name = path.stem
    start = time.time()

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"--- {name} 処理開始: file={index}/{total} ---"
    )

    summary = {
        "method": method,
        "model": change_model,
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

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] 読込完了: rows={len(df)}, returns={len(returns)}"
        )

        ar_start = time.time()
        rmse = run_ar_model(returns)
        summary["rmse"] = rmse

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] AR終了: RMSE={rmse}, time={time.time() - ar_start:.2f}s"
        )

        cp_start = time.time()

        cps = detect_change_points(
            returns,
            method,
            change_model,
            combo_index,
            total_combos
        )

        summary["n_change_points"] = len(cps)

        log_change_point_details(
            name=name,
            returns=returns,
            times=times,
            cps=cps,
            method=method,
            change_model=change_model,
            combo_index=combo_index,
            total_combos=total_combos
        )

        for cp in cps:
            cp_time = times[cp] if cp < len(times) else ""
            strength = calc_cp_strength(returns, cp)

            pairs.append({
                "method": method,
                "model": change_model,
                "symbol": name,
                "file": path.name,
                "cp_index": int(cp),
                "datetime": cp_time,
                "strength": strength
            })

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] CP処理終了: current_file_cps={len(cps)}, time={time.time() - cp_start:.2f}s"
        )

    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"[{name}] 処理失敗: {e}"
        )

    progress = index / total * 100

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"--- {name} 処理終了: current_file_cps={summary['n_change_points']}, "
        f"progress={progress:.1f}%, total_time={time.time() - start:.2f}s ---"
    )

    return summary, pairs


# =========================================================
# 10. 1組み合わせ処理
# =========================================================

def process_one_combination(
    files,
    method,
    change_model,
    combo_index,
    total_combos,
    grand_total_cps
):
    combo_start = time.time()

    log_combo(combo_index, total_combos, method, change_model, "=" * 70)
    log_combo(combo_index, total_combos, method, change_model, "組み合わせ処理開始")

    all_summary = []
    all_pairs = []

    total_files = len(files)
    combo_total_cps = 0

    for i, path in enumerate(files, start=1):
        summary, pairs = process_one_file(
            path=path,
            index=i,
            total=total_files,
            method=method,
            change_model=change_model,
            combo_index=combo_index,
            total_combos=total_combos
        )

        all_summary.append(summary)
        all_pairs.extend(pairs)

        current_file_cps = int(summary.get("n_change_points", 0))
        combo_total_cps += current_file_cps
        grand_total_cps += current_file_cps

        log_combo(
            combo_index,
            total_combos,
            method,
            change_model,
            f"累計変化点数: current_file={current_file_cps}, "
            f"current_combo={combo_total_cps}, all_combos={grand_total_cps}"
        )

    summary_df = pd.DataFrame(all_summary)
    pairs_df = pd.DataFrame(all_pairs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = f"{OUTPUT_PREFIX}_summary_{method}_{change_model}_{timestamp}.csv"
    pairs_path = f"{OUTPUT_PREFIX}_pairs_{method}_{change_model}_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False, encoding=CSV_ENCODING)
    pairs_df.to_csv(pairs_path, index=False, encoding=CSV_ENCODING)

    log_combo(combo_index, total_combos, method, change_model, f"summary保存: {summary_path}")
    log_combo(combo_index, total_combos, method, change_model, f"pairs保存: {pairs_path}")

    log_combo(
        combo_index,
        total_combos,
        method,
        change_model,
        f"組み合わせ処理終了: combo_total_cps={combo_total_cps}, "
        f"all_combos_cps={grand_total_cps}, time={time.time() - combo_start:.2f}s"
    )

    return summary_df, pairs_df, grand_total_cps


# =========================================================
# 11. メイン処理
# =========================================================

def main():
    if USE_FIXED_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    log("処理開始")
    log(f"DATA_DIR={DATA_DIR}")
    log(f"FILE_PATTERN={FILE_PATTERN}")

    files = sorted(DATA_DIR.glob(FILE_PATTERN))

    files = [
        f for f in files
        if not f.name.startswith(OUTPUT_PREFIX)
    ]

    if not files:
        log("対象CSVが見つかりません")
        return

    combinations = []

    for method in CP_METHODS:
        for model in CHANGE_MODELS:
            combinations.append((method, model))

    total_combos = len(combinations)

    log(f"対象ファイル数: {len(files)}")
    log(f"試行する組み合わせ数: {total_combos}")
    log(f"CP_METHODS={CP_METHODS}")
    log(f"CHANGE_MODELS={CHANGE_MODELS}")

    master_summary_list = []
    master_pairs_list = []

    grand_total_cps = 0

    for combo_index, (method, change_model) in enumerate(combinations, start=1):
        try:
            summary_df, pairs_df, grand_total_cps = process_one_combination(
                files=files,
                method=method,
                change_model=change_model,
                combo_index=combo_index,
                total_combos=total_combos,
                grand_total_cps=grand_total_cps
            )

            master_summary_list.append(summary_df)
            master_pairs_list.append(pairs_df)

        except Exception as e:
            log_combo(
                combo_index,
                total_combos,
                method,
                change_model,
                f"組み合わせ全体で失敗: {e}"
            )

    log("全組み合わせの個別処理完了")

    if master_summary_list:
        master_summary_df = pd.concat(master_summary_list, ignore_index=True)
    else:
        master_summary_df = pd.DataFrame()

    if master_pairs_list:
        master_pairs_df = pd.concat(master_pairs_list, ignore_index=True)
    else:
        master_pairs_df = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    master_summary_path = f"{OUTPUT_PREFIX}_summary_ALL_METHODS_ALL_MODELS_{timestamp}.csv"
    master_pairs_path = f"{OUTPUT_PREFIX}_pairs_ALL_METHODS_ALL_MODELS_{timestamp}.csv"

    master_summary_df.to_csv(master_summary_path, index=False, encoding=CSV_ENCODING)
    master_pairs_df.to_csv(master_pairs_path, index=False, encoding=CSV_ENCODING)

    log(f"統合summary保存: {master_summary_path}")
    log(f"統合pairs保存: {master_pairs_path}")
    log(f"全体累計変化点数: {grand_total_cps}")

    if len(master_summary_df) > 0:
        try:
            agg = (
                master_summary_df
                .groupby(["method", "model"], dropna=False)
                .agg(
                    files=("file", "count"),
                    ok_count=("status", lambda x: (x == "ok").sum()),
                    error_count=("status", lambda x: (x == "error").sum()),
                    total_change_points=("n_change_points", "sum"),
                    mean_change_points=("n_change_points", "mean"),
                    max_change_points=("n_change_points", "max"),
                    mean_rmse=("rmse", "mean")
                )
                .reset_index()
            )

            agg_path = f"{OUTPUT_PREFIX}_summary_AGG_ALL_METHODS_ALL_MODELS_{timestamp}.csv"
            agg.to_csv(agg_path, index=False, encoding=CSV_ENCODING)

            log(f"組み合わせ別集計保存: {agg_path}")
            print(agg, flush=True)

        except Exception as e:
            log(f"組み合わせ別集計に失敗: {e}")

    log("処理完了")


if __name__ == "__main__":
    main()