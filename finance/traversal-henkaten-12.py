# -*- coding: utf-8 -*-
"""
変化点検知 完全版
25組み合わせ実行 + utf-8-sig保存 + 統合summary/pairs + 日別プロット

主な修正:
- cp932保存を廃止し、utf-8-sigで保存
- 絵文字、₸、中国語などを含む銘柄名でも落ちにくい
- 個別保存に失敗しても、統合用リストは保持
- 最後に統合summary、統合pairs、日別変化点数CSV/PNGを保存
"""

import time
import random
import warnings
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ruptures as rpt
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")


# =========================================================
# 0. 設定
# =========================================================

DATA_DIR = Path(r"D:\musashino-university\finance\coingecko_by_coin")
FILE_PATTERN = "*.csv"

OUTPUT_DIR = Path(r"D:\musashino-university\finance\change_point_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "daily_change_point_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "change_point"

CP_METHODS = ["pelt", "binseg", "bottomup", "window", "dynp"]
CHANGE_MODELS = ["l1", "l2", "rbf", "normal", "linear"]

CHANGE_N_BKPS = 30
CHANGE_PEN_BASE = 0.2

CHANGE_MIN_SIZE = 2
CHANGE_JUMP = 1
WINDOW_WIDTH = 10

MIN_RETURNS_FOR_CP = 10
MAX_POINTS_FOR_CP = 2000

# dynpは重いので上限を別にする
DYN_MAX_POINTS_FOR_CP = 500

AR_LAGS = 5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

START_TIME = datetime.now()
RUN_ID = START_TIME.strftime("%Y%m%d_%H%M%S")


# =========================================================
# 1. ログ関数
# =========================================================

def format_elapsed(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:2d}時間{m:02d}分{s:02d}秒"
    return f"{m:4d}分{s:02d}秒"


def log(msg):
    now = datetime.now()
    elapsed = (now - START_TIME).total_seconds()
    print(
        f"[{now:%Y-%m-%d %H:%M:%S} | +{format_elapsed(elapsed)}] {msg}",
        flush=True
    )


# =========================================================
# 2. 補助関数
# =========================================================

def safe_filename(name, max_len=120):
    """
    Windowsで危険なファイル名文字を置換。
    絵文字などはそのまま残してもutf-8-sig保存なら問題ないが、
    ファイル名として長すぎる場合に備えて短縮する。
    """
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._ ")

    if not name:
        name = "unknown"

    return name[:max_len]


def read_csv_auto(path):
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


def detect_datetime_column(df):
    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "Date",
        "Datetime",
        "Timestamp",
        "snapped_at",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def detect_price_column(df):
    candidates = [
        "price",
        "Price",
        "close",
        "Close",
        "market_price",
        "current_price",
        "usd",
        "value",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        return numeric_cols[0]

    return None


def load_price_series(file_path):
    df = read_csv_auto(file_path)

    if df.empty:
        raise ValueError("CSVが空です")

    dt_col = detect_datetime_column(df)
    price_col = detect_price_column(df)

    if price_col is None:
        raise ValueError("価格列が見つかりません")

    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce")
    else:
        dt = pd.Series(pd.RangeIndex(len(df)), index=df.index)

    price = pd.to_numeric(df[price_col], errors="coerce")

    out = pd.DataFrame({
        "datetime": dt,
        "price": price
    })

    out = out.dropna(subset=["price"])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["price"])

    if dt_col is not None:
        out = out.dropna(subset=["datetime"])
        out = out.drop_duplicates(subset=["datetime"])
        out = out.sort_values("datetime")

    out = out.reset_index(drop=True)

    if len(out) < 3:
        raise ValueError("有効な価格データが少なすぎます")

    return out


def compute_returns(price_df):
    prices = price_df["price"].astype(float)

    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.where(prices > 0)
    log_price = np.log(prices)

    returns = log_price.diff()

    out = price_df.copy()
    out["return"] = returns

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["return"])
    out = out.reset_index(drop=True)

    return out


def fit_ar_rmse(returns, lags=5):
    x = np.asarray(returns, dtype=float)

    if len(x) <= lags + 5:
        return np.nan

    try:
        model = AutoReg(x, lags=lags, old_names=False)
        res = model.fit()
        pred = res.predict(start=lags, end=len(x) - 1)
        actual = x[lags:]

        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        return float(rmse)

    except Exception:
        return np.nan


def downsample_signal(signal, datetimes, max_points):
    n = len(signal)

    if n <= max_points:
        return signal, datetimes, np.arange(n)

    idx = np.linspace(0, n - 1, max_points).astype(int)
    idx = np.unique(idx)

    return signal[idx], datetimes.iloc[idx].reset_index(drop=True), idx


def make_signal(returns):
    signal = np.asarray(returns, dtype=float)
    signal = signal.reshape(-1, 1)

    std = float(np.std(signal))

    if std <= 0 or np.isnan(std):
        raise ValueError("returnsの標準偏差が0です")

    signal = signal / std

    return signal


def run_change_point_detection(method, model, signal):
    n = len(signal)

    if n < MIN_RETURNS_FOR_CP:
        return []

    if method == "dynp" and n > DYN_MAX_POINTS_FOR_CP:
        raise RuntimeError(
            f"dynpスキップ: len={n} > DYN_MAX_POINTS_FOR_CP={DYN_MAX_POINTS_FOR_CP}"
        )

    n_bkps = min(CHANGE_N_BKPS, max(1, n // max(CHANGE_MIN_SIZE, 2) - 1))

    if method == "pelt":
        algo = rpt.Pelt(
            model=model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)

        pen = CHANGE_PEN_BASE * np.log(max(n, 2))
        cps = algo.predict(pen=pen)

    elif method == "binseg":
        algo = rpt.Binseg(
            model=model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)

        cps = algo.predict(n_bkps=n_bkps)

    elif method == "bottomup":
        algo = rpt.BottomUp(
            model=model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)

        cps = algo.predict(n_bkps=n_bkps)

    elif method == "window":
        width = min(WINDOW_WIDTH, max(2, n // 2))

        algo = rpt.Window(
            width=width,
            model=model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)

        cps = algo.predict(n_bkps=n_bkps)

    elif method == "dynp":
        algo = rpt.Dynp(
            model=model,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)

        cps = algo.predict(n_bkps=n_bkps)

    else:
        raise ValueError(f"未知のmethodです: {method}")

    # rupturesは最後にnを返すので除外
    cps = [int(cp) for cp in cps if int(cp) < n]

    return cps


def estimate_strength(signal, cp, window=5):
    n = len(signal)

    left_start = max(0, cp - window)
    left_end = cp
    right_start = cp
    right_end = min(n, cp + window)

    if left_end <= left_start or right_end <= right_start:
        return np.nan

    left_mean = float(np.mean(signal[left_start:left_end]))
    right_mean = float(np.mean(signal[right_start:right_end]))

    return abs(right_mean - left_mean)


def save_csv_safe(df, path):
    """
    cp932ではなくutf-8-sigで保存。
    Excelでも比較的開きやすく、Unicode文字で落ちにくい。
    """
    df.to_csv(path, index=False, encoding="utf-8-sig")


def extract_daily_counts_from_pairs(df_pairs):
    if df_pairs.empty:
        return pd.DataFrame(columns=["date", "change_point_count"])

    if "cp_datetime" not in df_pairs.columns:
        return pd.DataFrame(columns=["date", "change_point_count"])

    dt = pd.to_datetime(df_pairs["cp_datetime"], errors="coerce").dropna()

    if dt.empty:
        return pd.DataFrame(columns=["date", "change_point_count"])

    tmp = pd.DataFrame({"datetime": dt})
    tmp["date"] = tmp["datetime"].dt.date

    daily = (
        tmp.groupby("date")
        .size()
        .reset_index(name="change_point_count")
        .sort_values("date")
    )

    return daily


def plot_daily_counts(daily_df, out_png, title):
    if daily_df.empty:
        return False

    plt.figure(figsize=(12, 5))
    plt.bar(daily_df["date"].astype(str), daily_df["change_point_count"])
    plt.xlabel("date")
    plt.ylabel("change point count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return True


# =========================================================
# 3. ファイル検索
# =========================================================

log("ファイル検索開始")

files = sorted(DATA_DIR.glob(FILE_PATTERN))

if not files:
    raise FileNotFoundError(f"対象ファイルが見つかりません: {DATA_DIR / FILE_PATTERN}")

log(f"対象ファイル数: {len(files)}")
log(f"出力先: {OUTPUT_DIR.resolve()}")


# =========================================================
# 4. 全組み合わせ処理
# =========================================================

all_summary_rows = []
all_pair_rows = []

total_combos = len(CP_METHODS) * len(CHANGE_MODELS)
combo_index = 0

for method in CP_METHODS:
    for model in CHANGE_MODELS:
        combo_index += 1

        combo_start = datetime.now()

        log("============================================================")
        log(f"[COMBO {combo_index}/{total_combos}] method={method} | model={model} 開始")
        log("============================================================")

        combo_summary_rows = []
        combo_pair_rows = []

        current_combo_cps = 0

        for file_i, file_path in enumerate(files, start=1):
            file_start = datetime.now()

            raw_symbol = file_path.stem
            symbol = raw_symbol

            log(
                f"[COMBO {combo_index}/{total_combos}] "
                f"[method={method} | model={model}] "
                f"--- {symbol} 処理開始: file={file_i}/{len(files)} ---"
            )

            try:
                price_df = load_price_series(file_path)
                ret_df = compute_returns(price_df)

                rows = len(price_df)
                returns_count = len(ret_df)

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"[{symbol}] 読込完了: rows={rows}, returns={returns_count}"
                )

                if returns_count < MIN_RETURNS_FOR_CP:
                    raise ValueError(
                        f"returnsが少なすぎます: {returns_count} < {MIN_RETURNS_FOR_CP}"
                    )

                ar_start = datetime.now()
                rmse = fit_ar_rmse(ret_df["return"], lags=AR_LAGS)
                ar_elapsed = (datetime.now() - ar_start).total_seconds()

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"[{symbol}] AR終了: RMSE={rmse}, time={format_elapsed(ar_elapsed)}"
                )

                signal_full = make_signal(ret_df["return"])

                if method == "dynp":
                    max_points = DYN_MAX_POINTS_FOR_CP
                else:
                    max_points = MAX_POINTS_FOR_CP

                signal, ds_datetimes, ds_idx = downsample_signal(
                    signal_full,
                    ret_df["datetime"],
                    max_points=max_points
                )

                cp_start = datetime.now()

                try:
                    cps = run_change_point_detection(method, model, signal)
                except RuntimeError as e:
                    if "dynpスキップ" in str(e):
                        log(
                            f"[COMBO {combo_index}/{total_combos}] "
                            f"[method={method} | model={model}] {e}"
                        )
                        cps = []
                    else:
                        raise

                cp_elapsed = (datetime.now() - cp_start).total_seconds()

                if not cps:
                    log(
                        f"[COMBO {combo_index}/{total_combos}] "
                        f"[method={method} | model={model}] "
                        f"[{symbol}] 変化点なし"
                    )

                cp_datetimes = []
                cp_indices_original = []
                cp_strengths = []

                for cp in cps:
                    cp = int(cp)

                    if cp < 0 or cp >= len(ds_datetimes):
                        continue

                    cp_dt = ds_datetimes.iloc[cp]
                    orig_idx = int(ds_idx[cp])
                    strength = estimate_strength(signal, cp)

                    cp_datetimes.append(cp_dt)
                    cp_indices_original.append(orig_idx)
                    cp_strengths.append(strength)

                    combo_pair_rows.append({
                        "run_id": RUN_ID,
                        "method": method,
                        "model": model,
                        "symbol": symbol,
                        "file_name": file_path.name,
                        "cp_index_downsampled": cp,
                        "cp_index_original": orig_idx,
                        "cp_datetime": cp_dt,
                        "cp_strength": strength,
                        "rows": rows,
                        "returns": returns_count,
                        "rmse": rmse,
                    })

                current_file_cps = len(cp_datetimes)
                current_combo_cps += current_file_cps

                summary_row = {
                    "run_id": RUN_ID,
                    "method": method,
                    "model": model,
                    "symbol": symbol,
                    "file_name": file_path.name,
                    "rows": rows,
                    "returns": returns_count,
                    "start_datetime": ret_df["datetime"].iloc[0],
                    "end_datetime": ret_df["datetime"].iloc[-1],
                    "min_price": float(price_df["price"].min()),
                    "max_price": float(price_df["price"].max()),
                    "last_price": float(price_df["price"].iloc[-1]),
                    "rmse": rmse,
                    "change_point_count": current_file_cps,
                    "cp_datetimes": " | ".join([str(x) for x in cp_datetimes]),
                    "cp_indices_original": " | ".join([str(x) for x in cp_indices_original]),
                    "cp_strengths": " | ".join([str(x) for x in cp_strengths]),
                    "status": "ok",
                    "error": "",
                }

                combo_summary_rows.append(summary_row)

                file_elapsed = (datetime.now() - file_start).total_seconds()
                progress = file_i / len(files) * 100

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"[{symbol}] CP処理終了: current_file_cps={current_file_cps}, "
                    f"time={format_elapsed(cp_elapsed)}"
                )

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"--- {symbol} 処理終了: current_file_cps={current_file_cps}, "
                    f"progress={progress:.1f}%, total_time={format_elapsed(file_elapsed)} ---"
                )

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"累計変化点数: current_file={current_file_cps}, "
                    f"current_combo={current_combo_cps}, "
                    f"all_combos={len(all_pair_rows) + len(combo_pair_rows)}"
                )

            except Exception as e:
                file_elapsed = (datetime.now() - file_start).total_seconds()

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"[{symbol}] ファイル処理失敗: {e}"
                )

                combo_summary_rows.append({
                    "run_id": RUN_ID,
                    "method": method,
                    "model": model,
                    "symbol": symbol,
                    "file_name": file_path.name,
                    "rows": np.nan,
                    "returns": np.nan,
                    "start_datetime": "",
                    "end_datetime": "",
                    "min_price": np.nan,
                    "max_price": np.nan,
                    "last_price": np.nan,
                    "rmse": np.nan,
                    "change_point_count": 0,
                    "cp_datetimes": "",
                    "cp_indices_original": "",
                    "cp_strengths": "",
                    "status": "error",
                    "error": str(e),
                })

                log(
                    f"[COMBO {combo_index}/{total_combos}] "
                    f"[method={method} | model={model}] "
                    f"--- {symbol} 処理終了: error, total_time={format_elapsed(file_elapsed)} ---"
                )

        # 重要:
        # 保存の前に統合リストへ追加する。
        # これにより、個別CSV保存が失敗しても全体結果は失われない。
        all_summary_rows.extend(combo_summary_rows)
        all_pair_rows.extend(combo_pair_rows)

        combo_summary_df = pd.DataFrame(combo_summary_rows)
        combo_pairs_df = pd.DataFrame(combo_pair_rows)

        combo_summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary_{method}_{model}_{RUN_ID}.csv"
        combo_pairs_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_pairs_{method}_{model}_{RUN_ID}.csv"

        try:
            save_csv_safe(combo_summary_df, combo_summary_path)
            log(f"[COMBO {combo_index}/{total_combos}] summary保存: {combo_summary_path.name}")
        except Exception as e:
            log(f"[COMBO {combo_index}/{total_combos}] summary保存失敗: {e}")

        try:
            save_csv_safe(combo_pairs_df, combo_pairs_path)
            log(f"[COMBO {combo_index}/{total_combos}] pairs保存: {combo_pairs_path.name}")
        except Exception as e:
            log(f"[COMBO {combo_index}/{total_combos}] pairs保存失敗: {e}")

        combo_elapsed = (datetime.now() - combo_start).total_seconds()

        log(
            f"[COMBO {combo_index}/{total_combos}] "
            f"method={method} | model={model} 完了: "
            f"combo_cps={len(combo_pair_rows)}, "
            f"elapsed={format_elapsed(combo_elapsed)}"
        )


# =========================================================
# 5. 統合保存
# =========================================================

log("全組み合わせの個別処理完了")

all_summary_df = pd.DataFrame(all_summary_rows)
all_pairs_df = pd.DataFrame(all_pair_rows)

all_summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary_ALL_METHODS_ALL_MODELS_{RUN_ID}.csv"
all_pairs_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_pairs_ALL_METHODS_ALL_MODELS_{RUN_ID}.csv"

save_csv_safe(all_summary_df, all_summary_path)
log(f"統合summary保存: {all_summary_path.name}")

save_csv_safe(all_pairs_df, all_pairs_path)
log(f"統合pairs保存: {all_pairs_path.name}")

log(f"全体累計変化点数: {len(all_pairs_df)}")

if all_pairs_df.empty:
    log("変化点一覧は空です")
else:
    log("変化点一覧あり")


# =========================================================
# 6. 集計保存
# =========================================================

if not all_summary_df.empty:
    agg_summary = (
        all_summary_df
        .groupby(["method", "model"], dropna=False)
        .agg(
            files=("file_name", "count"),
            ok_files=("status", lambda x: int((x == "ok").sum())),
            error_files=("status", lambda x: int((x == "error").sum())),
            total_change_points=("change_point_count", "sum"),
            mean_rmse=("rmse", "mean"),
        )
        .reset_index()
        .sort_values(["total_change_points", "mean_rmse"], ascending=[False, True])
    )

    agg_summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary_AGG_ALL_METHODS_ALL_MODELS_{RUN_ID}.csv"
    save_csv_safe(agg_summary, agg_summary_path)
    log(f"統合集計summary保存: {agg_summary_path.name}")

    log("===== 組み合わせ別ランキング =====")
    print(agg_summary.to_string(index=False), flush=True)


# =========================================================
# 7. 日別変化点数CSV/PNG
# =========================================================

daily_df = extract_daily_counts_from_pairs(all_pairs_df)

daily_csv_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_daily_counts_ALL_METHODS_ALL_MODELS_{RUN_ID}.csv"
save_csv_safe(daily_df, daily_csv_path)
log(f"日別変化点数CSV保存: {daily_csv_path.name}")

daily_png_path = PLOT_DIR / f"{OUTPUT_PREFIX}_daily_counts_ALL_METHODS_ALL_MODELS_{RUN_ID}.png"

if plot_daily_counts(
    daily_df,
    daily_png_path,
    title="Daily Change Points - ALL METHODS / ALL MODELS"
):
    log(f"日別変化点数プロット保存: {daily_png_path.name}")
else:
    log("日別変化点数プロットは作成されませんでした")


# =========================================================
# 8. 最終表示
# =========================================================

total_elapsed = (datetime.now() - START_TIME).total_seconds()

log("====================================")
log("処理結果")
log(f"summary行数: {len(all_summary_df)}")
log(f"pairs行数: {len(all_pairs_df)}")
log(f"全体累計変化点数: {len(all_pairs_df)}")
log(f"出力先: {OUTPUT_DIR.resolve()}")
log(f"総処理時間: {format_elapsed(total_elapsed)}")
log("完了")