import random
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt


# ========================================
# パラメータ（すべて冒頭）
# ========================================

# ------------------------
# 再現性
# ------------------------
USE_FIXED_SEED = False # False にすると毎回ランダム
RANDOM_SEED = 42

# ------------------------
# 入出力
# ------------------------
DATA_DIR = Path.cwd() / "coingecko_by_coin"
OUTPUT_SUMMARY_CSV = Path.cwd() / "change_point_summary.csv"
OUTPUT_PAIRS_CSV = Path.cwd() / "change_point_pairs.csv"

# ------------------------
# 対象銘柄数
# ------------------------
NUM_COINS = 800

# ------------------------
# AR
# ------------------------
AR_LAGS = 5
AR_MIN_EXTRA_POINTS = 5   # len(series) > AR_LAGS + AR_MIN_EXTRA_POINTS ならAR実行

# ------------------------
# 変化点検知
# 止まりにくいように軽量寄り
# ------------------------
CP_METHOD = "pelt"        # "pelt" または "binseg"

CHANGE_MODEL = "rbf"       # "rbf" より軽い
# CHANGE_MODEL = "l2"       # "rbf" より軽い

CHANGE_PEN_BASE = 2.0     # PELT用
CHANGE_N_BKPS = 10        # Binseg用
MIN_CHANGE_DISTANCE = 3
CHANGE_MIN_SIZE = 5
CHANGE_JUMP = 5
MIN_RETURNS_FOR_CP = 10
MAX_POINTS_FOR_CP = 2000  # 長い系列はここまで間引く

# ------------------------
# 列候補
# ------------------------
PRICE_COL_CANDIDATES = ["price", "Price", "close", "Close"]
TIME_COL_CANDIDATES = ["timestamp", "Timestamp", "date", "Date", "time", "Time"]

# ------------------------
# ログ表示
# ------------------------
SHOW_SELECTED_FILES = True
SHOW_PER_SYMBOL_SUMMARY = True
SHOW_CHANGE_POINTS_DETAIL = True


# ========================================
# グローバル時刻
# ========================================
GLOBAL_START = datetime.now()


# ========================================
# ログ
# ========================================
def log(msg: str):
    now = datetime.now()
    elapsed = (now - GLOBAL_START).total_seconds()
    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:9.2f}s] {msg}",
        flush=True
    )


# ========================================
# CSV一覧を取得
# ========================================
def find_coin_files(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {data_dir}")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {data_dir}")

    return files


# ========================================
# 列名を自動判定
# ========================================
def detect_column(df: pd.DataFrame, candidates: list, col_type: str):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"{col_type}列が見つかりません。候補: {candidates}, 実際の列: {list(df.columns)}"
    )


# ========================================
# CSV読み込み
# ========================================
def load_coin_csv(file_path: Path):
    df = pd.read_csv(file_path)

    time_col = detect_column(df, TIME_COL_CANDIDATES, "時刻")
    price_col = detect_column(df, PRICE_COL_CANDIDATES, "価格")

    df = df[[time_col, price_col]].copy()
    df.columns = ["timestamp", "price"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["timestamp", "price"]).copy()
    df = df[df["price"] > 0].copy()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"有効なデータがありません: {file_path.name}")

    return df


# ========================================
# ファイル名から銘柄名
# ========================================
def symbol_from_filename(file_path: Path):
    return file_path.stem


# ========================================
# 対数収益率
# ========================================
def make_log_returns(df: pd.DataFrame):
    prices = df["price"].values.astype(float)

    if len(prices) < 2:
        return pd.DataFrame(columns=["timestamp", "log_return"])

    log_returns = np.diff(np.log(prices))
    return_times = df["timestamp"].iloc[1:].reset_index(drop=True)

    ret_df = pd.DataFrame({
        "timestamp": return_times,
        "log_return": log_returns
    })

    ret_df = ret_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return ret_df


# ========================================
# AR適用
# ========================================
def apply_ar_returns_in_sample(ret_df: pd.DataFrame, symbol: str, lags: int):
    series = pd.Series(ret_df["log_return"].values.astype(float))

    if len(series) <= lags + AR_MIN_EXTRA_POINTS:
        log(f"[{symbol}] ARスキップ: 対数収益率データが短すぎます (len={len(series)})")
        return None, None

    log(f"[{symbol}] AR開始")
    t0 = time.perf_counter()

    model = AutoReg(series, lags=lags, old_names=False)
    res = model.fit()

    fitted = pd.Series(res.fittedvalues).reset_index(drop=True)
    actual = series.iloc[lags:].reset_index(drop=True)

    min_len = min(len(actual), len(fitted))
    actual = actual.iloc[:min_len]
    fitted = fitted.iloc[:min_len]

    mse = float(((actual.values - fitted.values) ** 2).mean())
    rmse = float(np.sqrt(mse))

    t1 = time.perf_counter()
    log(f"[{symbol}] AR終了: RMSE={rmse:.6e}, time={t1 - t0:.2f}s")
    return mse, rmse


# ========================================
# 変化点の間引き
# ========================================
def filter_change_points_by_distance(bkps, min_distance):
    filtered = []
    for b in sorted(bkps):
        if not filtered or (b - filtered[-1] >= min_distance):
            filtered.append(b)
    return filtered


# ========================================
# 変化点検知
# ========================================
def detect_change_points(
    ret_df: pd.DataFrame,
    symbol: str,
):
    returns_full = ret_df["log_return"].values.astype(float)
    times_full = ret_df["timestamp"].reset_index(drop=True)

    if len(returns_full) < MIN_RETURNS_FOR_CP:
        log(f"[{symbol}] CPスキップ: データが短すぎます (len={len(returns_full)})")
        return [], [], np.array([]), pd.Series(dtype="datetime64[ns]")

    std = returns_full.std()
    if std < 1e-12:
        log(f"[{symbol}] CPスキップ: 分散がほぼ0です")
        return [], [], returns_full, times_full

    # ------------------------
    # 長い系列を間引く
    # ------------------------
    if len(returns_full) > MAX_POINTS_FOR_CP:
        idx = np.linspace(0, len(returns_full) - 1, MAX_POINTS_FOR_CP).astype(int)
        returns = returns_full[idx]
        times = times_full.iloc[idx].reset_index(drop=True)
        log(f"[{symbol}] CP用に {len(returns_full)} -> {len(returns)} 点へ間引き")
    else:
        returns = returns_full
        times = times_full

    std = returns.std()
    returns_scaled = (returns - returns.mean()) / (std + 1e-8)
    signal = returns_scaled.reshape(-1, 1)

    log(
        f"[{symbol}] CP開始: method={CP_METHOD}, model={CHANGE_MODEL}, "
        f"len={len(signal)}, min_size={CHANGE_MIN_SIZE}, jump={CHANGE_JUMP}"
    )
    t0 = time.perf_counter()

    if CP_METHOD.lower() == "pelt":
        pen_value = CHANGE_PEN_BASE * np.log(len(signal))
        algo = rpt.Pelt(
            model=CHANGE_MODEL,
            min_size=CHANGE_MIN_SIZE,
            jump=CHANGE_JUMP
        ).fit(signal)
        raw_bkps = algo.predict(pen=pen_value)
        log(f"[{symbol}] CP raw完了: pen={pen_value:.6f}, raw={raw_bkps}")

    elif CP_METHOD.lower() == "binseg":
        algo = rpt.Binseg(
            model=CHANGE_MODEL,
            jump=CHANGE_JUMP,
            min_size=CHANGE_MIN_SIZE
        ).fit(signal)
        raw_bkps = algo.predict(n_bkps=CHANGE_N_BKPS)
        log(f"[{symbol}] CP raw完了: n_bkps={CHANGE_N_BKPS}, raw={raw_bkps}")

    else:
        raise ValueError(f"未対応のCP_METHODです: {CP_METHOD}")

    # ruptures は末尾 len(signal) を返すので除外
    valid_bkps = [b for b in raw_bkps if 0 < b < len(returns)]
    valid_bkps = filter_change_points_by_distance(valid_bkps, MIN_CHANGE_DISTANCE)

    t1 = time.perf_counter()
    log(f"[{symbol}] CP終了: filtered={valid_bkps}, time={t1 - t0:.2f}s")

    return raw_bkps, valid_bkps, returns_scaled, times


# ========================================
# 変化点候補表示
# ========================================
def summarize_change_points(times: pd.Series, returns_scaled: np.ndarray, symbol: str, valid_bkps):
    print(f"\n=== {symbol} の変化点候補 ===", flush=True)

    if not valid_bkps:
        print("変化点は検出されませんでした。", flush=True)
        return

    for i, b in enumerate(valid_bkps, start=1):
        left_idx = b - 1
        right_idx = b
        if 0 <= left_idx < len(times) and 0 <= right_idx < len(times):
            t_left = times.iloc[left_idx]
            t_right = times.iloc[right_idx]
            v_left = returns_scaled[left_idx]
            v_right = returns_scaled[right_idx]
            print(
                f"{i}. 境界: {t_left.strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"{t_right.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"scaled_return: {v_left:.4f} -> {v_right:.4f}",
                flush=True
            )


# ========================================
# 変化点の日付抽出
# ========================================
def extract_change_point_dates(times: pd.Series, valid_bkps):
    dates = []
    for b in valid_bkps:
        if 0 < b < len(times):
            dates.append(times.iloc[b])
    return dates


# ========================================
# 変化点強度抽出
# ========================================
def extract_change_point_strengths(times: pd.Series, returns_scaled: np.ndarray, valid_bkps):
    strengths = []
    for b in valid_bkps:
        if 0 < b < len(returns_scaled):
            strength = abs(returns_scaled[b] - returns_scaled[b - 1])
        else:
            strength = np.nan
        strengths.append(float(strength) if pd.notna(strength) else np.nan)
    return strengths


# ========================================
# {銘柄, 日付} ペア作成
# ========================================
def build_symbol_date_pairs(change_point_summary: dict):
    pairs = []

    for symbol, rows in change_point_summary.items():
        for row in rows:
            pairs.append({
                "symbol": symbol,
                "date": row["date"],
                "strength": row["strength"],
            })

    pairs.sort(key=lambda x: (x["date"], x["symbol"]))
    return pairs


# ========================================
# 一覧表示
# ========================================
def print_pair_format(change_point_summary: dict):
    print("\n==============================", flush=True)
    print("変化点一覧 {銘柄, 日付}", flush=True)
    print("==============================", flush=True)

    pairs = build_symbol_date_pairs(change_point_summary)

    if not pairs:
        print("データなし", flush=True)
        return

    for row in pairs:
        print(f"{{{row['symbol']}, {row['date'].strftime('%Y-%m-%d %H:%M:%S')}}}", flush=True)


# ========================================
# summary_df 作成
# ========================================
def build_summary_dataframe(results: list):
    rows = []

    for r in results:
        cp_dates = r.get("change_points", [])
        cp_strengths = r.get("change_strengths", [])

        cp_dates_str = " | ".join(
            [d.strftime("%Y-%m-%d %H:%M:%S") for d in cp_dates]
        ) if cp_dates else ""

        cp_strengths_str = " | ".join(
            [f"{x:.6f}" for x in cp_strengths if pd.notna(x)]
        ) if cp_strengths else ""

        rows.append({
            "symbol": r.get("symbol"),
            "rows": r.get("rows"),
            "return_rows": r.get("return_rows"),
            "start_time": r.get("start_time"),
            "end_time": r.get("end_time"),
            "min_price": r.get("min_price"),
            "max_price": r.get("max_price"),
            "last_price": r.get("last_price"),
            "mse": r.get("mse"),
            "rmse": r.get("rmse"),
            "n_change_points": len(cp_dates),
            "change_points": cp_dates_str,
            "change_strengths": cp_strengths_str,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["n_change_points", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return df


# ========================================
# pairs_df 作成
# ========================================
def build_pairs_dataframe(change_point_summary: dict):
    pairs = build_symbol_date_pairs(change_point_summary)

    if not pairs:
        return pd.DataFrame(columns=["symbol", "date", "strength"])

    df_pairs = pd.DataFrame(pairs)
    df_pairs["date"] = pd.to_datetime(df_pairs["date"], errors="coerce")
    df_pairs = df_pairs.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df_pairs


# ========================================
# ランキング表示
# ========================================
def print_rankings(summary_df: pd.DataFrame):
    print("\n==============================", flush=True)
    print("ランキング", flush=True)
    print("==============================", flush=True)

    valid_rmse = summary_df.dropna(subset=["rmse"]).sort_values("rmse")
    if len(valid_rmse) > 0:
        print("\n[AR RMSE が小さい上位10件]", flush=True)
        for i, (_, row) in enumerate(valid_rmse.head(10).iterrows(), start=1):
            print(f"{i}. {row['symbol']}  RMSE={row['rmse']:.6f}", flush=True)

        print("\n[AR RMSE が大きい上位10件]", flush=True)
        for i, (_, row) in enumerate(
            valid_rmse.sort_values("rmse", ascending=False).head(10).iterrows(),
            start=1
        ):
            print(f"{i}. {row['symbol']}  RMSE={row['rmse']:.6f}", flush=True)
    else:
        print("\nRMSEを計算できた銘柄がありません。", flush=True)

    cp_rank = summary_df[summary_df["n_change_points"] > 0].sort_values(
        ["n_change_points", "symbol"], ascending=[False, True]
    )
    if len(cp_rank) > 0:
        print("\n[変化点が多い上位10件]", flush=True)
        for i, (_, row) in enumerate(cp_rank.head(10).iterrows(), start=1):
            print(f"{i}. {row['symbol']}  変化点数={int(row['n_change_points'])}", flush=True)
    else:
        print("\n変化点が1件以上ある銘柄はありません。", flush=True)


# ========================================
# メイン
# ========================================
def main():
    # ------------------------
    # seed
    # ------------------------
    if USE_FIXED_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        log(f"固定seed使用: {RANDOM_SEED}")
    else:
        log("固定seed未使用: 毎回ランダム抽出")

    # ------------------------
    # ファイル一覧
    # ------------------------
    log(f"データディレクトリ: {DATA_DIR}")
    coin_files = find_coin_files(DATA_DIR)
    log(f"CSV総数: {len(coin_files)}")

    if NUM_COINS > len(coin_files):
        raise ValueError(
            f"NUM_COINS={NUM_COINS} は CSV数 {len(coin_files)} を超えています。"
        )

    # ------------------------
    # ランダム抽出
    # ------------------------
    log(f"ランダム抽出開始: {NUM_COINS}件")
    selected_files = random.sample(coin_files, NUM_COINS)
    log("ランダム抽出完了")

    if SHOW_SELECTED_FILES:
        print(f"\n選ばれた{NUM_COINS}ファイル:", flush=True)
        for f in selected_files:
            print(" -", f.name, flush=True)

    dfs = {}
    returns_map = {}
    results = []
    change_point_summary = {}

    # ------------------------
    # データ読み込み
    # ------------------------
    log("=== データ読み込み開始 ===")
    for idx, file_path in enumerate(selected_files, start=1):
        symbol = symbol_from_filename(file_path)
        progress_pct = 100.0 * idx / NUM_COINS

        try:
            log(f"({idx}/{NUM_COINS}, {progress_pct:.1f}%) Loading {file_path.name} 開始")
            t0 = time.perf_counter()

            df = load_coin_csv(file_path)
            ret_df = make_log_returns(df)

            dfs[symbol] = df
            returns_map[symbol] = ret_df

            t1 = time.perf_counter()
            log(
                f"[{symbol}] 読み込み完了: rows={len(df)}, return_rows={len(ret_df)}, "
                f"period={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}, "
                f"time={t1 - t0:.2f}s"
            )
        except Exception as e:
            log(f"[{symbol}] 読み込み失敗: {e}")

    if not dfs:
        log("読み込めた銘柄がありません。終了します。")
        return None, None

    # ------------------------
    # AR + 変化点検知
    # ------------------------
    log("=== AR・変化点検知開始 ===")
    total_symbols = len(dfs)

    for idx, (symbol, df) in enumerate(dfs.items(), start=1):
        progress_pct = 100.0 * idx / total_symbols
        symbol_start = time.perf_counter()

        log(f"--- ({idx}/{total_symbols}, {progress_pct:.1f}%) {symbol} 処理開始 ---")

        ret_df = returns_map[symbol]

        mse = None
        rmse = None
        cp_dates = []
        cp_strengths = []

        try:
            mse, rmse = apply_ar_returns_in_sample(ret_df, symbol, lags=AR_LAGS)
        except Exception as e:
            log(f"[{symbol}] AR失敗: {e}")

        try:
            raw_bkps, valid_bkps, returns_scaled, times = detect_change_points(
                ret_df=ret_df,
                symbol=symbol,
            )

            if SHOW_CHANGE_POINTS_DETAIL:
                summarize_change_points(times, returns_scaled, symbol, valid_bkps)

            cp_dates = extract_change_point_dates(times, valid_bkps)
            cp_strengths = extract_change_point_strengths(times, returns_scaled, valid_bkps)

            change_point_summary[symbol] = [
                {"date": d, "strength": s}
                for d, s in zip(cp_dates, cp_strengths)
            ]
        except Exception as e:
            log(f"[{symbol}] 変化点検知失敗: {e}")
            change_point_summary[symbol] = []

        results.append({
            "symbol": symbol,
            "rows": len(df),
            "return_rows": len(ret_df),
            "start_time": df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "min_price": float(df["price"].min()),
            "max_price": float(df["price"].max()),
            "last_price": float(df["price"].iloc[-1]),
            "mse": mse,
            "rmse": rmse,
            "change_points": cp_dates,
            "change_strengths": cp_strengths,
        })

        symbol_end = time.perf_counter()
        log(
            f"--- {symbol} 処理終了: total_time={symbol_end - symbol_start:.2f}s, "
            f"progress={progress_pct:.1f}% ---"
        )

        if SHOW_PER_SYMBOL_SUMMARY:
            log(
                f"[{symbol}] summary: rows={len(df)}, return_rows={len(ret_df)}, "
                f"rmse={rmse}, n_change_points={len(cp_dates)}"
            )

    # ------------------------
    # DataFrame化
    # ------------------------
    log("DataFrame化開始")
    summary_df = build_summary_dataframe(results)
    pairs_df = build_pairs_dataframe(change_point_summary)
    log(f"DataFrame化完了: summary_rows={len(summary_df)}, pairs_rows={len(pairs_df)}")

    # ------------------------
    # 表示
    # ------------------------
    print("\n==============================", flush=True)
    print("銘柄別サマリー", flush=True)
    print("==============================", flush=True)
    print(summary_df, flush=True)

    print_rankings(summary_df)
    print_pair_format(change_point_summary)

    print("\n==============================", flush=True)
    print("summary_df", flush=True)
    print("==============================", flush=True)
    print(summary_df, flush=True)

    print("\n==============================", flush=True)
    print("pairs_df", flush=True)
    print("==============================", flush=True)
    print(pairs_df, flush=True)

    # ------------------------
    # CSV保存
    # ------------------------
    try:
        summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        log(f"summary_df 保存完了: {OUTPUT_SUMMARY_CSV}")
    except Exception as e:
        log(f"summary_df 保存失敗: {e}")

    try:
        if len(pairs_df) > 0:
            pairs_df_to_save = pairs_df.copy()
            pairs_df_to_save["date"] = pairs_df_to_save["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            pairs_df_to_save.to_csv(OUTPUT_PAIRS_CSV, index=False, encoding="utf-8-sig")
            log(f"pairs_df 保存完了: {OUTPUT_PAIRS_CSV}")
        else:
            log("pairs_df は空のため CSV 保存をスキップしました。")
    except Exception as e:
        log(f"pairs_df 保存失敗: {e}")

    total_elapsed = (datetime.now() - GLOBAL_START).total_seconds()
    log(f"全処理終了: total_elapsed={total_elapsed:.2f}s")

    return summary_df, pairs_df


if __name__ == "__main__":
    summary_df, pairs_df = main()