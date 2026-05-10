import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt


# ========================================
# パラメータ（すべてここに集約）
# ========================================

# --- 再現性 ---
RANDOM_SEED = 42
USE_FIXED_SEED = False  # False にすると毎回ランダム

# --- データ ---
NUM_COINS = 200
DATA_DIR = Path.cwd() / "coingecko_by_coin"

# --- 出力 ---
OUTPUT_SUMMARY_CSV = Path.cwd() / "change_point_summary.csv"
OUTPUT_PAIRS_CSV = Path.cwd() / "change_point_pairs.csv"

# --- AR ---
AR_LAGS = 5
AR_MIN_EXTRA_POINTS = 5

# --- Change Point ---
CHANGE_MODEL = "rbf"
CHANGE_PEN_BASE = 1.0
MIN_CHANGE_DISTANCE = 2
CHANGE_MIN_SIZE = 2
CHANGE_JUMP = 1
MIN_RETURNS_FOR_CP = 10

# --- 列候補 ---
PRICE_COL_CANDIDATES = ["price", "Price", "close", "Close"]
TIME_COL_CANDIDATES = ["timestamp", "Timestamp", "date", "Date", "time", "Time"]


# ========================================
# ログ
# ========================================
GLOBAL_START = datetime.now()


def log(msg: str):
    now = datetime.now()
    elapsed = (now - GLOBAL_START).total_seconds()
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:8.2f}s] {msg}")


# ========================================
# CSV一覧
# ========================================
def find_coin_files(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {data_dir}")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {data_dir}")

    log(f"CSVファイル総数: {len(files)}")
    return files


# ========================================
# 列検出
# ========================================
def detect_column(df, candidates, col_type):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"{col_type}列なし: {df.columns}")


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

    df = df.dropna()
    df = df[df["price"] > 0]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    return df


# ========================================
# 対数収益率
# ========================================
def make_log_returns(df):
    prices = df["price"].values
    if len(prices) < 2:
        return pd.DataFrame(columns=["timestamp", "log_return"])

    returns = np.diff(np.log(prices))

    return pd.DataFrame({
        "timestamp": df["timestamp"].iloc[1:].reset_index(drop=True),
        "log_return": returns
    }).dropna().reset_index(drop=True)


# ========================================
# AR
# ========================================
def apply_ar_returns_in_sample(ret_df, symbol):
    series = pd.Series(ret_df["log_return"].values)

    if len(series) <= AR_LAGS + AR_MIN_EXTRA_POINTS:
        log(f"[{symbol}] ARスキップ（データ不足）")
        return None, None

    model = AutoReg(series, lags=AR_LAGS, old_names=False)
    res = model.fit()

    fitted = res.fittedvalues
    actual = series.iloc[AR_LAGS:]

    mse = float(((actual - fitted) ** 2).mean())
    rmse = float(np.sqrt(mse))

    log(f"[{symbol}] AR完了 RMSE={rmse:.6e}")
    return mse, rmse


# ========================================
# 変化点
# ========================================
def detect_change_points(ret_df, symbol):
    returns = ret_df["log_return"].values

    if len(returns) < MIN_RETURNS_FOR_CP:
        log(f"[{symbol}] CPスキップ（短すぎ）")
        return []

    std = returns.std()
    if std < 1e-12:
        log(f"[{symbol}] CPスキップ（分散0）")
        return []

    scaled = (returns - returns.mean()) / (std + 1e-8)

    pen = CHANGE_PEN_BASE * np.log(len(scaled))

    algo = rpt.Pelt(
        model=CHANGE_MODEL,
        min_size=CHANGE_MIN_SIZE,
        jump=CHANGE_JUMP
    ).fit(scaled.reshape(-1, 1))

    bkps = algo.predict(pen=pen)

    bkps = [b for b in bkps if 0 < b < len(returns)]

    # 距離制約
    filtered = []
    for b in sorted(bkps):
        if not filtered or b - filtered[-1] >= MIN_CHANGE_DISTANCE:
            filtered.append(b)

    log(f"[{symbol}] CP検出数={len(filtered)} (raw={len(bkps)})")
    return filtered


# ========================================
# メイン
# ========================================
def main():

    # --- seed ---
    if USE_FIXED_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        log(f"固定seed使用: {RANDOM_SEED}")
    else:
        log("ランダムseed使用")

    # --- ファイル取得 ---
    files = find_coin_files(DATA_DIR)

    if NUM_COINS > len(files):
        raise ValueError("ファイル数不足")

    selected = random.sample(files, NUM_COINS)

    log(f"{NUM_COINS}件ランダム抽出完了")

    results = []

    # ========================================
    # メインループ
    # ========================================
    for i, f in enumerate(selected, 1):

        symbol = f.stem
        t0 = datetime.now()

        log(f"--- ({i}/{NUM_COINS}) {symbol} 開始 ---")

        try:
            df = load_coin_csv(f)
            ret_df = make_log_returns(df)

            mse, rmse = apply_ar_returns_in_sample(ret_df, symbol)
            bkps = detect_change_points(ret_df, symbol)

            results.append({
                "symbol": symbol,
                "rows": len(df),
                "returns": len(ret_df),
                "rmse": rmse,
                "n_cp": len(bkps)
            })

        except Exception as e:
            log(f"[{symbol}] エラー: {e}")

        t1 = datetime.now()
        log(f"--- {symbol} 完了 ({(t1 - t0).total_seconds():.2f}s) ---")

    # ========================================
    # 結果
    # ========================================
    df = pd.DataFrame(results)

    log("全処理完了")
    log(f"総処理時間: {(datetime.now() - GLOBAL_START).total_seconds():.2f}s")

    print(df.sort_values("n_cp", ascending=False).head(20))

    return df


if __name__ == "__main__":
    df = main()