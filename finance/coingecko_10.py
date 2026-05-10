import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd


# ------------------------
# 設定
# ------------------------
N_SELECT = 1000
DAYS = 90
VS_CURRENCY = "usd"

MIN_SLEEP_SEC = 2.3
JITTER_SEC = 0.7
MAX_RETRIES = 5
TIMEOUT_SEC = 30
RANDOM_SEED = 42

# ★ 保存先
OUTPUT_DIR = Path(r"D:\musashino-university\finance\coingecko_by_coin")

COINS_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"


# ------------------------
# セッション
# ------------------------
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def controlled_sleep():
    time.sleep(MIN_SLEEP_SEC + random.uniform(0, JITTER_SEC))


# ------------------------
# リトライ付きリクエスト
# ------------------------
def request_with_retry(url, params=None):
    wait = MIN_SLEEP_SEC

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT_SEC)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                sleep_sec = wait * 2
                log(f"429 → {sleep_sec:.1f}s待機 ({attempt})")
                time.sleep(sleep_sec)
                wait *= 2
                continue

            if r.status_code in {500, 502, 503, 504}:
                sleep_sec = wait * 2
                log(f"{r.status_code} → {sleep_sec:.1f}s待機 ({attempt})")
                time.sleep(sleep_sec)
                wait *= 2
                continue

            r.raise_for_status()

        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            sleep_sec = wait * 2
            log(f"通信エラー → {sleep_sec:.1f}s待機 ({attempt})")
            time.sleep(sleep_sec)
            wait *= 2

    raise RuntimeError("retry失敗")


# ------------------------
# データ取得
# ------------------------
def fetch_coins_list():
    log("coins/list 取得中...")
    r = request_with_retry(COINS_LIST_URL)
    df = pd.DataFrame(r.json())

    df = df.dropna(subset=["id", "symbol", "name"])
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    log(f"coins数: {len(df)}")
    return df


def fetch_market_chart(coin_id):
    url = MARKET_CHART_URL.format(id=coin_id)
    params = {"vs_currency": VS_CURRENCY, "days": DAYS}

    r = request_with_retry(url, params=params)
    data = r.json()

    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("timestamp")

    return df


# ------------------------
# メイン
# ------------------------
def main():
    random.seed(RANDOM_SEED)
    start_time = time.time()

    # ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coins = fetch_coins_list()

    actual_n = min(N_SELECT, len(coins))
    sampled = coins.sample(n=actual_n, random_state=RANDOM_SEED).reset_index(drop=True)

    failed = []

    # ------------------------
    # ループ
    # ------------------------
    for i, row in sampled.iterrows():
        coin_id = row["id"]
        symbol = row["symbol"]
        name = row["name"]

        try:
            df = fetch_market_chart(coin_id)

            if not df.empty:
                df["coin_id"] = coin_id
                df["symbol"] = symbol
                df["name"] = name

                # ★ ファイル名
                safe_symbol = symbol.replace("/", "_")
                file_path = OUTPUT_DIR / f"{safe_symbol}_{coin_id}.csv"

                df.to_csv(file_path, index=False, encoding="utf-8-sig")

                log(f"保存: {file_path}")

            else:
                failed.append(symbol)

        except Exception as e:
            log(f"{symbol} 失敗: {e}")
            failed.append(symbol)

        # ------------------------
        # 進捗
        # ------------------------
        elapsed = time.time() - start_time
        done = i + 1
        rate = done / actual_n

        avg_time = elapsed / done
        remain = avg_time * (actual_n - done)

        log(
            f"[{done}/{actual_n} | {rate*100:.1f}%] {symbol}\n"
            f"    経過: {fmt_time(elapsed)} / 残り: {fmt_time(remain)}"
        )

        controlled_sleep()

    # ------------------------
    # 失敗一覧
    # ------------------------
    if failed:
        fail_path = OUTPUT_DIR / "failed_coins.csv"
        pd.DataFrame({"symbol": failed}).to_csv(fail_path, index=False)
        log(f"失敗一覧: {fail_path}")

    total_time = time.time() - start_time
    log(f"総処理時間: {fmt_time(total_time)}")


if __name__ == "__main__":
    main()