import argparse
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# ------------------------
# デフォルト設定
# ------------------------
DEFAULT_N_SELECT = 100
DEFAULT_DAYS = 90
DEFAULT_VS_CURRENCY = "usd"

DEFAULT_MIN_SLEEP_SEC = 2.3
DEFAULT_JITTER_SEC = 0.7
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_RANDOM_SEED = 42

DEFAULT_OUTPUT_DIR = Path("./coingecko_by_coin")

COINS_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"


# ------------------------
# 引数
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CoinGeckoからランダムに暗号通貨を取得し、銘柄ごとにCSV保存する"
    )
    parser.add_argument("--n-select", type=int, default=DEFAULT_N_SELECT, help="取得する銘柄数")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="取得日数")
    parser.add_argument("--vs-currency", type=str, default=DEFAULT_VS_CURRENCY, help="基準通貨")
    parser.add_argument("--min-sleep-sec", type=float, default=DEFAULT_MIN_SLEEP_SEC, help="最小待機秒数")
    parser.add_argument("--jitter-sec", type=float, default=DEFAULT_JITTER_SEC, help="待機ジッター秒数")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="最大リトライ回数")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="HTTPタイムアウト秒数")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="乱数シード")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="出力先ディレクトリ")
    parser.add_argument("--yes", action="store_true", help="確認なしで実行する")
    return parser.parse_args()


# ------------------------
# セッション
# ------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; stable-demo/1.0)"
})


# ------------------------
# 共通関数
# ------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fmt_time(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


def safe_name(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._")
    return text[:120] if text else "unknown"


def format_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def calc_days(df: pd.DataFrame) -> float:
    if df.empty or "timestamp" not in df.columns:
        return 0.0
    delta = df["timestamp"].max() - df["timestamp"].min()
    return round(delta.total_seconds() / 86400, 2)


def controlled_sleep(min_sleep_sec: float, jitter_sec: float):
    time.sleep(min_sleep_sec + random.uniform(0, jitter_sec))


# ------------------------
# 実行前確認
# ------------------------
def confirm_execution(args, output_dir: Path):
    print("\n=== 実行パラメータ確認 ===")
    print(f"取得銘柄数        : {args.n_select}")
    print(f"取得日数          : {args.days}")
    print(f"通貨              : {args.vs_currency}")
    print(f"最小待機時間      : {args.min_sleep_sec} 秒")
    print(f"ジッター          : ±{args.jitter_sec} 秒")
    print(f"最大リトライ回数  : {args.max_retries}")
    print(f"タイムアウト      : {args.timeout_sec} 秒")
    print(f"乱数シード        : {args.random_seed}")
    print(f"保存先            : {output_dir.resolve()}")
    print("=========================\n")

    ans = input("実行しますか？ (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        print("中止しました。")
        return False

    print("実行開始します...\n")
    return True


# ------------------------
# リトライ付きリクエスト
# ------------------------
def request_with_retry(url, params, timeout_sec, max_retries, min_sleep_sec):
    wait = min_sleep_sec

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout_sec)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_sec = max(float(retry_after), wait * 2)
                    except ValueError:
                        sleep_sec = wait * 2
                else:
                    sleep_sec = wait * 2

                log(f"429 Too Many Requests → {sleep_sec:.1f}s待機 ({attempt}/{max_retries})")
                time.sleep(sleep_sec)
                wait *= 2
                continue

            if r.status_code in {500, 502, 503, 504}:
                sleep_sec = wait * 2
                log(f"{r.status_code} サーバーエラー → {sleep_sec:.1f}s待機 ({attempt}/{max_retries})")
                time.sleep(sleep_sec)
                wait *= 2
                continue

            r.raise_for_status()

        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(f"request失敗: {e}") from e

            sleep_sec = wait * 2
            log(f"通信エラー → {sleep_sec:.1f}s待機 ({attempt}/{max_retries}) : {e}")
            time.sleep(sleep_sec)
            wait *= 2

    raise RuntimeError("retry失敗")


# ------------------------
# データ取得
# ------------------------
def fetch_coins_list(timeout_sec, max_retries, min_sleep_sec):
    log("coins/list 取得中...")
    r = request_with_retry(
        COINS_LIST_URL,
        params=None,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        min_sleep_sec=min_sleep_sec,
    )
    df = pd.DataFrame(r.json())

    required_cols = ["id", "symbol", "name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"coins/list に必要列がありません: {col}")

    df = df.dropna(subset=required_cols)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    log(f"coins数: {len(df)}")
    return df


def fetch_market_chart(coin_id, vs_currency, days, timeout_sec, max_retries, min_sleep_sec):
    url = MARKET_CHART_URL.format(id=coin_id)
    params = {"vs_currency": vs_currency, "days": days}

    r = request_with_retry(
        url,
        params=params,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        min_sleep_sec=min_sleep_sec,
    )
    data = r.json()

    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return df


# ------------------------
# 保存
# ------------------------
def save_coin_csv(df, coin_id, symbol, name, output_dir: Path):
    file_name = f"{safe_name(coin_id)}_{safe_name(symbol)}.csv"
    out_path = output_dir / file_name

    save_df = df.copy()
    save_df["coin_id"] = coin_id
    save_df["symbol"] = symbol
    save_df["name"] = name

    save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, len(save_df)


# ------------------------
# メイン
# ------------------------
def main():
    args = parse_args()

    if args.n_select <= 0:
        raise ValueError("--n-select は 1 以上にしてください")
    if args.days <= 0:
        raise ValueError("--days は 1 以上にしてください")
    if args.min_sleep_sec < 0:
        raise ValueError("--min-sleep-sec は 0 以上にしてください")
    if args.jitter_sec < 0:
        raise ValueError("--jitter-sec は 0 以上にしてください")
    if args.max_retries <= 0:
        raise ValueError("--max-retries は 1 以上にしてください")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec は 1 以上にしてください")

    output_dir = Path(args.output_dir)

    if not args.yes:
        if not confirm_execution(args, output_dir):
            return

    random.seed(args.random_seed)
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"summary_{args.n_select}_{ts}.csv"
    failed_csv = output_dir / f"failed_{args.n_select}_{ts}.csv"
    empty_csv = output_dir / f"empty_{args.n_select}_{ts}.csv"

    log("処理開始")
    coins = fetch_coins_list(
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        min_sleep_sec=args.min_sleep_sec,
    )

    actual_n = min(args.n_select, len(coins))
    if actual_n < args.n_select:
        log(f"指定件数 {args.n_select} は多すぎるため、{actual_n} 件に調整します")

    sampled = coins.sample(n=actual_n, random_state=args.random_seed).reset_index(drop=True)

    success_count = 0
    fail_count = 0
    empty_count = 0

    saved_records = []
    failed_records = []
    empty_records = []

    for i, row in sampled.iterrows():
        loop_start = time.time()

        coin_id = row["id"]
        symbol = str(row["symbol"])
        name = str(row["name"])

        try:
            df = fetch_market_chart(
                coin_id=coin_id,
                vs_currency=args.vs_currency,
                days=args.days,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                min_sleep_sec=args.min_sleep_sec,
            )

            if df.empty:
                empty_count += 1
                empty_records.append({
                    "coin_id": coin_id,
                    "symbol": symbol,
                    "name": name,
                })
                log(f"{symbol} ({coin_id}) は価格データなし")

            else:
                out_path, n_rows = save_coin_csv(df, coin_id, symbol, name, output_dir)
                success_count += 1

                file_size_bytes = out_path.stat().st_size
                file_size_text = format_file_size(file_size_bytes)
                actual_days = calc_days(df)

                saved_records.append({
                    "coin_id": coin_id,
                    "symbol": symbol,
                    "name": name,
                    "rows": n_rows,
                    "days": actual_days,
                    "file_size_bytes": file_size_bytes,
                    "file_size_human": file_size_text,
                    "file_path": str(out_path.resolve()),
                })

                log(
                    f"{symbol} 保存完了: {out_path.name}\n"
                    f"    行数: {n_rows} / 取得日数: {actual_days}日 / サイズ: {file_size_text}"
                )

        except Exception as e:
            fail_count += 1
            failed_records.append({
                "coin_id": coin_id,
                "symbol": symbol,
                "name": name,
                "error": str(e),
            })
            log(f"{symbol} ({coin_id}) 失敗: {e}")

        elapsed = time.time() - start_time
        done = i + 1
        rate = done / actual_n
        avg_time = elapsed / done
        remain = avg_time * (actual_n - done)
        step_time = time.time() - loop_start

        log(
            f"[{done}/{actual_n} | {rate * 100:.1f}%] {symbol} 完了\n"
            f"    今回: {step_time:.2f}s / 経過: {fmt_time(elapsed)} / "
            f"残り目安: {fmt_time(remain)} / 平均: {avg_time:.2f}s/件\n"
            f"    成功: {success_count} / 空: {empty_count} / 失敗: {fail_count}"
        )

        if done < actual_n:
            controlled_sleep(args.min_sleep_sec, args.jitter_sec)

    if saved_records:
        summary_df = pd.DataFrame(saved_records)
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        log(f"一覧CSV保存完了: {summary_csv.resolve()}")
    else:
        log("保存できた銘柄はありませんでした")

    if empty_records:
        empty_df = pd.DataFrame(empty_records)
        empty_df.to_csv(empty_csv, index=False, encoding="utf-8-sig")
        log(f"空データ一覧CSV保存完了: {empty_csv.resolve()}")

    if failed_records:
        failed_df = pd.DataFrame(failed_records)
        failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        log(f"失敗一覧CSV保存完了: {failed_csv.resolve()}")

    total_time = time.time() - start_time
    log(f"総処理時間: {fmt_time(total_time)}")
    log("処理終了")


if __name__ == "__main__":
    main()