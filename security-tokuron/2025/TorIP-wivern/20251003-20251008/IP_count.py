import os
import pandas as pd
from collections import Counter

# === カレントディレクトリ ===
TARGET_DIR = os.getcwd()

# === 集計用カウンタ ===
counter = Counter()

# === 再帰的に探索して処理 ===
for root, _, files in os.walk(TARGET_DIR):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(root, file)
            try:
                df = pd.read_csv(path)
                if "ip" in df.columns:
                    counter.update(df["ip"].dropna().astype(str))
            except Exception as e:
                print(f"⚠️ 読み込み失敗: {path} ({e})")

# === 結果をデータフレームに変換 ===
result = pd.DataFrame(counter.items(), columns=["IPアドレス", "出現回数"]).sort_values("出現回数", ascending=False)

# === 結果を表示 ===
print(result)

# === CSVとして保存 ===
out_path = os.path.join(TARGET_DIR, "ip_count_summary.csv")
result.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 出力完了: {out_path}")
