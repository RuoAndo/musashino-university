import requests, re, pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

URL = "https://www.wivern.com/security20170414.html"
r = requests.get(URL, timeout=20)
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")
text = soup.get_text("\n")

lines = text.splitlines()
ip_re = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
results = []
for line in lines:
    m = ip_re.search(line)
    if m:
        ip = m.group(0)
        ts_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
        ts = ts_match.group(0) if ts_match else ""
        results.append((ip, ts))

dedup = {}
for ip, ts in results:
    if ip not in dedup or (ts and ts > dedup[ip]):
        dedup[ip] = ts

df = pd.DataFrame(list(dedup.items()), columns=["ip", "timestamp"])

# ---- 最後の行を削除 ----
if not df.empty:
    df = df.iloc[:-1]

# 現在時刻をファイル名に付ける（例: 20251003151000-TorIP.csv）
now_str = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"{now_str}-TorIP.csv"

# 保存先ディレクトリ
save_dir = r"C:\Users\user\OneDrive\【武蔵野大学】\2025年 サイバーセキュリティ特論\TorIP-wivern"
os.makedirs(save_dir, exist_ok=True)

filepath = os.path.join(save_dir, filename)

# CSV形式で保存（カンマ区切り、UTF-8）
df.to_csv(filepath, index=False, encoding="utf-8")

# ====== 取得データを表示 ======
print("取得したTor IPリスト（最後の行を削除後）：")
print(df)

print(f"\nSaved: {filepath}")
