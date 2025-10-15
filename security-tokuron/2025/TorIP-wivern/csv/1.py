import os
import numpy as np
from scipy.stats import wasserstein_distance

# カレントディレクトリ内のCSVファイルを対象
files = [f for f in os.listdir('.') if f.endswith('.csv')]
data = {}

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        if len(lines) >= 4:
            coords = []
            for line in lines[2:4]:  # 3行目と4行目
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    lat_str, lon_str = parts[2].strip(), parts[3].strip()
                    # 3列目と4列目に値が入っている場合のみ処理
                    if lat_str and lon_str:
                        try:
                            lat = float(lat_str)
                            lon = float(lon_str)
                            coords.append((lat, lon))
                        except ValueError:
                            # 数値変換できない場合はスキップ
                            continue
            if coords:
                # 1次元に変換して保存（Wasserstein距離は1次元専用）
                data[f] = np.array([v for pair in coords for v in pair])

# ファイル間のワッサースタイン距離を計算
file_list = list(data.keys())
n = len(file_list)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        d = wasserstein_distance(data[file_list[i]], data[file_list[j]])
        dist_matrix[i, j] = dist_matrix[j, i] = d

# 結果を表示
print("=== Wasserstein距離行列 ===")
print(file_list)
print(dist_matrix)

