import os
import numpy as np
from scipy.stats import wasserstein_distance

def first_k_valid_latlon(lines, k=2, start_row=1):
    """ヘッダを除き、start_row(0基準)以降からlat/lonが両方数値の行をk件集める"""
    coords = []
    for line in lines[start_row:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            lat_s, lon_s = parts[2].strip(), parts[3].strip()
            if lat_s and lon_s:
                try:
                    coords.append((float(lat_s), float(lon_s)))
                    if len(coords) == k:
                        break
                except ValueError:
                    pass
    return coords

# 対象ファイル
files = [f for f in os.listdir('.') if f.endswith('.csv')]

# 各ファイルから2点ずつ取得（ヘッダ1行想定なので start_row=1、3行目固定にしない）
lat_data = {}
lon_data = {}
picked_rows = 2  # 2点
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    coords = first_k_valid_latlon(lines, k=picked_rows, start_row=1)  # 必要なら start_row=2 など調整
    if len(coords) == picked_rows:
        lats = np.array([c[0] for c in coords], dtype=float)
        lons = np.array([c[1] for c in coords], dtype=float)
        lat_data[f] = lats
        lon_data[f] = lons

file_list = sorted(lat_data.keys())
n = len(file_list)
dist_matrix = np.zeros((n, n), dtype=float)

for i in range(n):
    for j in range(i + 1, n):
        d_lat = wasserstein_distance(lat_data[file_list[i]], lat_data[file_list[j]])
        d_lon = wasserstein_distance(lon_data[file_list[i]], lon_data[file_list[j]])
        # 簡易2次元距離（各軸WDの合成）
        d = (d_lat**2 + d_lon**2) ** 0.5
        dist_matrix[i, j] = dist_matrix[j, i] = d

print("=== 合成Wasserstein距離行列（lat/lon合成） ===")
print(file_list)
print(dist_matrix)

