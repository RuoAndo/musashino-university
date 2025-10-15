import os
import numpy as np
from scipy.stats import wasserstein_distance

def load_valid_latlon(filepath):
    """3列目と4列目が数値の行のみ抽出して返す"""
    lats, lons = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # ヘッダスキップ
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                lat_s, lon_s = parts[2].strip(), parts[3].strip()
                if lat_s and lon_s:
                    try:
                        lats.append(float(lat_s))
                        lons.append(float(lon_s))
                    except ValueError:
                        continue
    return np.array(lats), np.array(lons)

# === メイン処理 ===
files = [f for f in os.listdir('.') if f.endswith('.csv')]
lat_data, lon_data = {}, {}

for f in files:
    lats, lons = load_valid_latlon(f)
    if len(lats) > 0 and len(lons) > 0:
        lat_data[f] = lats
        lon_data[f] = lons

file_list = sorted(lat_data.keys())
n = len(file_list)
dist_matrix = np.zeros((n, n))

# lat・lonのWasserstein距離を合成して2次元距離化
for i in range(n):
    for j in range(i + 1, n):
        d_lat = wasserstein_distance(lat_data[file_list[i]], lat_data[file_list[j]])
        d_lon = wasserstein_distance(lon_data[file_list[i]], lon_data[file_list[j]])
        dist = np.sqrt(d_lat**2 + d_lon**2)
        dist_matrix[i, j] = dist_matrix[j, i] = dist

print("=== 2D Wasserstein距離行列 ===")
print(file_list)
print(dist_matrix)

