import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import re

def extract_date_from_filename(file_name):
    match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    else:
        return "日付不明"

def run_dbscan(file_name, eps=0.001, min_samples=10):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"指定されたファイルが存在しません: {file_name}")

    df = pd.read_csv(file_name)

    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    coordinates = df.iloc[:, 1:3].dropna().values

    if coordinates.shape[0] == 0:
        raise ValueError("緯度・経度データが空です。CSVファイルの内容を確認してください。")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine', algorithm='ball_tree')
    coordinates_rad = np.radians(coordinates)
    clusters = dbscan.fit_predict(coordinates_rad)

    df = df.iloc[:len(clusters)]
    df['cluster'] = clusters

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    inferred_date = extract_date_from_filename(file_name)

    # ノイズ点の座標を取得
    noise_points = df[df['cluster'] == -1].iloc[:, 1:3]
    noise_coords = noise_points.values

    # 結果の表示
    print(f"{inferred_date},{n_clusters},{n_noise}", end="")
    if noise_coords.size > 0:
        print(",", end="")
        print(", ".join([f"{lat}, {lng}" for lat, lng in noise_coords]))
    else:
        print(", ノイズ点の座標なし")

    # 結果をCSVに保存
    base_name = os.path.basename(file_name)
    output_file = f'dbscan_result_{base_name}'
    df.to_csv(output_file, index=False)

    plot_clusters(df)

def plot_clusters(df):
    plt.figure(figsize=(10, 6))
    unique_clusters = set(df['cluster'])

    colormap = plt.colormaps.get_cmap('tab10')

    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        color = 'black' if cluster_id == -1 else colormap(cluster_id % 10)
        plt.scatter(
            cluster_data.iloc[:, 2], cluster_data.iloc[:, 1],
            s=50, c=[color], label=f'Cluster {cluster_id}', alpha=0.6
        )

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('DBSCAN Clustering Result')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBSCANでクラスタリングを実行し、結果をプロットします。')
    parser.add_argument('file', help='クラスタリング対象のCSVファイル名を指定してください。')
    parser.add_argument('--eps', type=float, default=0.3, help='クラスタ間の最大距離 (eps)')
    parser.add_argument('--min_samples', type=int, default=3, help='1クラスタの最小サンプル数')

    args = parser.parse_args()
    run_dbscan(args.file, eps=args.eps, min_samples=args.min_samples)
