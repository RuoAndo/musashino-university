import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

# データの作成
data = {
    'IP Address': ['185.195.71.2', '109.70.100.28', '51.75.64.23', '82.221.128.191', '109.70.100.31', '185.220.100.254', '185.195.71.2', '185.220.103.9', '195.176.3.23', '185.220.100.243', '185.220.100.245', '198.58.107.53'],
    'Latitude': [47.1449, 48.1936, 48.8582, 65.0, 48.1936, 49.4617, 47.1449, 40.7064, 46.2334, 49.4617, 49.4617, 32.9473],
    'Longitude': [8.1551, 16.3726, 2.3387, -18.0, 16.3726, 11.0731, 8.1551, -73.9473, 6.1164, 11.0731, 11.0731, -96.7028]
}

# データフレームの作成
df = pd.DataFrame(data)

# 緯度と経度のデータを抽出
coords = df[['Latitude', 'Longitude']]

# DBSCANによるクラスタリング
dbscan = DBSCAN(eps=1.5, min_samples=2)  # epsとmin_samplesは適宜調整
clusters = dbscan.fit_predict(coords)

# クラスタ結果をデータフレームに追加
df['Cluster'] = clusters

# クラスタリング結果を表示
print(df)

# クラスタリング結果のプロット
plt.figure(figsize=(10, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='viridis', marker='o')

# 各クラスタに囲み線を引く
unique_clusters = set(clusters)
for cluster in unique_clusters:
    if cluster == -1:  # ノイズポイントは無視
        continue
    cluster_points = df[df['Cluster'] == cluster][['Longitude', 'Latitude']].values
    if len(cluster_points) >= 3:  # 凸包を描くには最低3点が必要
        hull = ConvexHull(cluster_points)
        hull_points = np.append(hull.vertices, hull.vertices[0])  # 凸包の線を閉じる
        plt.plot(cluster_points[hull_points, 0], cluster_points[hull_points, 1], 'r--', lw=2)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering with Convex Hulls')
plt.show()
