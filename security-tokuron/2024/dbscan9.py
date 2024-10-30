import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def run_dbscan(file_name, eps=0.01, min_samples=3):
    # CSVファイルの読み込み
    df = pd.read_csv(file_name)

    # 2列目と3列目の緯度・経度を数値型に変換して抽出
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

    # NaNを含む行を除去して緯度・経度を取得
    coordinates = df.iloc[:, 1:3].dropna().values

    # データが空でないかチェック
    if coordinates.shape[0] == 0:
        raise ValueError("緯度・経度データが空です。CSVファイルの内容を確認してください。")

    # DBSCANの設定
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine', algorithm='ball_tree')

    # 緯度・経度をラジアンに変換
    coordinates_rad = np.radians(coordinates)

    # クラスタリングの実行
    clusters = dbscan.fit_predict(coordinates_rad)

    # クラスタ結果を元のデータに追加
    df = df.iloc[:len(clusters)]  # クラスタ結果に合わせてデータを調整
    df['cluster'] = clusters

    # クラスタ数とノイズ数を計算・表示
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"クラスタ数: {n_clusters}")
    print(f"ノイズ数: {n_noise}")

    # 出力ファイル名の生成（元のファイル名からディレクトリを除去）
    base_name = os.path.basename(file_name)
    output_file = f'dbscan_result_{base_name}'

    # 結果の保存
    df.to_csv(output_file, index=False)
    print(f'クラスタリング結果を {output_file} に保存しました。')

    # クラスタリング結果のプロット
    plot_clusters(df)

def plot_clusters(df):
    plt.figure(figsize=(10, 6))

    # クラスタごとに色を変えてプロット
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    for cluster_id in set(df['cluster']):
        cluster_data = df[df['cluster'] == cluster_id]
        color = 'black' if cluster_id == -1 else colors[cluster_id % len(colors)]
        plt.scatter(
            cluster_data.iloc[:, 2], cluster_data.iloc[:, 1], 
            s=50, c=color, label=f'Cluster {cluster_id}', alpha=0.6, edgecolors='w'
        )

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('DBSCAN Clustering Result')
    plt.legend(loc='best')
    plt.show()

# メイン処理部分
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBSCANでクラスタリングを実行し、結果をプロットします。')
    parser.add_argument('file', help='クラスタリング対象のCSVファイル名を指定してください。')
    parser.add_argument('--eps', type=float, default=0.01, help='クラスタ間の最大距離 (eps)')
    parser.add_argument('--min_samples', type=int, default=3, help='1クラスタの最小サンプル数')

    args = parser.parse_args()

    # 指定されたファイルでDBSCANを実行
    run_dbscan(args.file, eps=args.eps, min_samples=args.min_samples)
