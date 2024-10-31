import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import sys
import re

def extract_date_from_filename(filename):
    """ファイル名から8桁の日付（YYYYMMDD形式）を抽出する関数"""
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    else:
        return "unknown_date"

def perform_dbscan_and_save(csv_file_path, eps=13, min_samples=3):
    # ファイル名から日付を取得
    date_str = extract_date_from_filename(csv_file_path)

    try:
        # ファイルの読み込み
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    try:
        # 緯度・経度の抽出
        coordinates = data[['Latitude', 'Longitude']].values

        # DBSCANクラスタリングの実行
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

        # クラスタ番号の取得
        data['Cluster'] = db.labels_

    except Exception as e:
        print(f"DBSCAN処理中にエラーが発生しました: {e}")
        return

    # 出力ファイル名の生成
    output_csv_path = f'clustered_output_{date_str}.csv'

    try:
        # データをCSVに保存
        data.to_csv(output_csv_path, index=False)
        print(f"データが '{output_csv_path}' に保存されました。")
    except Exception as e:
        print(f"CSVファイルの保存中にエラーが発生しました: {e}")

    # クラスタリング結果をプロット
    plot_clusters(data, date_str)

def plot_clusters(data, date_str):
    """クラスタリング結果をプロットする関数"""
    plt.figure(figsize=(10, 6))

    # クラスタごとに色を分けてプロット
    unique_clusters = data['Cluster'].unique()
    for cluster in unique_clusters:
        cluster_data = data[data['Cluster'] == cluster]
        if cluster == -1:
            # 外れ値は黒でプロット
            plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                        color='black', marker='x', label='Outlier')
        else:
            plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                        label=f'Cluster {cluster}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'DBSCAN Clustering Result ({date_str})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # コマンドライン引数からCSVファイルのパスを取得
    if len(sys.argv) < 2:
        print("使用方法: python script_name.py <CSVファイル名>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    perform_dbscan_and_save(csv_file_path)
