import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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

def perform_lof_and_save(csv_file_path, n_neighbors=20, contamination=0.1):
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

        # LOFによる異常検知の実行
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        data['Outlier'] = lof.fit_predict(coordinates)

        # LOFでの結果が -1 の場合は外れ値、1 の場合は通常データ
        data['Outlier'] = data['Outlier'].apply(lambda x: 'Outlier' if x == -1 else 'Inlier')

    except Exception as e:
        print(f"LOF処理中にエラーが発生しました: {e}")
        return

    # 出力ファイル名の生成
    output_csv_path = f'lof_output_{date_str}.csv'

    try:
        # データをCSVに保存
        data.to_csv(output_csv_path, index=False)
        print(f"データが '{output_csv_path}' に保存されました。")
    except Exception as e:
        print(f"CSVファイルの保存中にエラーが発生しました: {e}")

    # 異常検知結果をプロット
    plot_lof(data, date_str)

def plot_lof(data, date_str):
    """LOFの結果をプロットする関数"""
    plt.figure(figsize=(10, 6))

    # 異常データと通常データで色を分けてプロット
    inliers = data[data['Outlier'] == 'Inlier']
    outliers = data[data['Outlier'] == 'Outlier']

    plt.scatter(inliers['Longitude'], inliers['Latitude'], label='Inlier')
    plt.scatter(outliers['Longitude'], outliers['Latitude'], color='red', marker='x', label='Outlier')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Local Outlier Factor Result ({date_str})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # コマンドライン引数からCSVファイルのパスを取得
    if len(sys.argv) < 2:
        print("使用方法: python script_name.py <CSVファイル名>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    perform_lof_and_save(csv_file_path)
