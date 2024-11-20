import csv
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

def extract_date_from_filename(file_name):
    # ファイル名から日付を推測（形式: YYYYMMDD）
    match = re.search(r'\d{8}', file_name)  # 8桁の日付を検索
    if match:
        # YYYYMMDD を YYYY-MM-DD に変換
        raw_date = match.group(0)
        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        return formatted_date
    return None

def calculate_statistics(file_path, column_index):
    scores = []
    
    # ファイルを読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # 数値データを取得（7列目の場合は column_index = 6）
                score = float(row[column_index])
                scores.append(score)
            except (ValueError, IndexError):
                # 数値に変換できない行をスキップ
                continue

    # 平均、分散、最大値を計算
    if scores:
        mean_score = np.mean(scores)
        variance_score = np.var(scores)
        max_score = np.max(scores)
        return mean_score, variance_score, max_score
    else:
        return None, None, None

def process_directory(directory_path, column_index):
    data_points = []  # 日付と統計値を格納するリスト

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            date_str = extract_date_from_filename(file_name)  # ファイル名から日付を取得
            
            if date_str:
                mean, variance, max_value = calculate_statistics(file_path, column_index)
                if mean is not None:
                    data_points.append((date_str, mean, variance, max_value))
                    print(f"日付: {date_str}, 平均: {mean}, 分散: {variance}, 最大値: {max_value}")
                else:
                    print(f"ファイル名: {file_name} - スコアが見つかりませんでした。")
    
    # 日付順にソート
    data_points.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

    # プロット用データを作成
    dates = [datetime.strptime(dp[0], '%Y-%m-%d') for dp in data_points]
    means = [dp[1] for dp in data_points]
    variances = [dp[2] for dp in data_points]
    max_values = [dp[3] for dp in data_points]

    # 時系列プロット（サブプロット）
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 平均値のプロット
    axes[0].plot(dates, means, marker='o', linestyle='-', label='平均値', color='blue')
    axes[0].set_ylabel('平均値')
    axes[0].grid(True)
    axes[0].legend()

    # 分散のプロット
    axes[1].plot(dates, variances, marker='s', linestyle='-', label='分散', color='orange')
    axes[1].set_ylabel('分散')
    axes[1].grid(True)
    axes[1].legend()

    # 最大値のプロット
    axes[2].plot(dates, max_values, marker='^', linestyle='-', label='最大値', color='green')
    axes[2].set_xlabel('日付')
    axes[2].set_ylabel('最大値')
    axes[2].grid(True)
    axes[2].legend()

    # タイトルの設定
    fig.suptitle('平均値、分散、最大値の時系列データ', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # コマンドライン引数でディレクトリパスを取得
    if len(sys.argv) != 2:
        print("使い方: python script_name.py <ディレクトリ名>")
        sys.exit(1)

    directory_path = sys.argv[1]  # コマンドライン引数からディレクトリパスを取得
    column_index = 6              # 7列目はインデックス6
    process_directory(directory_path, column_index)
