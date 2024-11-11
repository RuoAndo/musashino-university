import csv
import argparse
from collections import defaultdict
from glob import glob
import os
import matplotlib.pyplot as plt

def calculate_combined_score_sum_by_date(directory_path):
    combined_score_by_date = defaultdict(int)
    
    # ディレクトリ内のすべてのCSVファイルを取得
    csv_files = glob(os.path.join(directory_path, '*.csv'))
    
    # 各ファイルに対してスコアの合計を計算
    for file_path in csv_files:
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                date = row['Date']
                score = int(row['Score'])
                combined_score_by_date[date] += score
    
    return combined_score_by_date

def plot_combined_score_sum_by_date(combined_score_by_date):
    # 日付でソートしてリスト化
    dates = sorted(combined_score_by_date.keys())
    scores = [combined_score_by_date[date] for date in dates]
    
    # プロット設定
    plt.figure(figsize=(12, 6))
    plt.plot(dates, scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Total Score Sum')
    plt.title('Combined Score Sum by Date Across All Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 引数を解析する設定
    parser = argparse.ArgumentParser(description="Calculate and plot the combined sum of 'Score' by date from all CSV files in a directory.")
    parser.add_argument('directory_path', type=str, help="Path to the directory containing CSV files")
    args = parser.parse_args()
    
    # ディレクトリ内のファイルを処理してスコアの合計を取得し、プロット
    combined_score_by_date = calculate_combined_score_sum_by_date(args.directory_path)
    plot_combined_score_sum_by_date(combined_score_by_date)
