import csv
import numpy as np
import sys
import os
import re

def extract_date_from_filename(file_name):
    # ファイル名から日付を推測（形式: YYYYMMDD）
    match = re.search(r'\d{8}', file_name)  # 8桁の日付を検索
    if match:
        # YYYYMMDD を YYYY-MM-DD に変換
        raw_date = match.group(0)
        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        return formatted_date
    return "不明"

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
    # 指定ディレクトリ内のすべてのCSVファイルを処理
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            date = extract_date_from_filename(file_name)  # ファイル名から日付を取得
            mean, variance, max_value = calculate_statistics(file_path, column_index)
            
            # ファイル名と結果をCSV形式で出力
            if mean is not None:
                #print("日付,平均,分散,最大値")
                print(f"{date},{mean},{variance},{max_value}")
                #print()  # ファイルごとに改行
            else:
                print(f"ファイル名: {file_name} - スコアが見つかりませんでした。")

if __name__ == "__main__":
    # コマンドライン引数でディレクトリパスを取得
    if len(sys.argv) != 2:
        print("使い方: python script_name.py <ディレクトリ名>")
        sys.exit(1)

    directory_path = sys.argv[1]  # コマンドライン引数からディレクトリパスを取得
    column_index = 6              # 7列目はインデックス6
    process_directory(directory_path, column_index)
