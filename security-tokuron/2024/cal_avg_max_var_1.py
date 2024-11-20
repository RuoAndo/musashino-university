import csv
import numpy as np
import sys

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
        
        print(f"平均: {mean_score}")
        print(f"分散: {variance_score}")
        print(f"最大値: {max_score}")
    else:
        print("スコアが見つかりませんでした。")

if __name__ == "__main__":
    # コマンドライン引数でファイル名を取得
    if len(sys.argv) != 2:
        print("使い方: python script_name.py <ファイル名>")
        sys.exit(1)

    file_path = sys.argv[1]  # コマンドライン引数からファイル名を取得
    column_index = 6         # 7列目はインデックス6
    calculate_statistics(file_path, column_index)
