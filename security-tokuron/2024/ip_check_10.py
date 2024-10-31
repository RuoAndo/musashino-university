import pandas as pd
import requests
import sys
from time import sleep  # 過剰なAPIリクエストを避けるための待機

def check_ip(ip_address, api_key, days=30):
    """AbuseIPDB APIを使ってIPアドレスをチェックし、信頼度スコアを取得する"""
    url = "https://api.abuseipdb.com/api/v2/check"
    querystring = {
        "ipAddress": ip_address,
        "maxAgeInDays": days
    }
    headers = {
        "Accept": "application/json",
        "Key": api_key
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        score = data['data']['abuseConfidenceScore']
        print(f"IP: {ip_address} - 信頼度スコア: {score}")
        return score
    else:
        print(f"エラー: {response.status_code} - {response.text}")
        return None

def check_outlier_ips(csv_file_path, api_key):
    """クラスタ番号が -1 のIPアドレスをチェックし、スコア付きで保存する"""
    try:
        # クラスタリング結果のファイルを読み込む
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # クラスタ番号が -1 のIPアドレスを抽出
    outliers = data[data['Cluster'] == -1].copy()

    if outliers.empty:
        print("外れ値となるIPアドレスはありません。")
        return

    # スコア列を追加して初期化
    outliers['Score'] = None

    print(f"{len(outliers)} 個の外れ値が見つかりました。IPアドレスをチェックします...")

    # 各外れ値のIPアドレスをチェックし、スコアを取得
    for idx, row in outliers.iterrows():
        ip = row['IP Address']
        print(f"\n[{idx + 1}/{len(outliers)}] {ip} をチェック中...")
        score = check_ip(ip, api_key)
        outliers.at[idx, 'Score'] = score if score is not None else "エラー"

        # 過剰なリクエストを避けるために少し待機
        sleep(1)

    # 出力ファイル名を生成
    output_csv_path = csv_file_path.replace('.csv', '_scored.csv')

    try:
        # スコア付きデータをCSVに保存
        outliers.to_csv(output_csv_path, index=False)
        print(f"\nデータが '{output_csv_path}' に保存されました。")
    except Exception as e:
        print(f"CSVファイルの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python script_name.py <クラスタリング結果のCSVファイル名> <APIキー>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    api_key = sys.argv[2]

    # 外れ値のIPアドレスをチェック
    check_outlier_ips(csv_file_path, api_key)
