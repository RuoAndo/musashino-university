import geoip2.database
import pandas as pd
import sys
import re
import ipaddress  # IPアドレスの検証用

def extract_date_from_filename(filename):
    """ファイル名から8桁の日付（YYYYMMDD形式）を抽出する関数"""
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)  # YYYYMMDD形式の日付を返す
    else:
        return "unknown_date"

def is_valid_ip(ip):
    """IPアドレスが有効かどうかをチェックする関数"""
    try:
        ipaddress.ip_address(ip)  # 有効なIP形式かチェック
        return True
    except ValueError:
        return False

def save_latlng_to_csv(ip_file_path):
    # GeoLite2データベースのパス
    db_path = 'GeoLite2-City.mmdb'
    date_str = extract_date_from_filename(ip_file_path)  # ファイル名から日付を取得
    output_csv_path = f'latlng_output_{date_str}.csv'  # 出力CSVファイルのパス

    try:
        # GeoLite2データベースの読み込み
        reader = geoip2.database.Reader(db_path)
    except FileNotFoundError:
        print("GeoLite2-City.mmdb ファイルが見つかりません。正しいパスを指定してください。")
        return

    try:
        # CSVファイルを読み込み、1列目（IPアドレス列）を取得
        data = pd.read_csv(ip_file_path, header=None, usecols=[0], names=['ip_address'])
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 出力用データを格納するリスト
    output_data = []

    for index, ip in enumerate(data['ip_address'].drop_duplicates(), start=1):  # 重複するIPを除外し番号を付与
        if not is_valid_ip(ip):
            print(f"無効なIPアドレスをスキップしました: {ip}")
            continue  # 無効なIPはスキップ

        try:
            response = reader.city(ip)
            lat = response.location.latitude
            lon = response.location.longitude
            if lat is not None and lon is not None:
                output_data.append([index, date_str, ip, lat, lon])  # データ番号、日付、IP、緯度、経度を格納
        except geoip2.errors.AddressNotFoundError:
            print(f"IPアドレス {ip} の位置情報が見つかりませんでした。")
        except Exception as e:
            print(f"IPアドレス {ip} の処理中にエラーが発生しました: {e}")

    reader.close()  # データベースを閉じる

    # データをCSVに保存
    if output_data:
        df = pd.DataFrame(output_data, columns=['Data Number', 'Date', 'IP Address', 'Latitude', 'Longitude'])
        df.to_csv(output_csv_path, index=False)
        print(f"データが '{output_csv_path}' に保存されました。")
    else:
        print("有効な位置情報が取得できませんでした。")

if __name__ == "__main__":
    # コマンドライン引数からIPアドレスファイルを取得
    if len(sys.argv) < 2:
        print("使用方法: python script_name.py <IPアドレスファイル名>")
        sys.exit(1)

    ip_file_path = sys.argv[1]
    save_latlng_to_csv(ip_file_path)
