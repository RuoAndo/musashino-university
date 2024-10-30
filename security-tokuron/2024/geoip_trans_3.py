import geoip2.database
import pandas as pd
import folium
from folium.plugins import HeatMap
import io

def create_heatmap(csv_file_path, db_path, output_csv_path, output_html_path):
    # 1. GeoLite2データベースを読み込む
    reader = geoip2.database.Reader(db_path)

    # 2. IPアドレスリストをCSVから読み込む
    data = pd.read_csv(csv_file_path, header=None, names=['ip_address', 'value'])

    # 3. 緯度・経度を取得し、リストに保存する
    locations = []
    output_data = []  # IPアドレス、緯度、経度を保存するリスト
    for ip in data['ip_address']:
        try:
            response = reader.city(ip)
            lat = response.location.latitude
            lon = response.location.longitude
            if lat and lon:
                locations.append([lat, lon])
                # IPアドレス、緯度、経度をリストに追加
                output_data.append([ip, lat, lon])
        except geoip2.errors.AddressNotFoundError:
            pass  # 位置情報が見つからない場合は無視
        except Exception as e:
            print(f"Error retrieving data for IP: {ip} - {e}")

    # データベースを閉じる
    reader.close()

    # 4. データをCSVファイルに保存
    df = pd.DataFrame(output_data, columns=['IP Address', 'Latitude', 'Longitude'])
    df.to_csv(output_csv_path, index=False)
    print(f"IPアドレスと緯度経度が '{output_csv_path}' に保存されました。")

    # 5. ヒートマップを作成する
    m = folium.Map(location=[0, 0], zoom_start=2)  # 地図を作成（世界地図に焦点を合わせる）
    HeatMap(locations).add_to(m)

    # 6. ヒートマップをHTMLとして保存
    m.save(output_html_path)
    print(f"ヒートマップが '{output_html_path}' に保存されました。")


# 使用例
csv_file_path = 'tor.20220322.txt'  # 読み込むIPアドレスのCSVファイル
db_path = 'GeoLite2-City.mmdb'   # GeoLite2-City.mmdb のファイルパス
output_csv_path = 'latlng_output.csv'  # 緯度経度を保存するCSVファイル
output_html_path = 'heatmap_output.html'  # ヒートマップを保存するHTMLファイル

create_heatmap(csv_file_path, db_path, output_csv_path, output_html_path)
