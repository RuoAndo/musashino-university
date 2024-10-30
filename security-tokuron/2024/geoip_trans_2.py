import geoip2.database
import pandas as pd
import folium
from folium.plugins import HeatMap
import io

# 1. GeoLite2データベースを読み込む
db_path = 'GeoLite2-City.mmdb'  # GeoLite2-City.mmdb のファイルパスを指定
reader = geoip2.database.Reader(db_path)

# 2. IPアドレスリストをCSVから読み込む
csv_data = """185.195.71.2,32
109.70.100.28,32
51.75.64.23,32
82.221.128.191,32
109.70.100.31,32
185.220.100.254,32
185.195.71.2,32
185.220.103.9,32
195.176.3.23,32
185.220.100.243,32
185.220.100.245,32
198.58.107.53,32"""

# pandasでCSVデータを読み込み (io.StringIOを使用)
data = pd.read_csv(io.StringIO(csv_data), header=None, names=['ip_address', 'value'])

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
df.to_csv('latlng.csv', index=False)
print("IPアドレスと緯度経度が 'latlng.csv' に保存されました。")

# 5. ヒートマップを作成する
m = folium.Map(location=[0, 0], zoom_start=2)  # 地図を作成（世界地図に焦点を合わせる）
HeatMap(locations).add_to(m)

# 6. ヒートマップをHTMLとして保存
m.save('heatmap.html')
print("ヒートマップが 'heatmap.html' に保存されました。")
