import geoip2.database
import pandas as pd
import matplotlib.pyplot as plt
import io

def plot_coordinates(csv_file_path, db_path, output_csv_path, output_plot_path):
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

    # 5. 2次元散布図を作成する
    latitudes = [loc[0] for loc in locations]
    longitudes = [loc[1] for loc in locations]

    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, latitudes, c='blue', marker='o')
    plt.title('IP Address Geolocation (Latitude vs Longitude)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

    # 6. グラフを画像ファイルとして保存
    plt.savefig(output_plot_path)
    print(f"2次元座標プロットが '{output_plot_path}' に保存されました。")

    # 7. プロットを表示
    plt.show()

# 使用例
csv_file_path = 'tor.20220322.txt'  # 読み込むIPアドレスのCSVファイル
db_path = 'GeoLite2-City.mmdb'   # GeoLite2-City.mmdb のファイルパス
output_csv_path = 'latlng_output.csv'  # 緯度経度を保存するCSVファイル
output_plot_path = 'coordinates_plot.png'  # 2次元座標プロットを保存する画像ファイル

plot_coordinates(csv_file_path, db_path, output_csv_path, output_plot_path)
