# pip install plotly pandas
import pandas as pd
import plotly.express as px

# 1) CSV 読み込み（ip, count, country, city, lat, lon）
df = pd.read_csv("ip_city_geodata.csv")

# 2) 目視しやすいようにサイズをスケール調整
#    （頻度のレンジに応じて適宜調整）
min_dot, max_dot = 5, 40
if not df.empty:
    cmin, cmax = df["count"].min(), df["count"].max()
    if cmax == cmin:
        df["_size"] = min_dot  # 全部同じなら固定サイズ
    else:
        df["_size"] = (df["count"] - cmin) / (cmax - cmin) * (max_dot - min_dot) + min_dot
else:
    df["_size"] = min_dot

# 3) Plotly で散布（Shapefile不要）
fig = px.scatter_geo(
    df,
    lat="lat",
    lon="lon",
    size="_size",
    hover_name="ip",
    hover_data={"country": True, "city": True, "count": True, "lat": False, "lon": False, "_size": False},
    projection="natural earth",
    title="IPヒートマップ（GeoLite2-City, Scatter on World）"
)
fig.update_traces(opacity=0.6)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

# 4) HTML に保存（ダブルクリックで開ける）
fig.write_html("ip_city_heatmap.html", include_plotlyjs="cdn")
print("🗺️ 出力: ip_city_heatmap.html")
