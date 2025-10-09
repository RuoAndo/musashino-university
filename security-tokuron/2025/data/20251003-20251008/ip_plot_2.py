# 事前: pip install -U plotly kaleido pillow pandas
import pandas as pd
import plotly.express as px

# 1) 座標付きCSVを読む（ip, count, country, city, lat, lon）
df = pd.read_csv("ip_city_geodata.csv")

# 2) 点サイズを頻度からスケーリング
min_dot, max_dot = 5, 40
cmin, cmax = df["count"].min(), df["count"].max()
df["_size"] = min_dot if cmax == cmin else (df["count"]-cmin)/(cmax-cmin)*(max_dot-min_dot)+min_dot

# 3) 地図作成（Shapefile不要）
fig = px.scatter_geo(
    df, lat="lat", lon="lon", size="_size",
    hover_name="ip",
    hover_data={"country": True, "city": True, "count": True, "_size": False, "lat": False, "lon": False},
    projection="natural earth",
    title="IPヒートマップ（GeoLite2-City, Plotly）"
)
fig.update_traces(opacity=0.6)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")

# 4) HTML と JPEG を保存（JPEG失敗ならPNG→JPEG変換）
fig.write_html("ip_city_heatmap.html", include_plotlyjs="cdn")

try:
    # kaleido が jpg 対応ならこれで直接OK
    fig.write_image("ip_city_heatmap.jpg", scale=2)  # or format="jpg"
except Exception:
    # フォールバック: いったんPNGで保存→PillowでJPEG化
    fig.write_image("ip_city_heatmap.png", scale=2)
    from PIL import Image
    Image.open("ip_city_heatmap.png").convert("RGB").save(
        "ip_city_heatmap.jpg", "JPEG", quality=90, optimize=True
    )

print("✅ 保存: ip_city_heatmap.html / ip_city_heatmap.jpg")

# 5) すぐ開く（Windows）
import os, webbrowser, sys
webbrowser.open(os.path.abspath("ip_city_heatmap.html"))
try:
    os.startfile(os.path.abspath("ip_city_heatmap.jpg"))
except Exception:
    pass  # WSL等では無視
