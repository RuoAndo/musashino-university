#!/bin/bash
# カレントディレクトリ内のCSVファイルすべてに対して
# 1行目を削除し、新しいヘッダー ip,timestamp,lat,lng を追加

for f in *.csv; do
  [ -e "$f" ] || continue  # ファイルが無ければスキップ

  # 一時ファイルにヘッダー追加＋元データの2行目以降を追記
  {
    echo "ip,timestamp,lat,lng"
    tail -n +2 "$f"
  } > "$f.tmp"

  # 元のファイルに上書き
  mv "$f.tmp" "$f"

  echo "更新しました: $f"
done

