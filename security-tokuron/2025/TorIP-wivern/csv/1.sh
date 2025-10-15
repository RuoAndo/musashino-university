#!/bin/bash
# カレントディレクトリ内の全CSVファイルから1行目を削除するスクリプト

for f in *.csv; do
  # ファイルが存在しない場合はスキップ
  [ -e "$f" ] || continue

  # 一時ファイルを作って先頭行を除く
  tail -n +2 "$f" > "$f.tmp" && mv "$f.tmp" "$f"

  echo "1行目を削除しました: $f"
done

