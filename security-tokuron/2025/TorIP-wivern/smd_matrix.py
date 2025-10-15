# swd_matrix.py
# フォルダ内 *.csv を対象に、各ファイルから最大200行サンプルして
# スライスド・ワッサースタイン距離（1D/2D）行列を計算・保存
import argparse, glob, os, numpy as np, pandas as pd

def pick_numeric_cols(df, cols_arg):
    if cols_arg:
        cols = cols_arg
    else:
        # lat/lon があれば優先
        lc = {c.lower(): c for c in df.columns}
        if 'lat' in lc and 'lon' in lc:
            cols = [lc['lat'], lc['lon']]
        else:
            # 数値列を自動検出
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
            cols = num_cols[:2] if len(num_cols) >= 2 else num_cols[:1]
    if not cols:
        raise ValueError("数値列が見つかりません。--cols で列名を指定してください。")
    return cols

def sample_array(df, cols, n=200):
    sub = df[cols].dropna()
    if len(sub) == 0:
        raise ValueError("指定列に有効な数値データがありません。")
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    return sub.to_numpy()

def wasserstein_1d(x, y, q=1001):
    # 量子化で近似（逆CDFのL1）
    u = np.linspace(0, 1, q)
    xq = np.quantile(x, u, interpolation="linear")
    yq = np.quantile(y, u, interpolation="linear")
    return np.mean(np.abs(xq - yq))

def sliced_wasserstein(X, Y, projections=64):
    d = X.shape[1]
    if d == 1:
        return wasserstein_1d(X.ravel(), Y.ravel())
    total = 0.0
    rng = np.random.default_rng(42)
    for _ in range(projections):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12
        x1 = X @ v
        y1 = Y @ v
        total += wasserstein_1d(x1, y1)
    return total / projections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.csv", help="対象CSVパターン（例：*.csv）")
    ap.add_argument("--sample", type=int, default=200, help="各ファイルのサンプル行数上限")
    ap.add_argument("--cols", nargs="*", help="使用する列名（例：--cols lat lon）。1列指定で1D計算")
    ap.add_argument("--projections", type=int, default=64, help="スライス本数（2D以上）")
    ap.add_argument("--out", default="wasserstein_matrix.csv", help="出力CSVファイル名")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit("CSVが見つかりませんでした。")

    # 読み込み＆前処理
    data = []
    names = []
    first_cols = None
    for f in files:
        df = pd.read_csv(f)
        cols = pick_numeric_cols(df, args.cols)
        if first_cols is None and not args.cols:
            first_cols = cols  # ログ用
        arr = sample_array(df, cols, n=args.sample)
        # 1列なら shape=(n,1) に整形
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        data.append(arr)
        names.append(os.path.basename(f))

    # 行列計算
    n = len(data)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = sliced_wasserstein(data[i], data[j], projections=args.projections)
            M[i, j] = M[j, i] = d

    # 保存
    dfM = pd.DataFrame(M, index=names, columns=names)
    dfM.to_csv(args.out, encoding="utf-8-sig")
    print(f"✅ 距離行列を保存しました: {args.out}")
    if first_cols:
        print(f"（自動選択列: {first_cols}）" )

if __name__ == "__main__":
    main()
