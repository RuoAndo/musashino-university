# 必要なライブラリをインストール
#!pip install lime scikit-learn pandas

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lime.lime_tabular

# 1. カレントディレクトリの確認
print("カレントディレクトリ:", os.getcwd())

# 2. CSVファイルの読み込み（カレントディレクトリから）
data = pd.read_csv("lags_12months_features.csv")

# 3. データの確認（カラム名と先頭5行を表示）
print("カラム名一覧:", data.columns)
print(data.head())

# 4. 目的変数のカラム名を指定（'t'が目的変数と仮定）
target_column = 't'

# 5. 目的変数と特徴量の分離
try:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
except KeyError:
    print(f"エラー: '{target_column}' カラムが見つかりません。")
    exit()

# 6. データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. ランダムフォレスト回帰モデルの構築と学習
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 8. テストデータでの予測と評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"平均二乗誤差 (MSE): {mse}")

# 9. LIMEによる予測の解釈
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    mode='regression'
)

# 10. テストデータの最初のインスタンスでLIMEを使った解釈を表示
i = 0  # インデックスを指定
exp = explainer.explain_instance(X_test.iloc[i].values, model.predict, num_features=5)

# 11. 解釈結果を表示
exp.show_in_notebook()
