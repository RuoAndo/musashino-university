# 必要なライブラリをインストール
#!pip install lime scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular

# 1. データの読み込み
data = pd.read_csv("lags_12months_features.csv")

# 2. データの前処理
# 目的変数 (target) と特徴量 (features) の分離
target_column = 'target'  # 目的変数のカラム名を適宜設定
X = data.drop(target_column, axis=1)
y = data[target_column]

# 学習用とテスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. モデルの学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# テストデータでの予測結果の確認
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. LIMEによる解釈
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=['class_0', 'class_1'],  # クラス名を適宜変更
    mode='classification'
)

# 予測の解釈を行うインスタンスを選択
i = 0  # テストデータのインデックスを指定
exp = explainer.explain_instance(X_test.iloc[i].values, model.predict_proba, num_features=5)

# 結果を表示
exp.show_in_notebook()
