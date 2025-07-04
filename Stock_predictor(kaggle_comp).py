import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
print("HEAD")
print(df.head())

print("\n INFO")
print(df.info())

print("\n DESCRIBE")
print(df.describe())
                 
null_ratio = df.isnull().mean()
useless_cols = null_ratio[null_ratio > 0.99].index.tolist()

print("Columns dropped (mostly missing):", useless_cols)

df.drop(columns=useless_cols, inplace=True)

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"Dropped {before - after} duplicate.rows")

print(df.isnull().sum().sort_values(ascending=False))

print("Before drop:", len(df))
df.dropna(inplace=True)
print("After Drop:", len(df))

df['DailyReturn'] = (df['Close'] - df['Open']) / df ['Open']

df['HighLowSpread'] = (df['High'] - df['Low']) / df['Low']

features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'AdjustmentFactor', 'DailyReturn', 'HighLowSpread'

target = 'Target'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
