# Trenowanie modelu XGBoost i obliczenie RMSE

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Wczytanie danych
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Tworzenie i trenowanie modelu
model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# Przewidywanie i obliczenie RMSE
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)

print("Model XGBoost został wytrenowany.")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
