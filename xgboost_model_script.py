# Trenowanie modelu XGBoost i obliczenie RMSE

import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

from dataset_loader import DatasetLoader

# Wczytanie danych
loader = DatasetLoader()
x_train, x_test, y_train, y_test = loader.load(predict="salary_in_usd")

# Tworzenie i trenowanie modelu
model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective="reg:squarederror"
)

model.fit(x_train, y_train)

# Przewidywanie i obliczenie RMSE
preds = model.predict(x_test)
rmse = root_mean_squared_error(y_test, preds)

print("Model XGBoost został wytrenowany.")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
