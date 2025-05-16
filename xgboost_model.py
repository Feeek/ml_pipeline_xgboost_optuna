import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import ast
import matplotlib.pyplot as plt

from dataset_loader import DatasetLoader

# Wczytanie danych
loader = DatasetLoader()
x_train, x_test, y_train, y_test = loader.load(predict="salary_in_usd")

# Wczytanie najlepszych parametrów z pliku
with open("best_params.txt", "r") as f:
    best_params = ast.literal_eval(f.read())

# Trenowanie i ewaluacja modelu
model = xgb.XGBRegressor(**best_params)
model.fit(x_train, y_train)
preds = model.predict(x_test)
rmse = root_mean_squared_error(y_test, preds,)
print(f"RMSE najlepszego modelu: {rmse:.2f}")

# Wykresy ważności cech
xgb.plot_importance(model)
plt.tight_layout()
plt.show()
