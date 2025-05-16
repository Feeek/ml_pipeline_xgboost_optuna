import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import ast
import matplotlib.pyplot as plt

# Wczytanie danych
X_train = pd.read_csv("X_train.csv")
X_train = X_train.drop(columns=["Unnamed: 0"])
X_test = pd.read_csv("X_test.csv")
X_test = X_test.drop(columns=["Unnamed: 0"])
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Wczytanie najlepszych parametrów z pliku
with open("best_params.txt", "r") as f:
    best_params = ast.literal_eval(f.read())

# Trenowanie i ewaluacja modelu
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"RMSE najlepszego modelu: {rmse:.2f}")

# Wykresy ważności cech
xgb.plot_importance(model)
plt.tight_layout()
plt.show()
