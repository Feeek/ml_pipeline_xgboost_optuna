import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Wczytanie danych
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Funkcja celu do optymalizacji
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'objective': 'reg:squarederror'
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse

# Optymalizacja
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Najlepsze parametry i wynik
print("Najlepsze parametry:")
print(study.best_params)

best_model = xgb.XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)
final_rmse = mean_squared_error(y_test, preds, squared=False)
print(f"RMSE najlepszego modelu: {final_rmse:.2f}")

# Wykres historii prób
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.show()
