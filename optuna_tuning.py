import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error

# Wczytanie danych
X_train = pd.read_csv("X_train.csv")
X_train = X_train.drop(columns=["Unnamed: 0"])
X_test = pd.read_csv("X_test.csv")
X_test = X_test.drop(columns=["Unnamed: 0"])
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

# Zapis najlepszych parametrów do pliku
with open("best_params.txt", "w") as f:
    f.write(str(study.best_params))
