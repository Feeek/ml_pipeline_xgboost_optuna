import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error

from etl import ETLPipeline

# Wczytanie danych
loader = ETLPipeline()
x_train, x_test, y_train, y_test = loader.extract(predict="salary_in_usd")

# Funkcja celu do optymalizacji
def objective(trial: optuna.Trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'objective': 'reg:squarederror'
    }

    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return root_mean_squared_error(y_test, preds)

# Optymalizacja
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Zapis najlepszych parametrów do pliku
with open("best_params.txt", "w") as f:
    f.write(str(study.best_params))
