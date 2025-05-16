import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

from dataset_loader import DatasetLoader

# Wczytanie danych
loader = DatasetLoader()
x_train, x_test, y_train, y_test = loader.load(predict="salary_in_usd")

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
study.optimize(objective, n_trials=30)

# Najlepsze parametry i wynik
print("Najlepsze parametry:")
print(study.best_params)

best_model = xgb.XGBRegressor(**study.best_params)
best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)
final_rmse = root_mean_squared_error(y_test, preds)
print(f"RMSE najlepszego modelu: {final_rmse:.2f}")

# Wykres historii prób
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.show()
