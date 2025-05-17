import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

import ast
from dataset_loader import DatasetLoader

# Wczytanie danych
loader = DatasetLoader()
x_train, x_test, y_train, y_test = loader.load(predict="salary_in_usd")

# Trenowanie modelu z najlepszymi parametrami (Optuna)
with open("best_params.txt", "r") as f:
    best_params = ast.literal_eval(f.read())

# Trenowanie i ewaluacja modelu
model = xgb.XGBRegressor(**best_params)
model.fit(x_train, y_train)

preds = model.predict(x_test)
rmse = root_mean_squared_error(y_test, preds,)
print(f"RMSE najlepszego modelu: {rmse:.2f}")

# 1. Feature importance
xgb.plot_importance(model, importance_type="weight")
plt.title("XGBoost - Feature Importance (weight)")
plt.tight_layout()
plt.show()

# 2. SHAP values (interpretacja modelu)
explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_test)

# Summary plot (ważność cech)
shap.summary_plot(shap_values, x_test, plot_type="bar")

# Summary plot (rozrzut wpływu cech na predykcje)
shap.summary_plot(shap_values, x_test)
