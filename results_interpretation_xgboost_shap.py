import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from dataset_loader import DatasetLoader

# Wczytanie danych
loader = DatasetLoader()
x_train, x_test, y_train, y_test = loader.load(predict="salary_in_usd")

# Trenowanie modelu z najlepszymi parametrami (Optuna)
model = xgb.XGBRegressor(
    max_depth=5,
    learning_rate=0.215,
    n_estimators=249,
    objective="reg:squarederror"
)
model.fit(x_train, y_train)

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
