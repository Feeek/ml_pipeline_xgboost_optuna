import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# Wczytanie danych
X_train = pd.read_csv("X_train.csv")
X_train = X_train.drop(columns=["Unnamed: 0"])
X_test = pd.read_csv("X_test.csv")
X_test = X_test.drop(columns=["Unnamed: 0"])
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Trenowanie modelu z najlepszymi parametrami (Optuna)
model = xgb.XGBRegressor(
    max_depth=5,
    learning_rate=0.215,
    n_estimators=249,
    objective="reg:squarederror"
)
model.fit(X_train, y_train)

# 1. Feature importance
xgb.plot_importance(model, importance_type="weight")
plt.title("XGBoost - Feature Importance (weight)")
plt.tight_layout()
plt.show()

# 2. SHAP values (interpretacja modelu)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (ważność cech)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot (rozrzut wpływu cech na predykcje)
shap.summary_plot(shap_values, X_test)
