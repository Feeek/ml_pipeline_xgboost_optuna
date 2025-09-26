from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

from pandas import DataFrame


class XGBoost:
    def __init__(self, dataset: DataFrame, target: str = "salary_in_usd", test_size: float = 0.2, random_state: int = 42):
        self.X = dataset.drop(columns=[target])
        self.y = dataset[target]

        for col in self.X.select_dtypes(include="object"):
            self.X[col] = self.X[col].astype("category")

        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.best_params = {}

    def tune(self, n_trials: int = 500):
        def objective(trial: optuna.trial.Trial):
            params = {
                "objective": "reg:squarederror",
                "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
            }

            X_train, X_val, y_train, y_val = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )

            model = XGBRegressor(
                **params,
                early_stopping_rounds=20,
                enable_categorical=True,
                tree_method="hist",
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("✅ Best trial:", study.best_trial.params)
        self.best_params = {**self.best_params, **study.best_trial.params}


    def fit(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        self.model = XGBRegressor(
            **self.best_params,
            early_stopping_rounds=20,
            enable_categorical=True,
            tree_method="hist",
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = self.model.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        print("📊 Evaluation on validation set:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return self.model

    def predict(self) -> np.ndarray:
        return self.model.predict(self.X)

    def _evaluate(self, y_true, y_pred) -> dict:
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
