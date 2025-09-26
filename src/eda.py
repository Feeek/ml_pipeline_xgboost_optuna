import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

NEW_LINE = '\n\n'


class EDA():
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset

    def describe(self):
        missing_values = self.dataset.isnull().sum().sum()
        print(f"Missing values or NaNs: {missing_values}")

        print(f"Data dimensions (w x h): {self.dataset.shape[0]:,} x {self.dataset.shape[1]}", end=NEW_LINE)
        print(self.dataset.nunique().sort_values(ascending=False).to_frame("unique_values"), end=NEW_LINE)

        desc_num = self.dataset.describe().T
        print(desc_num[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2), end=NEW_LINE)

        desc_cat = self.dataset.describe(include=["object"]).T
        print(desc_cat[["unique", "top", "freq"]], end=NEW_LINE)


    def correlations(self, target: str, exclude_cols: list) -> pd.Series:
        dataset = self.dataset.drop(columns=exclude_cols)

        methods = ["pearson", "spearman", "kendall"]
        results = {}

        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(dataset.columns) * 0.3)))

        for i, method in enumerate(methods):
            num_corr = self._numeric_corr(dataset, target, method=method)
            cat_corr = self._categorical_corr(dataset, target)

            assoc = pd.concat([num_corr, cat_corr]).sort_values(ascending=False)
            results[method] = assoc

            sns.heatmap(
                assoc.to_frame("association"),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True,
                ax=axes[i]
            )
            axes[i].set_title(f"{method.capitalize()} correlation")

        plt.tight_layout()
        plt.show()

    def _categorical_corr(self, dataset: DataFrame, target: str) -> pd.Series:
        cat_df = dataset.select_dtypes(include=["object", "category"])

        y = dataset[target]
        results = {}
        for col in cat_df.columns:
            results[col] = self._correlation_ratio(dataset[col], y)

        return pd.Series(results).dropna()

    def _correlation_ratio(self, categories: pd.Series, values: pd.Series) -> float:
        codes, _ = pd.factorize(categories)
        y = values.to_numpy(dtype=float)

        counts = np.bincount(codes)
        means = np.bincount(codes, weights=y) / counts
        overall_mean = y.mean()

        ss_between = (counts * (means - overall_mean) ** 2).sum()
        ss_total = ((y - overall_mean) ** 2).sum()

        return np.sqrt(ss_between / ss_total)

    def _numeric_corr(self, dataset: DataFrame, target: str, method: str) -> pd.Series:
        num_df = dataset.select_dtypes(include=["number"])
        return num_df.corrwith(num_df[target], method=method).drop(target).dropna()

    def outliers(self, exclude_cols: list, top_n: int = 10):
        dataset = self.dataset.drop(columns=exclude_cols)

        num_cols = dataset.select_dtypes(include=["number"]).columns
        cat_cols = dataset.select_dtypes(include=["object", "category"]).columns
        
        # --- numeryczne ---
        fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, len(num_cols) * 2))
        if len(num_cols) == 1:
            axes = [axes]
        for i, col in enumerate(num_cols):
            sns.boxplot(x=dataset[col], ax=axes[i], orient="h")
            axes[i].set_title(f"Outliers in {col}")
        plt.tight_layout()
        plt.show()
        
        # --- kategoryczne ---
        n = len(cat_cols)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            counts = dataset[col].value_counts()
            if len(counts) > top_n:
                other = pd.Series({"Other": counts[top_n:].sum()})
                counts = pd.concat([counts[:top_n], other])
            
            counts.index.name = None

            sns.barplot(x=counts.values, y=counts.index, ax=axes[i])
            axes[i].set_title(f"{col} (Top {top_n})", fontsize=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
        
        # ukryj puste subplots jeśli kolumn mniej niż nrows*ncols
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=3.0)
        plt.show()


    def geography_salary(self, top_n=15):
        """95. percentyl wynagrodzeń wg lokalizacji pracownika z dynamicznym shrinkage"""
        grouped = self.dataset.groupby("employee_residence")["salary_in_usd"]
        counts = grouped.count()
        quantiles = grouped.quantile(0.95)   # zamiast mean()
        global_q95 = self.dataset["salary_in_usd"].quantile(0.95)

        n_max = counts.max()
        weights = np.log1p(counts) / np.log1p(n_max)
        scores = weights * quantiles + (1 - weights) * global_q95
        top = scores.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=top.values, y=top.index)
        for i, (val, n) in enumerate(zip(top.values, counts.loc[top.index])):
            ax.text(val, i, f" n={n}", va="center", ha="left", fontsize=9)
        plt.title(f"Top {top_n} employee residences by adjusted 95th percentile salary (dynamic shrinkage)")
        plt.show()
