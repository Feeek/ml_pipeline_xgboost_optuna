import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
import json


class RegionEDA():
    def __init__(self, engl_nations: DataFrame, rest_nations: DataFrame, order_data: str):
        with open(order_data, "r", encoding="utf-8") as f:
            self.order = json.load(f)

        self.engl_dataset = engl_nations
        self.rest_dataset = rest_nations

    def _plot_two(self, plot_func, title: str, figsize=(14, 6)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # lewy panel: US+CA+UK
        plot_func(self.engl_dataset, axes[0], "US+CA+UK")

        # prawy panel: Rest
        plot_func(self.rest_dataset, axes[1], "Rest")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


    def geography_salary(self):
        def _plot(df, ax, label):
            geo = (
                df.groupby("employee_residence")["salary_in_usd"]
                .median()
                .sort_values(ascending=False)
            )
            sns.barplot(x=geo.values, y=geo.index, ax=ax)
            ax.set_title(f"{label}: median salary")
            ax.set_xlabel("Median salary (USD)")

        self._plot_two(_plot, "Median salaries by employee residence")

    def field_profiles(self):
        cols = [c for c in self.engl_dataset.columns if c.startswith("p_")]

        def _plot(df, ax, label):
            field_strength = df[cols].mean().sort_values(ascending=False)
            sns.barplot(x=field_strength.index, y=field_strength.values, ax=ax)
            ax.set_title(f"{label}: average field scores")
            ax.set_ylim(0, 1)

        self._plot_two(_plot, "Average field scores across dataset")

        def _plot_corr(df, ax, label):
            corr = df[cols + ["salary_in_usd"]].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))

            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                square=True,
                ax=ax,
                cbar=False
            )
            ax.set_title(f"{label}: corr fields ↔ salary")

        self._plot_two(_plot_corr, "Correlation between fields and salary", figsize=(8, 6))

    def salary_distributions(self):
        def _plot(df, ax, label):
            sns.histplot(df["salary_in_usd"], bins=40, kde=True, ax=ax)
            ax.set_title(f"{label}: salary distribution")

        self._plot_two(_plot, "Distribution of salaries (USD)")

    def salary_vs_work_model(self):
        def _plot(df, ax, label):
            sns.boxplot(
                data=df,
                x="work_model",
                y="salary_in_usd",
                order=self.order["work_model"],
                ax=ax
            )
            ax.set_title(f"{label}: salaries by work model")

        self._plot_two(_plot, "Work model comparison")

    def salary_vs_company_size(self):
        def _plot(df, ax, label):
            sns.violinplot(
                data=df,
                x="company_size",
                y="salary_in_usd",
                order=self.order["company_size"],
                ax=ax
            )
            ax.set_title(f"{label}: salaries by company size")

        self._plot_two(_plot, "Company size comparison")

    def salary_trends(self):
        def _plot(df, ax, label):
            yearly = df.groupby("work_year")["salary_in_usd"].median()
            sns.lineplot(x=yearly.index, y=yearly.values, marker="o", ax=ax)
            ax.set_title(f"{label}: median salary trend")
            ax.set_ylabel("Median salary (USD)")

        self._plot_two(_plot, "Salary trends over years")
