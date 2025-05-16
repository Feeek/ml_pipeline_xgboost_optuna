# EDA ds_salaries.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_loader import DatasetLoader
from pandas import DataFrame

# Wczytanie danych z pliku CSV
loader = DatasetLoader()
df: DataFrame = loader.load()

print("\nBraki danych:")
print(df.isnull().sum())

# 3. Statystyki opisowe
print("\nStatystyki opisowe:")
print(df.describe())

# 4. Korelacje (tylko zmienne liczbowe)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Macierz korelacji")
plt.tight_layout()
plt.show()

# 5. Rozkład wynagrodzenia
g = sns.FacetGrid(df, col="work_year", col_wrap=3, height=4, sharex=True, sharey=True)
g.map(sns.histplot, "salary_in_usd", kde=True, bins=30)
g.figure.suptitle("Rozkład pensji w USD na przestrzeni lat", fontsize=16)
g.tight_layout()
g.figure.subplots_adjust(top=0.9)  # żeby tytuł się nie nakładał
plt.show()

# 6. Box ploty wynagrodzeń
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    x="work_year",
    y="salary_in_usd",
    data=df
)

ax.set_title("Porównanie wynagrodzeń w USD między latami")
ax.set_xlabel("Rok")
ax.set_ylabel("Wynagrodzenie w USD")

medians = df.groupby("work_year")["salary_in_usd"].median()
for idx, (year, median) in enumerate(medians.items()):
    ax.text(
        idx, median,
        f"{median:,.0f}",
        ha="center",
        va="bottom",
        fontsize="small",
        fontweight="semibold",
        color="white"
    )

plt.tight_layout()
plt.show()