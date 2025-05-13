# EDA ds_salaries.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych z pliku CSV
df = pd.read_csv("ds_salaries.csv")

# === 1. Podstawowe informacje ===
print("Rozmiar danych (wiersze, kolumny):", df.shape)
print("\nPodgląd danych:")
print(df.head())

# === 2. Typy danych i braki ===
print("\nTypy danych:")
print(df.dtypes)

print("\nBraki danych:")
print(df.isnull().sum())

# === 3. Statystyki opisowe ===
print("\nStatystyki opisowe:")
print(df.describe())

# === 4. Korelacje (tylko zmienne liczbowe) ===
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Macierz korelacji")
plt.tight_layout()
plt.show()

# === 5. Rozkład wynagrodzenia ===
plt.figure(figsize=(8, 4))
sns.histplot(df["salary_in_usd"], kde=True, bins=30)
plt.title("Rozkład pensji (USD)")
plt.xlabel("Wynagrodzenie w USD")
plt.ylabel("Liczba wystąpień")
plt.tight_layout()
plt.show()


