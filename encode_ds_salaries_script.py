# zamiana tekstu na liczby w ds_salaries.csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Wczytanie danych
df = pd.read_csv("ds_salaries.csv")

# Wypisanie oryginalnych kolumn tekstowych
print("Kolumny typu 'object' (tekstowe):")
print(df.select_dtypes(include='object').columns)

# Zastosowanie LabelEncoder do każdej kolumny tekstowej
label_encoders = {}  # słownik do późniejszego odkodowania, jeśli potrzebne

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Zakodowano kolumnę: {col}")

# Podgląd wyników
print("\nPodgląd przekształconych danych:")
print(df.head())

# Zapisanie przekształconego DataFrame do pliku
df.to_csv("ds_salaries_encoded.csv", index=False)
print("\nZapisano plik jako ds_salaries_encoded.csv")
