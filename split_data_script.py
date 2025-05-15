
import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych przekształconych po LabelEncoder
df = pd.read_csv("ds_salaries_encoded.csv")

# Podział danych
X = df.drop(columns="salary_in_usd")  # cechy (features)
y = df["salary_in_usd"]               # target (co przewidujemy)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Zapisanie wyników do plików
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Dane zostały podzielone i zapisane jako:")
print("- X_train.csv")
print("- X_test.csv")
print("- y_train.csv")
print("- y_test.csv")
