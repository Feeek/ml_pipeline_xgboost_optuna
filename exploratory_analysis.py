from dataset_loader import DatasetLoader
from pandas import DataFrame

import numpy as np

loader = DatasetLoader()
df: DataFrame = loader.load()

print("\nStatystyki opisowe:")
describe: DataFrame = df.describe()
describe = describe.applymap(lambda x: f'{x:,.2f}')
print(describe)

print("\nLiczba brakujących danych:")
print(df.isnull().sum())

for col in df.select_dtypes(include='object').columns:
    print(f"\nTop wartości w kolumnie '{col}':")
    print(df[col].value_counts().head())

def detect_outliers_iqr(df: DataFrame, threshold: float = 1.5):
    outlier_indices = {}
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            outlier_indices[col] = outliers.index.tolist()

    return outlier_indices

print("\nOutliery wg IQR:")
outliers = detect_outliers_iqr(df)
for col, idxs in outliers.items():
    print(f"{col}: {len(idxs)} wartości odstających")
