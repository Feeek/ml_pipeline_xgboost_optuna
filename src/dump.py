

df: pd.DataFrame = etl.load()

assert df.isnull().all().sum() == 0
# df = df.drop(df.columns[0], axis=1)
df = df.drop(['salary', 'employee_residence'], axis=1)
print(df)

df_encoded = etl.encode_labels(df)

corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 8))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
            xticklabels=df.columns, yticklabels=df.columns)
plt.title("Macierz korelacji (z kolumnami tekstowymi jako kategorie)")
plt.tight_layout()
plt.show()

target = "salary_in_usd"
years = sorted(df["work_year"].unique())

# Ustawiamy układ subplots — 1 wiersz, tyle kolumn ile lat
fig, axes = plt.subplots(1, len(years), figsize=(6 * len(years), 5), sharey=True)

for ax, year in zip(axes, years):
    # Filtrujemy dane dla danego roku
    df_year = df[df["work_year"] == year]
    df_year_encoded = etl.encode_labels(df_year)
    
    # Korelacje względem salary_in_usd
    corr_with_salary = df_year_encoded.corr()[target].drop(target).sort_values(ascending=False)
    
    # Rysujemy barplot
    sns.barplot(x=corr_with_salary.values, y=corr_with_salary.index, ax=ax, palette="coolwarm")
    ax.set_title(f"Korelacja z {target} ({year})")
    ax.set_xlabel("Korelacja")
    ax.set_ylabel("")  # Tylko pierwszy wykres może mieć etykietę osi Y

plt.tight_layout()
plt.show()
