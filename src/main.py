import os

from eda import EDA
from region_eda import RegionEDA
from etl import ETLPipeline
from feature_eng import FeatureEngineer
from xgb import XGBoost


etl = ETLPipeline()

etl.extract("ruchi798/data-science-job-salaries", "ds_salaries.csv")
etl.extract("sazidthe1/data-science-salaries", "data_science_salaries.csv")
etl.extract("arnabchaki/data-science-salaries-2025", "salaries.csv")

etl.transform(
    columns_map=os.path.join("mappings", "columns.json"),
    values_map=os.path.join("mappings", "values.json"),
)

dataset = etl.load()


eda = EDA(dataset)
eda.describe()
eda.correlations(target="salary_in_usd", exclude_cols=["salary"])
eda.outliers(exclude_cols=["work_year", "salary"], top_n=5)
eda.geography_salary()


eng = FeatureEngineer(dataset)
eng.cleanup()
eng.prepare()
eng.cluster_careers(os.path.join("mappings", "topics.json"))
eng.print_examples(n = 10)

dataset = eng.dataset


# ile próbek ma każdy kraj
counts = dataset["employee_residence"].value_counts()

# kraje z min 100 obserwacjami
valid_countries = counts[counts >= 200].index
filtered = dataset[dataset["employee_residence"].isin(valid_countries)].copy()

# przypisz etykiety regionów
filtered["region_group"] = filtered["employee_residence"].apply(
    lambda c: "US+CA+UK" if c in {"United States", "Canada", "United Kingdom"} else "Rest"
)

# podział na regiony P95
def trim_region(df, col="salary_in_usd", q=0.95):
    cutoff = df[col].quantile(q)
    return df[df[col] <= cutoff].copy()

us_ca_uk = trim_region(filtered[filtered["region_group"] == "US+CA+UK"])
rest     = trim_region(filtered[filtered["region_group"] == "Rest"])


report_eda = RegionEDA(us_ca_uk, rest, os.path.join("mappings", "orders.json"))

report_eda.geography_salary()
report_eda.field_profiles()
report_eda.salary_distributions()
report_eda.salary_trends()
report_eda.salary_vs_company_size()
report_eda.salary_vs_work_model()


xgb_us_ca_uk = XGBoost(us_ca_uk)

xgb_us_ca_uk.tune(1)
xgb_us_ca_uk.fit()


xgb_rest = XGBoost(rest)

xgb_rest.tune(1)
xgb_rest.fit()



import matplotlib.pyplot as plt
import shap

# zrób wersję danych numerycznych tylko do SHAP
X_numeric = xgb_rest.X.copy()
for col in X_numeric.select_dtypes(include="category"):
    X_numeric[col] = X_numeric[col].cat.codes

# TreeExplainer dla XGBoost
explainer = shap.TreeExplainer(xgb_rest.model, feature_perturbation="tree_path_dependent")
shap_values = explainer(X_numeric)

# summary plot z tytułem
shap.summary_plot(shap_values, X_numeric, show=False)
plt.title("SHAP Summary - US/CA/UK Model")
plt.show()

# bar plot z tytułem
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Barplot - Feature Importance")
plt.show()

# waterfall dla jednej obserwacji
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Waterfall - Example 0")
plt.show()
