from etl import ETLPipeline

# Data extraction / transformation / loading
etl = ETLPipeline()

etl.extract("ruchi798/data-science-job-salaries", "ds_salaries.csv")
etl.extract("sazidthe1/data-science-salaries", "data_science_salaries.csv")
etl.extract("arnabchaki/data-science-salaries-2025", "salaries.csv")

etl.transform()

dataset = etl.load()


missing_values = dataset.isnull().sum().sum()
print(f"Missing values or NaNs: {missing_values}")

