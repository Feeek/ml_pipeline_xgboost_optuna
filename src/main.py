from eda import EDA
from etl import ETLPipeline
from feature_eng import FeatureEngineer


etl = ETLPipeline()

etl.extract("ruchi798/data-science-job-salaries", "ds_salaries.csv")
etl.extract("sazidthe1/data-science-salaries", "data_science_salaries.csv")
etl.extract("arnabchaki/data-science-salaries-2025", "salaries.csv")

etl.transform()

dataset = etl.load()


eda = EDA(dataset)
eda.describe()
eda.correlations(target="salary_in_usd", exclude_cols=["salary"])
eda.outliers(exclude_cols=["work_year", "salary"], top_n=5)


eng = FeatureEngineer(dataset)
eng.cleanup()
eng.prepare()
eng.cluster_careers("mappings/topics.json")
eng.print_examples()


