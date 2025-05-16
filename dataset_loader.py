import os
from typing import Tuple
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from pandas import DataFrame


class DatasetLoader():

	def __init__(self):
		os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(".kaggle-cache")
		
		self.path = "ruchi798/data-science-job-salaries"
		self.csv = "ds_salaries.csv"

	def load(self, predict: str = None) -> \
		DataFrame | Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:

		df: DataFrame = kagglehub.load_dataset(
			KaggleDatasetAdapter.PANDAS, self.path, self.csv
		)
		df = df.iloc[:, 1:]

		if predict:
			df = self._encode(df)

			x = df.drop(columns=predict)
			y = df[predict]

			return train_test_split(x, y, test_size=0.2, random_state=42)

		else:
			return df

	def _encode(self, df: DataFrame) -> DataFrame:
		obj_cols = df.select_dtypes(include='object').columns
		df[obj_cols] = OrdinalEncoder().fit_transform(df[obj_cols])

		return df
