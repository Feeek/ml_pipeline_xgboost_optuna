import os
import json
import country_converter as coco

import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

class ETLPipeline():

	def __init__(self):
		os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(".kaggle-cache")
		self.datasets: list[DataFrame] = []
		self.cc = coco.CountryConverter()
		self.cc_cache = {}

	def extract(self, url: str, file: str):
		def is_index_column(series: pd.Series) -> bool:
			if not pd.api.types.is_integer_dtype(series):
				return False

			n = len(series)
			return (
				series.tolist() == list(range(n)) or
				series.tolist() == list(range(1, n+1))
			)

		df: DataFrame = kagglehub.load_dataset(
			KaggleDatasetAdapter.PANDAS, url, file
		)

		first_col = df.columns[0]
		if is_index_column(df[first_col]):
			df = df.drop(columns=first_col)

		self.datasets.append(df)

	def transform(self):
		with open(os.path.join("mappings", "columns.json")) as file:
			columns: dict = json.load(file)

		with open(os.path.join("mappings", "values.json")) as file:
			values: dict = json.load(file)

		def map_values(df: DataFrame, column: str):
			if column in df:
				if all(isinstance(k, str) for k in values[column].keys()):
					df[column] = df[column].astype(str)

				df[column] = df[column].replace(values[column])
		
		def normalize_country(value: str) -> str:
			if value in self.cc_cache:
				return self.cc_cache[value]

			self.cc_cache[value] = self.cc.convert(names=value, to="name_short")
			return self.cc_cache[value]

		for dataset in tqdm(self.datasets, desc="Transforming"):
			for key in values.keys():
				map_values(dataset, key)
			
			for column in ["employee_residence", "company_location"]:
				if column in dataset:
					dataset[column] = dataset[column].apply(normalize_country)
			
			dataset.rename(columns=columns, inplace=True)


	def load(self) -> DataFrame:
		return (
			pd.concat(self.datasets, axis=0, ignore_index=True)
			.drop_duplicates()
			.reset_index(drop=True)
			.sort_values("work_year", ascending=True)
		)
