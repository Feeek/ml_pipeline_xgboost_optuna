import os
import kagglehub
from kagglehub import KaggleDatasetAdapter


class DatasetLoader():

	def __init__(self):
		os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(".kaggle-cache")
		
		self.path = "ruchi798/data-science-job-salaries"
		self.csv = "ds_salaries.csv"

	def load(self):
		return kagglehub.load_dataset(
			KaggleDatasetAdapter.PANDAS, self.path, self.csv
		)
