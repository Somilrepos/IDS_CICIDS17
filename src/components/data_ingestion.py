import os
import sys
from src.exception import CustomException
from src.logger import logging

import re
import pandas as pd 
import numpy as np
from src.utils import getData, cleanData
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("data", "MachineLearningCVE")

class DataIngestion:

    def __init__(self, test_size, seed):
        self.DataIngestionConfig = DataIngestionConfig
        self.test_size = test_size
        self.seed = seed

    def initiate_data_ingestion(self):

        try:
            logging.info("Reading data as datafame")
            df = getData(self.DataIngestionConfig.raw_data_path)
            logging.info("Reading data completed")


            logging.info("Cleaning data initiated")
            df = cleanData(df)
            logging.info("Cleaning data completed")


            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=self.test_size, random_state=self.seed)

            train_set.to_csv(self.DataIngestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.DataIngestionConfig.test_data_path, index=False, header=True)
            logging.info("Train test split completed")

            return (
                self.DataIngestionConfig.train_data_path,
                self.DataIngestionConfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    