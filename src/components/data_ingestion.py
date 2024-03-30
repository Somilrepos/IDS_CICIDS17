import os
import sys
import yaml
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import re
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import getData, cleanData, load_config

@dataclass
class DataIngestionConfig:
    config = load_config()
    train_data_path: str=config['data-ingestion']['train-data-path']
    test_data_path: str=config['data-ingestion']['test-data-path']
    raw_data_path: str=config['data-ingestion']['raw-data-path']
    
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
    