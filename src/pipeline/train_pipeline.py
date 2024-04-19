import logging
import sys

from src import logger
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        pass
    
    def train(self, test_size, SEED):
        logging.info("Training Called.")
        
        logging.info("Initiating data ingestion.") 
        ingestionObj = DataIngestion(test_size=test_size,
                            seed=SEED)
        train_path, test_path = ingestionObj.initiate_data_ingestion()
        logging.info("data ingestion completed.") 
        
        logging.info("Initiating data transformation.") 
        transfromObj = DataTransformation()
        train_arr, test_arr = transfromObj.initiate_data_transformation(train_path=train_path,
                                                                        test_path=test_path)
        logging.info("Data transformation completed.") 
        
        logging.info("Initiating model training")
        trainObj = ModelTrainer()
        train_report = trainObj.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed.")

        return train_report
    
# if __name__ == '__main__':
#     print("Hi there!")