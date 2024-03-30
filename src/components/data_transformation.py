import os
import sys
import yaml
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from statistics import variance

from utils import choose, save_object, load_config

@dataclass
class DataTransformationConfig:
    config = load_config()
    preprocessor_obj_file_path=config['data-transformation']['preprocessor-obj-file-path']
    SMOTE_pipeline_obj_file_path =config['data-transformation']['SMOTE-pipeline-obj-file-path']
    variance_threshold=config['data-transformation']['variance-threshold']
    SMOTE_sampling_strategy =config['data-transformation']['SMOTE-sampling_strategy']
    undesampling_sampling_strategy =config['data-transformation']['undesampling-strategy']  

class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            
            num_cols = [' Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std', 'Bwd Packet Length Max',
       ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
       ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
       ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
       ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
       ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
       ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
       ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
       ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
       ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
       ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
       'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
       ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
       ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
            
            cat_cols = [' Label']

            num_pipeline= Pipeline(
                steps=[
                    ("variance_filter",VarianceThreshold(threshold=self.DataTransformationConfig.variance_threshold)),
                    ("standard_scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, num_cols)
                ]
            )
            
            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            logging.info("Reading train and test initiated")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test complete")


            preprocessing_obj=self.get_data_transformer_object()
            target_column_name=" Label"

            train_df[' Label'] = train_df[' Label'].apply(choose)
            test_df[' Label'] = test_df[' Label'].apply(choose)

            cols_to_drop=['WeekDay', 'TimeOfDay']
            train_df.drop(columns=cols_to_drop, axis=1, inplace=True)
            test_df.drop(columns=cols_to_drop, axis=1, inplace=True)


            Xtrain=train_df.drop(columns=target_column_name,axis=1)
            ytrain = train_df[target_column_name]            

            Xtest=test_df.drop(columns=target_column_name,axis=1)
            ytest = test_df[target_column_name]            


            logging.info("preprocessing on train and test input started")            
            
            Xtrain=preprocessing_obj.fit_transform(Xtrain)
            Xtest=preprocessing_obj.transform(Xtest)

            logging.info("Applying preprocessor on train and test input completed")

            logging.info("Applying SMOTE and UnderSampling on training data.")

            over = SMOTE(sampling_strategy=0.3)
            under = RandomUnderSampler(sampling_strategy=0.5)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            Xtrain, ytrain = pipeline.fit_resample(Xtrain, ytrain)

            logging.info("Applying SMOTE and UnderSampling on training data completed.")

            train_arr = np.c_[Xtrain, np.array(ytrain)]
            test_arr = np.c_[Xtest, np.array(ytest)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved.")

            save_object(
                file_path=self.data_transformation_config.SMOTE_pipeline_obj_file_path,
                obj=pipeline
            )

            logging.info("SMOTEpipeline object saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.SMOTE_pipeline_obj_file_path
            )
            
            
        except Exception as e:
            
            raise CustomException(e,sys)