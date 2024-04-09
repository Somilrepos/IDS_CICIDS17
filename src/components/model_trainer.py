import os
import sys
import yaml
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import load_config

@dataclass
class ModelTrainerConfig:
    config = load_config()['model-trainer']
        
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def train_models(self, X, y, models, params, scoring, refit):
        """ Training function to generate a model report based on given parameters

        Args:
            X (Numpy array): Input data
            y (Numpy array): Output labels
            models (dict[sklearn-model]): Set of models to train
            params (dict[model-param]): Set of model training parameters 
            scoring (dict[sklearn.metrics]): Scoring function for model evaluation
            refit (str): Final model evaluation metric

        Returns:
            dict[model-train-result]: Set of model training results
        """
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para=params[list(models.keys())[i]]

                gs = GridSearchCV(model, 
                                  param_grid=params, 
                                  cv=self.model_trainer_config.config["cross-validations"], 
                                  scoring=scoring,
                                  refit=refit)
                
                gs.fit(X,y)

                model.set_params(**gs.best_params_)
                model.fit(X,y)

                report[list(models.keys())[i]] = gs.cv_results_

            return report

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_models(self, X, y, scoring_functions):
        report = {}
        
        for score_name, score_fn in scoring_functions:
            report[score_name] = score_fn(X, y)
        
        return report
    
    
    def initiate_model_trainer(self,train_array,test_array):
        """Model trainer function for the given data.

        Args:
            train_array (Numpy array): Training data
            test_array (Numpy array): Testing data
            
        Returns:
            dict: training report
        """
        try:
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                "Random Forest": RandomForestClassifier()  
            }
            
            params={
                "Random Forest": self.model_trainer_config.config['random-forest']       
            }
            
            scoring = self.model_trainer_config.config['scoring']
            
            train_report = self.train_models(X=X_train,
                              y=y_train,
                              models=models,
                              params=params,
                              scoring=scoring, 
                              refit=self.model_trainer_config.config['refit']
            )

            return train_report
            
            
        except Exception as e:
            raise CustomException(e, sys)