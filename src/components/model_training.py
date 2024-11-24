import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
from dataclasses import dataclass
from src.utils.utils import save_object
from src.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('Artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def iniatiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')  
            
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                
            ) 
            models={
                'LinearRgression':LinearRegression(),
                'Ridge':Ridge(),
                'lasso':Lasso(),
                'ElasticNet':ElasticNet()
                
            }  
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("="*35)
            print('\n') 
            logging.info(f'model_report:{model_report}')
            
            ##to get best model score from dictionanry
            best_model_score=max(sorted(model_report.values()))
            
            ##to get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception  as e:
            raise customexception(e,sys)    