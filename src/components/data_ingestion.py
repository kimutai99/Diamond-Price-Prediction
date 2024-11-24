import os 
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging

from src.exception import customexception

# from src.components.data_transformation import DataTransformationConfig
# from src.components.data_transformation import DataTransformation

from dataclasses import dataclass
#from pathlib import path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join('Artifacts','raw_data.csv')
    train_data_path:str=os.path.join('Artifacts','train_data.csv')
    test_data_path:str=os.path.join('Artifacts','test_data.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()  
        
    def iniatiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df=pd.read_csv('Notebook\\Data\\gemstone.csv')
            logging.info("Read the data as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Created the raw data file")
            
            logging.info("split the dataset into trained test split")
            train_data,test_data=train_test_split(df,test_size=0.20,random_state=42)
            logging.info("Data splitting is done")
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Created both train and test data files")
            logging.info("Data ingestion Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured while ingestion the data")
            raise customexception(e,sys)
# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data,test_data= obj.iniatiate_data_ingestion()  
    
#     data_transformation=DataTransformation() 
#     data_transformation.iniatiate_data_transformation(train_data,test_data)       
