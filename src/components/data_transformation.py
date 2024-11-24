import os
import sys
import pandas as  pd
import numpy as np

from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path :str=os.path.join('Artifacts','prepocessed.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info("Iniatiated Data Transformation")  
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Pipeline is iniatiated")
            
            ##Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )   
            ##Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('ordinal-encoded',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                    
                ]
            )
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )
            return preprocessor
        except Exception as e:
             logging.info("Exception occured in the initiate_datatransformation")
             raise customexception(e,sys)
         
    def iniatiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)  
            test_df=pd.read_csv(test_path) 
            logging.info('Read the Train and Test Data') 
            preprocessing_obj=self.get_data_transformation()
            target_column_name='price'
            drop_columns=[target_column_name,'id']
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on both training and testing dataset")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessed object as pickle")
            logging.info ("Data transformation completed")
            save_object(
                file_path=self.data_transformation_config.preprocessed_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
          