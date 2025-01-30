import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import customexception
from src.utils.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            # Log before loading preprocessor and model
            logging.info("Loading preprocessor and model...")
            preprocessed_path = os.path.join('Artifacts', 'preprocessed.pkl')
            model_path = os.path.join('Artifacts', 'model.pkl')
            
            # Load preprocessor and model
            preprocessor = load_object(preprocessed_path)
            model = load_object(model_path)
            
            # Log transformation step
            logging.info("Transforming input features using the preprocessor...")
            scaled_data = preprocessor.transform(features)
            
            # Log prediction step
            logging.info("Making predictions using the model...")
            pred = model.predict(scaled_data)
            return pred

        except Exception as e:
            logging.error(f"Error occurred in prediction: {str(e)}")
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):  
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
        
    def get_data_dataframe(self):
        try:
            # Ensure that all data values are provided
            if any(v is None or v == '' for v in [self.carat, self.depth, self.table, self.x, self.y, self.z, self.cut, self.color, self.clarity]):
                raise ValueError("All input values must be provided.")
                
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
                
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered successfully')
            return df
        except Exception as e:
            logging.error(f"Error occurred while creating dataframe: {str(e)}")
            raise customexception(e, sys)
