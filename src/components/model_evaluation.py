import os
import sys
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import logging
from src.exception import customexception
from src.utils.utils import load_object



class ModelEvaluation:
    def __init__(self):
        pass
    def eval_metrics(self,actual,pred):
        try:
            
            rmse=np.sqrt(mean_squared_error(actual,pred))
            mae=mean_absolute_error(actual,pred)
            r2=r2_score(actual,pred)
            return rmse,mae,r2
        except Exception as e:
            raise customexception(e, sys)
    def iniatiate_model_evaluation(self,train_arr,test_arr):
        try:
            
            # Split features and target from test array
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
             # Load the trained model
            model_path=os.path.join('Artifacts','Model.pkl')
            model=load_object(model_path)
            logging.info("loaded the trained model for evaluation")
            
            # Predict using the model
            predictions = model.predict(X_test)
            logging.info("Predictions made on test data.")
            
            # Evaluate metrics
            rmse, mae, r2 = self.eval_metrics(y_test, predictions)
            logging.info(f"Evaluation Metrics: RMSE={rmse}, MAE={mae}, R2={r2}")
            
            return {"rmse": rmse, "mae": mae, "r2": r2}
        except Exception as e:
            logging.error("Error occurred during model evaluation.")
            raise customexception(e, sys)
        
    

