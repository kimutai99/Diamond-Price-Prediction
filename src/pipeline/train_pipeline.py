from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src .components.model_training import ModelTrainer

obj=DataIngestion()
train_data_path,test_data_path=obj.iniatiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.iniatiate_data_transformation(train_data_path,test_data_path)

model_trainer=ModelTrainer()
model_trainer.iniatiate_model_training(train_arr,test_arr)

model_evaluation=ModelEvaluation()
model_evaluation.iniatiate_model_evaluation(train_arr,test_arr)
