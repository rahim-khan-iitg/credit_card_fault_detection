from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj1=DataIngestion()
    train_path,test_path=obj1.initiate_data_ingestion()
    print(train_path,test_path)
    obj2=DataTransformation()
    x_train,x_test,y_train,y_test= obj2.initiate_data_transformation(train_path,test_path)
    obj3=ModelTrainer()
    obj3.initiate_model_training(x_train,x_test,y_train,y_test)
