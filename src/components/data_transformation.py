from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
from dataclasses import dataclass
import os,sys
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("obtaining data transformation object")
            pipe=Pipeline(
                steps=[
                    ("imputer",KNNImputer(n_neighbors=3)),
                    ("scaler",RobustScaler()),
                ]
            )
            return pipe
        except Exception as e:
            logging.info("error occured during get_data_transformation_object")
            raise CustomException(e,sys)
    def resample(self,data:pd.DataFrame):
        try:
            logging.info("resampling initiated")
            majority_class=data[data['default payment next month']==0]
            minority_class=data[data["default payment next month"]==1]
            if len(majority_class)<len(minority_class):
                majority_class,minority_class=minority_class,majority_class
            minority_class_resampled=resample(minority_class,replace=True,n_samples=len(majority_class),random_state=40)
            resampled_data=pd.concat([majority_class,minority_class_resampled],ignore_index=True)
            logging.info(f"resampled data \n{resampled_data.head().to_string()}")
            return resampled_data
        except Exception as e:  
            logging.info("error occured during data transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("initiating data transformation")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            preprocessor=self.get_data_transformation_object()
            train_data_scaled=preprocessor.fit_transform(train_data)
            test_data_scaled=preprocessor.transform(test_data)
            train_data=pd.DataFrame(train_data_scaled,columns=preprocessor.get_feature_names_out())
            test_data=pd.DataFrame(test_data_scaled,columns=preprocessor.get_feature_names_out())
            logging.info(f"train data head\n{train_data.head(3).to_string()}")
            logging.info(f"test data head\n{test_data.head(3).to_string()}")
            train_data=self.resample(train_data)
            x_train=train_data.drop("default payment next month",axis=1)
            y_train=train_data['default payment next month']
            x_test=test_data.drop("default payment next month",axis=1)
            y_test=test_data['default payment next month']
            logging.info("saving the preprocessor")
            save_object(self.data_transformation_config.preprocessor_obj_file,preprocessor)
            logging.info("data transformation completed")
            return (
                x_train,x_test,y_train,y_test
            )
        except Exception as e:
            logging.info("error occured during data transformation")
            raise CustomException(e,sys)