import sys
import os
from typing import Any
import pandas as pd
import numpy as np
from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_model, load_preprocessor


class PredictPipeline:
    def __call__(self):
        pass

    def predict(self, features):
        try:
            logging.info("loading the model and preprocessor")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.h5")
            model = load_model(model_path)
            preprocessor = load_preprocessor(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            data_scaled=data_scaled[:,:-1]
            # print(data_scaled)
            pred = model.predict(data_scaled)
            prediction = np.argmax(pred)
            logging.info("prediction made successfully")
            return prediction

        except Exception as e:
            logging.info("error occured during prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self):
        self.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                        'default payment next month']
    
    def get_dataframe(self,data):
        df=pd.DataFrame(data)
        return df


