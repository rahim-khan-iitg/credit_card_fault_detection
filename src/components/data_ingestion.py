import os,sys
from src.logger import logging
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd 

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts",'train.csv')
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion is started")
        try:
            df=pd.read_csv(os.path.join("notebooks/data",'data.csv'))
            logging.info("data read in the dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("train test split")

            train_set,test_set=train_test_split(df,test_size=0.20,random_state=20)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("error occured during the data ingestion")
            raise CustomException(e,sys)
