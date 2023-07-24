from src.logger import logging
from src.exceptions import CustomException
from sklearn.metrics import accuracy_score
# from src.utils import save_object
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os,sys
import keras 
from keras.layers import Dense
from keras.models import Sequential

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.h5")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def model(self,x_train):
        layers_1=[
        keras.layers.Input(shape=x_train.shape[1:]),
        Dense(50,activation='relu'),
        Dense(30,activation='relu'),
        Dense(10,activation='relu'),
        Dense(2,activation='softmax')
        ]
        model=Sequential(layers_1)
        model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        return model

    def initiate_model_training(self,x_train,x_test,y_train,y_test):
        try:
            logging.info("model training initiated")
            model=self.model(x_train)
            logging.info("fitting the model")
            model.fit(x_train,y_train,batch_size=10,epochs=100,verbose=2)
            logging.info("saving the model")
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info("evaluating model")
            y_pred=model.predict(x_test)
            y_pred=np.argmax(y_pred,axis=1)
            acc=accuracy_score(y_test,y_pred)
            logging.info(f"model fitted with test accuracy {acc*100}%")

        except Exception as e:
            logging.info("error occured during model training")