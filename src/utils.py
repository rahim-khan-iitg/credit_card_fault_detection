import pickle
import keras

def save_object(file_path,obj):
    pickle.dump(obj,open(file_path,'wb'))

def load_model(file_path):
    model=keras.models.load_model(file_path)
    return model

def load_preprocessor(file_path):
    preprocessor=pickle.load(open(file_path,'rb'))
    return preprocessor