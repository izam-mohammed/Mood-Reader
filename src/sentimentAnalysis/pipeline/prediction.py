import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from sentimentAnalysis.utils.common import load_pickle


class PredictionPipeline:
    def __init__(self):
        self.model = load_pickle(path=Path('artifacts/model_trainer/model.pkl'))
        self.vectorizer = load_pickle(path=Path('artifacts/data_transformation/tfidf_vectorizer.pkl'))
        
    def _sentiment(self, prediction):
        if prediction == 1:
            return "Positive"
        else:
            return "Negative"
    
    def predict(self, data):
        try:
            matrix = self.vectorizer.transform(data)
            prediction = self.model.predict(matrix)
            sentiment = self._sentiment(prediction)
            print('predicting')
            return sentiment
        
        except Exception as e:
            raise e
        