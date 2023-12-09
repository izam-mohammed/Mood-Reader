import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentimentAnalysis.utils.common import load_pickle
from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.prediction import Prediction


class PredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction_config = Prediction(config=prediction_config)
        prediction_config.predict()
