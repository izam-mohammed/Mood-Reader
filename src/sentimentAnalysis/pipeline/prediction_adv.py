from pathlib import Path
from sentimentAnalysis.utils.common import load_pickle
from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.prediction_adv import Prediction


class PredictionPipeline:
    def __init__(self):
        """
        Initialize PredictionPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the prediction pipeline.

        Returns:
            None
        """
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction_config = Prediction(config=prediction_config)
        prediction_config.predict()
