import numpy as np
from sentimentAnalysis import logger
from sentimentAnalysis.entity.config_entity import PredictionConfig
from sentimentAnalysis.utils.common import load_pickle, read_text, save_json
from pathlib import Path


class Prediction:
    def __init__(self, config: PredictionConfig):
        """
        Initialize Prediction instance.

        Parameters:
        - config: PredictionConfig
            Configuration object containing necessary parameters.
        """
        self.config = config

    def predict(self):
        """
        Use the trained model to predict sentiment for new data and save the prediction.

        Returns:
            None
        """
        model = load_pickle(Path(self.config.model_path))
        vectorizer = load_pickle(Path(self.config.vectorizer_path))
        try:
            data = read_text(path=Path(self.config.data_path))
        except Exception as e:
            logger.info(f"file not found in {self.config.model_path}")
            logger.info("using the word 'no' instead")
            data = ["no"]
        data = np.array(data)[:, 0]

        matrix = vectorizer.transform(data)
        prediction = model.predict(matrix)
        logger.info(f"predicted the new data as {prediction[0]}")

        save_json(
            path=self.config.prediction_file, data={"prediction": "Positive" if prediction[0] else "Negative"}
        )
