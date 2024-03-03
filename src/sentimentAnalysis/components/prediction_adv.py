import numpy as np
from sentimentAnalysis import logger
from sentimentAnalysis.entity.config_entity import PredictionConfig
from sentimentAnalysis.utils.common import load_pickle, read_text, save_json
from pathlib import Path
from transformers import pipeline

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
        model = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
        try:
            data = read_text(path=Path(self.config.data_path))
        except Exception as e:
            logger.info(f"file not found in {self.config.model_path}")
            logger.info("using the word 'no' instead")
            data = ["no"]
        data = data[0]
        logger.info(f"data - {data}")
        results = model(data)
        sentiment = results[0]["label"]
        
        logger.info(f"predicted the new data as {sentiment}")

        save_json(
            path=self.config.prediction_file, data={"prediction": sentiment}
        )

if __name__ == "__main__":
    model = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
    result = model("hai")
    print(result["label"])