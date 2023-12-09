import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentimentAnalysis.utils.common import load_pickle, save_json, round_batch
from sentimentAnalysis import logger

from sentimentAnalysis.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    

    def _eval_metrics(self,actual, pred):
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        return round_batch(acc, precision, recall, f1)

    def evaluate(self):
        df = pd.read_csv(self.config.test_data_path)
        X = df.drop([self.config.target_column], axis=1).iloc[:, 0]
        y = df[self.config.target_column]

        vectorizer = load_pickle(path=Path(self.config.vectorizer_path))
        matrix = vectorizer.transform(X)

        model = load_pickle(path=Path(self.config.model_path))
        pred = model.predict(matrix)

        logger.info(f"predicted {pred.shape[0]} data points")

        acc, precision, recall, f1 = self._eval_metrics(y, pred)

        metric = {
            "Accuracy" : acc,
            "Precision" : precision,
            "Recall": recall,
            "F1 score": f1,
        }

        logger.info(f"metrics are - {metric}")

        save_json(path=Path(self.config.metric_file_name), data=metric)