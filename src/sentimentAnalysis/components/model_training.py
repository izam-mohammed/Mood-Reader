import pandas as pd
import os
from sentimentAnalysis import logger
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sentimentAnalysis.utils.common import load_pickle, save_pickle

from sentimentAnalysis.entity.config_entity import ModelTrainerConfig
from pathlib import Path


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize ModelTrainer instance.

        Parameters:
        - config: ModelTrainerConfig
            Configuration object containing necessary parameters.
        """
        self.config = config

    def train(self):
        """
        Train a model using the provided training data and save the best model.

        Returns:
            None
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        y_test = test_data[self.config.target_column]
        print(self.config.vocabulary_path)
        vocabulary = load_pickle(path=Path(self.config.vocabulary_path))
        vectorizer = load_pickle(path=Path(self.config.vectorizer_path))
        vectorizer.vocabulary_ = vocabulary

        X_train = vectorizer.transform(X_train["text"])

        logger.info(
            f"Transformed the X_train and X_test, new shape of X_train - {X_train.shape}"
        )

        grid_search = GridSearchCV(
            MultinomialNB(),
            self.config.model_params,
            cv=5,
            return_train_score=True,
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)
        logger.info(f"found best mode at {grid_search.best_params_}")

        model = grid_search.best_estimator_

        save_pickle(
            path=Path(os.path.join(self.config.root_dir, self.config.model_name)),
            data=model,
        )
