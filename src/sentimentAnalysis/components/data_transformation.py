import os
from sentimentAnalysis import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

from sentimentAnalysis.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize DataTransformation instance.

        Parameters:
            config: DataTransformationConfig
                Configuration object containing necessary parameters.
        """
        self.config = config

    def target_encode(self):
        """
        Encode the target column and save the result to a new CSV file.

        Returns:
            None
        """
        df = pd.read_csv(self.config.data_path, delimiter=self.config.delimiter)

        df.replace(to_replace="surprise", value=1, inplace=True)
        df.replace(to_replace="love", value=1, inplace=True)
        df.replace(to_replace="joy", value=1, inplace=True)
        df.replace(to_replace="fear", value=0, inplace=True)
        df.replace(to_replace="anger", value=0, inplace=True)
        df.replace(to_replace="sadness", value=0, inplace=True)

        df.to_csv(
            os.path.join(self.config.root_dir, self.config.target_col_encoded_file),
            index=False,
        )

        logger.info("encoded the target")
        logger.info("surprise, love, joy ---> 1")
        logger.info("fear, anger, sadness ---> 0")

    def clean_text_data(self):
        """
        Transform text data using TF-IDF vectorization and save the transformer and vocabulary.

        Returns:
            None
        """
        try:
            data = pd.read_csv(
                os.path.join(self.config.root_dir, self.confi.target_col_encoded_file)
            )
            logger.info("using the target column encoded df for splitting")
        except Exception as e:
            data = pd.read_csv(self.config.data_path, delimiter=self.config.delimiter)
            logger.info(f"exception {e} found when access the target column encoded df")
            logger.info(f"using the {self.config.data_path} file for splitting")

        logger.info("cleaning corpus started")

        lm = WordNetLemmatizer()
        corpus = []
        df_col = data["text"]
        for i in df_col:
            new_item = re.sub("[^a-zA-Z]", " ", str(i))  # taking only characters
            new_item = new_item.lower()  # lowering the text
            new_item = new_item.split()  # splitting the text into words
            # lemmatize with remove stop words
            new_item = [
                lm.lemmatize(word)
                for word in new_item
                if word not in set(stopwords.words("english"))
            ]
            corpus.append(" ".join(str(x) for x in new_item))  # back to sentence

        with open(
            os.path.join(self.config.root_dir, self.config.text_corpus_name),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            for i in corpus:
                writer.writerow([i])

        logger.info(f"Saved text corpus {self.config.text_corpus_name}")

    def transform_text(self):
        """
        Split the data into training and test sets and save them as CSV files.

        Returns:
            None
        """

        # access the clean df
        try:
            corpus = pd.read_csv(
                os.path.join(self.config.root_dir, self.config.text_corpus_name),
                names=["text"],
                header=None,
            )
            logger.info("using the cleaned text corpus for vectorizing")
        except Exception as e:
            corpus = pd.read_csv(self.config.data_path, delimiter=self.config.delimiter)
            corpus.drop(["label"], axis=1, inplace=True)
            logger.info(f"exception {e} found when access the cleaned text df")
            logger.info(f"using the {self.config.data_path} file for vectorizing")

        transformer = TfidfVectorizer()
        tfidf_matrix = transformer.fit_transform(corpus["text"])

        # Save the TF-IDF vectorizer to a file using pickle
        with open(
            os.path.join(self.config.root_dir, self.config.vectorizer_name), "wb"
        ) as file:
            pkl.dump(transformer, file)

        # Save the vocabulary separately
        with open(
            os.path.join(self.config.root_dir, self.config.vocabulary_name), "wb"
        ) as file:
            pkl.dump(transformer.vocabulary_, file)

        logger.info(
            f"saved the tfidf transformer with {len(transformer.vocabulary_)} words of vocabulary"
        )

    def train_test_spliting(self):
        # access hte target col encoded df
        try:
            data = pd.read_csv(
                os.path.join(self.config.root_dir, self.config.target_col_encoded_file)
            )
            logger.info(
                f"using the {self.config.target_col_encoded_file} for splitting"
            )
        except Exception as e:
            data = pd.read_csv(self.config.data_path, delimiter=self.config.delimiter)
            logger.info(f"exception {e} found when access the target column encoded df")
            logger.info(f"using the {self.config.data_path} file for splitting")

        # access the clean df
        try:
            corpus = pd.read_csv(
                os.path.join(self.config.root_dir, self.config.text_corpus_name),
                names=["text"],
                header=None,
            )
            logger.info("using the cleaned text corpus for splitting")
        except Exception as e:
            corpus = pd.read_csv(self.config.data_path, delimiter=self.config.delimiter)
            corpus.drop(["label"], axis=1, inplace=True)
            logger.info(f"exception {e} found when access the cleaned text df")
            logger.info(f"using the {self.config.data_path} file for splitting")

        # Split the data into training and test sets. (0.75, 0.25) split.
        X_train, X_test, y_train, y_test = train_test_split(
            corpus, data["label"], test_size=self.config.test_size
        )

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
