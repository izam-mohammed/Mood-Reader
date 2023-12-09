from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
    delimeter: str
    target_out: list
    target_col: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    delimiter: str
    text_corpus_name: str
    target_col_encoded_file: str
    vectorizer_name: str
    vocabulary_name: str
    test_size: float


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_params: dict
    target_column: str
    vectorizer_path: str
    vocabulary_path: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    vectorizer_path: str


@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    model_path: Path
    vectorizer_path: str
    data_path: str
    prediction_file: str
