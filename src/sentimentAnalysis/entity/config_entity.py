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


# enitity
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