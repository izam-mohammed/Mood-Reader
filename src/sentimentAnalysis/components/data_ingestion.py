import os
import urllib.request as request
import zipfile
from sentimentAnalysis import logger
from sentimentAnalysis.utils.common import get_size
from sentimentAnalysis.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    A class for Data Ingestion
    """

    def __init__(self, config: DataIngestionConfig):
        """Initialize the data ingestion

        Args:
            config: configuration file for data ingestion
        """
        self.config = config

    def download_file(self):
        """Download the data file and save it

        Args:
            None

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """Extracts the zip file into the data directory

        Args:
            None

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
