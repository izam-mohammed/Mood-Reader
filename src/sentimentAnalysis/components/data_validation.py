import os
from sentimentAnalysis import logger
import pandas as pd
from sentimentAnalysis.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        """
        Initialize DataValidation instance.

        Parameters:
            config: DataValidationConfig
                Configuration object containing necessary parameters.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate if all columns in the dataset match the expected schema.

        Returns:
            bool:
                True if all columns are valid, False otherwise.
        """
        try:
            validation_status = None

            data = pd.read_csv(
                self.config.unzip_data_dir, delimiter=self.config.delimeter
            )
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e

    def validate_labels(self) -> bool:
        """
        Validate if the target labels in the dataset match the expected target labels.

        Returns:
            bool:
                True if labels are valid, False otherwise.
        """
        try:
            data = pd.read_csv(
                self.config.unzip_data_dir, delimiter=self.config.delimeter
            )
            target_out = self.config.target_out
            current_out = data[self.config.target_col].unique()

            validation_label = True
            for i in current_out:
                if i not in target_out:
                    validation_label = False

            with open(self.config.STATUS_FILE, "a") as f:
                f.write(f"\nValidation out labels status: {validation_label}")

            return validation_label

        except Exception as e:
            raise e
