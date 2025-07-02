import argparse
from datetime import datetime

from data_manager import DATA_PATH


class ArgumentParser:
    """
    A class to parse command-line arguments for the feature extraction script.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Feature extraction script.")
        self._add_arguments()

    def _add_arguments(self):
        # Feature extraction arguments
        self.parser.add_argument("--index_model", type=int, default=1, help="Index model")
        self.parser.add_argument("--model_folder", type=str,
                                 default=f"{DATA_PATH}tabddpm_black_box/train/tabddpm_1/",
                                 help="Output model directory")
        self.parser.add_argument("--model_name", type=str, default="tabddpm", help="Output model directory")
        self.parser.add_argument("--output_dir", type=str,
                                 default=f"{DATA_PATH}embeddings_extraction_ML",
                                 help="Output directory")
        self.parser.add_argument("--mode", type=str, default='train', help="Mode: train, dev, or test")
        self.parser.add_argument("--seed", default=42, type=int)
        self.parser.add_argument("--include_account", action='store_true', help="Include account column in training")

        # Classifier training arguments
        self.parser.add_argument("--max_index_classifier_train", type=int, default=25, help="Index model")
        self.parser.add_argument("--classifier_name", type=str, default="MLP",
                                 help="Classifier type: XGBoost, CatBoost, or MLP")
        self.parser.add_argument("--input_embeddings_extraction", type=str,
                                 default="/dt/shabtaia/dt-sicpa/eyal/MIDST/embeddings_extraction_ML",
                                 help="Input embeddings directory")
        self.parser.add_argument("--evaluation_output_dir", type=str,
                                 default="/dt/shabtaia/dt-sicpa/eyal/MIDST/evaluation_ML", help="Output directory")
        self.parser.add_argument("--epoch", default=10, type=int)
        self.parser.add_argument("--save_model", action='store_true', help="Flag to save the trained classifier model")
        self.parser.add_argument("--type_test", type=str, default='blackbox_multi_table',
                                 help="Type of test: test or dev")
        self.parser.add_argument("--columns_lst", nargs='+', type=str,
                                 default=["actual", "error", "error_ratio", "accuracy", "prediction"],
                                 help="List of columns to be used for the classifier")

        # Create output folder
        self.parser.add_argument("--time", type=str, default="20250218_115430_test",
                                 help="Timestamp for output folder")
        self.parser.add_argument("--classifier_model_name", type=str, default="CatBoost",
                                 help="Model name: CatBoost, XGBoost, or MLP")


    def parse(self):
        return self.parser.parse_args()

# Usage example
# args = ArgumentParser().parse()
