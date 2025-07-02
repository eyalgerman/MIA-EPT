import os
import subprocess
from datetime import datetime
from pathlib import Path
import csv
import itertools
import train_classifier
from argument_parser import ArgumentParser
from data_manager import DATA_PATH, EMBEDDINGS_DATA_DIR, EVALUATION_DIR


def run_script(output_dir, max_index_classifier_train, input_embeddings_extraction, classifier_name, save_model,
               type_test, columns_lst):
    """
    Run the classifier training script.
    """
    args = ArgumentParser().parse()
    args.output_dir = output_dir
    args.max_index_classifier_train = max_index_classifier_train
    args.input_embeddings_extraction = input_embeddings_extraction
    args.classifier_name = classifier_name
    args.save_model = save_model
    args.type_test = type_test
    args.columns_lst = columns_lst

    return train_classifier.main(args)


def main(
    type_test="blackbox_multi_table",
    max_index_classifier_train = 30
):
    """
    Main function to run the classifier training script.

    :param type_test: Type of test (e.g., "blackbox_single_table", "blackbox_multi_table").
    :param max_index_classifier_train: Number of models to train. Determines if the mode is training or testing.
    :return: Results output directory.
    """

    base_output_dir = EVALUATION_DIR
    input_embeddings_extraction = EMBEDDINGS_DATA_DIR
    # Define the types of columns that can be used for the classifier
    columns_type = ["actual", "error", "error_ratio", "accuracy", "prediction"]

    # Iterate over classifier types
    classifier_types = ["XGBoost", "CatBoost", "MLP"]

    # Create a timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the output directory path
    output_dir = Path(base_output_dir) / type_test / timestamp
    mode = "test" if max_index_classifier_train == 30 else "train"
    output_dir = Path(f"{output_dir}_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the script for each classifier and columns combination
    if max_index_classifier_train < 30: # Training mode
        # CSV file to save results
        csv_path = output_dir / "results_summary.csv"
        with open(csv_path, mode="w", newline="") as csv_file:
            if type_test == "blackbox_single_table":
                fieldnames = [
                    "classifier",
                    "columns_lst",
                    "tabddpm_accuracy",
                    "tabddpm_auc_roc",
                    "tabddpm_tpr_fpr_10",
                    "tabddpm_tpr_fpr_1",
                    "tabddpm_tpr_fpr_0.1",
                    "tabsyn_accuracy",
                    "tabsyn_auc_roc",
                    "tabsyn_tpr_fpr_10",
                    "tabsyn_tpr_fpr_1",
                    "tabsyn_tpr_fpr_0.1",
                    "final_tpr_fpr_10",  # Max TPR at FPR=10% across models
                ]
            elif type_test == "blackbox_multi_table":
                fieldnames = [
                    "classifier",
                    "columns_lst",
                    "clavaddpm_accuracy",
                    "clavaddpm_auc_roc",
                    "clavaddpm_tpr_fpr_10",
                    "clavaddpm_tpr_fpr_1",
                    "clavaddpm_tpr_fpr_0.1",
                    "final_tpr_fpr_10",
                ]
            else:
                raise ValueError("Invalid type_test value.")

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for classifier_name in classifier_types:
                # Generate all possible combinations of all lengths
                for r in range(1, len(columns_type) + 1):
                    for columns_lst in itertools.combinations(columns_type, r):
                        try:
                            # Run the script and collect results
                            result = run_script(
                                output_dir=str(output_dir),
                                max_index_classifier_train=max_index_classifier_train,
                                input_embeddings_extraction=input_embeddings_extraction,
                                classifier_name=classifier_name,
                                save_model=False,
                                type_test=type_test,
                                columns_lst=columns_lst
                            )

                            # Write results to the CSV
                            writer.writerow(result)
                            print(f"Results written for {classifier_name}, columns: {columns_lst}")
                        except subprocess.CalledProcessError as e:
                            print(f"Error running script for {classifier_name}")
    else: # Testing mode
        for classifier_name in classifier_types:
            # Generate all possible combinations of all lengths
            for r in range(1, len(columns_type) + 1):
                for columns_lst in itertools.combinations(columns_type, r):
                    try:
                        # Run the script and collect results
                        result = run_script(
                            output_dir=str(output_dir),
                            max_index_classifier_train=max_index_classifier_train,
                            input_embeddings_extraction=input_embeddings_extraction,
                            classifier_name=classifier_name,
                            save_model=True,
                            type_test=type_test,
                            columns_lst=columns_lst
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"Error running script for {classifier_name}")

    print("Evaluation completed.")
    return output_dir

if __name__ == "__main__":
    main()