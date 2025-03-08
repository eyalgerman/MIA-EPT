import os
import subprocess
from datetime import datetime
from pathlib import Path
import csv
import itertools
import train_classifier
from argument_parser import ArgumentParser
from data_manager import DATA_PATH, EMBEDDINGS_DATA_DIR, EVALUATION_DATA_DIR, EVALUATION_DIR


def run_script_subprocess(script_path, output_dir, max_index_classifier_train, input_embeddings_extraction, classifier_name, save_model, type_test, columns_lst):
    """
    Run the classifier training script with the specified arguments and capture its output.

    Args:
        script_path (str): Path to the script to be executed.
        output_dir (str): Base output directory.
        max_index_classifier_train (int): Maximum index for training.
        input_embeddings_extraction (str): Path to input embeddings.
        model_name (str): Name of the embedding model.
        classifier_name (str): Classifier type (e.g., XGBoost, CatBoost, MLP).
        epoch (int): Number of training epochs.
        save_model (bool): Whether to save the model.

    Returns:
        dict: A dictionary containing results from the script execution.
    """
    python_path = "/sise/home/samirada/.conda/envs/gpu_env/bin/python3.9"
    cmd = [
        python_path, script_path,
        "--output_dir", output_dir,
        "--max_index_classifier_train", str(max_index_classifier_train),
        "--input_embeddings_extraction", input_embeddings_extraction,
        "--classifier_name", classifier_name,
        "--type_test", type_test,
        "--columns_lst", str(list(columns_lst)).replace("[", "").replace("]", "").replace("'", "")
    ]
    if save_model:
        cmd.append("--save_model")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)

    # Parse the output to extract metrics
    output_lines = result.stdout.splitlines()

    # Initialize a dictionary to hold results
    results = {
        "columns_lst": str(list(columns_lst)).replace(",", " ").replace("[", "").replace("]", "").replace("'", ""),
        "classifier": classifier_name,
    }

    # Variable to track the current model
    current_model = None

    for line in output_lines:
        line = line.strip()
        if "----- Results for" in line:
            # Identify the current model being processed
            current_model = line.split("----- Results for")[1].strip().rstrip(" -----")
        elif "Accuracy:" in line and current_model:
            results[f"{current_model}_accuracy"] = float(line.split(":")[1].strip())
        elif "AUC-ROC:" in line and current_model:
            results[f"{current_model}_auc_roc"] = float(line.split(":")[1].strip())
        elif "TPR at FPR=10.0%" in line and current_model:
            results[f"{current_model}_tpr_fpr_10"] = float(line.split(":")[1].strip())
        elif "TPR at FPR=1.0%" in line and current_model:
            results[f"{current_model}_tpr_fpr_1"] = float(line.split(":")[1].strip())
        elif "TPR at FPR=0.1%" in line and current_model:
            results[f"{current_model}_tpr_fpr_0.1"] = float(line.split(":")[1].strip())
    if "tabddpm_tpr_fpr_10" in results and "tabsyn_tpr_fpr_10" in results:
        results["final_tpr_fpr_10"] = (results["tabddpm_tpr_fpr_10"] + results["tabsyn_tpr_fpr_10"]) / 2

    return results


def run_script(output_dir, max_index_classifier_train, input_embeddings_extraction, classifier_name, save_model,
               type_test, columns_lst):
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

    base_output_dir = EVALUATION_DIR
    input_embeddings_extraction = EMBEDDINGS_DATA_DIR
    columns_type = ["actual", "error", "error_ratio", "accuracy"]

    # Iterate over classifier types and epochs
    classifier_types = ["XGBoost", "CatBoost", "MLP"]

    # Create a timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the output directory path
    output_dir = Path(base_output_dir) / type_test / timestamp
    mode = "test" if max_index_classifier_train == 30 else "train"
    output_dir = Path(f"{output_dir}_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_index_classifier_train < 30:
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
                    "final_tpr_fpr_10",  # Average TPR at FPR=10% across models
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
                                # script_path=script_path,
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
    else:
        for classifier_name in classifier_types:
            # Generate all possible combinations of all lengths
            for r in range(1, len(columns_type) + 1):
                for columns_lst in itertools.combinations(columns_type, r):
                    try:
                        # Run the script and collect results
                        result = run_script(
                            # script_path=script_path,
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