import csv
import zipfile
import numpy as np

from argument_parser import ArgumentParser
from metrics import get_tpr_at_fpr
import torch
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from train_classifier import MLPClassifier
import os
from data_manager import *
from datetime import datetime
import pandas as pd


def get_attack_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check the file extension to determine the model type
    if model_path.endswith(".pt"):  # PyTorch model (MLP)
        # Load the state dictionary from the model file
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # Find the first layer's weight tensor
        first_layer_weights = next(iter(state_dict.values()))
        # The input size is typically the size of the second dimension of the weight matrix of the first layer
        input_size = first_layer_weights.shape[1]

        model = MLPClassifier(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()  # Ensure the model is in evaluation mode
        print(f"Loaded MLP model from {model_path}")
    elif model_path.endswith(".json"):  # XGBoost or CatBoost
        # Determine if it's XGBoost or CatBoost based on the filename
        if "xgboost" in model_path.lower():
            model = XGBClassifier()
            model.load_model(model_path)
            print(f"Loaded XGBoost model from {model_path}")
        elif "catboost" in model_path.lower():
            model = CatBoostClassifier()
            model.load_model(model_path)
            print(f"Loaded CatBoost model from {model_path}")
        else:
            raise ValueError("Unknown model type in JSON file.")
    else:
        raise ValueError(f"Unsupported model file extension: {model_path}")

    return model


def concatenate_data(df, columns_lst):
    """
    Function to filter and concatenate data based on column suffixes
    """

    # Create an empty list to store the arrays for concatenation
    concat_list = []

    # Define suffix mapping based on args.columns_lst
    suffix_mapping = {
        'actual': lambda x: not (x.endswith('error') or x.endswith('error_ratio') or x.endswith('accuracy') or x.endswith('prediction')),
        'error': lambda x: x.endswith('error'),
        'error_ratio': lambda x: x.endswith('error_ratio'),
        'accuracy': lambda x: x.endswith('accuracy'),
        'prediction': lambda x: x.endswith('prediction')
    }

    columns_lst = [col.strip() for col in columns_lst]

    # Filter columns for each type in args.columns_lst
    selected_columns = [col for col_type in columns_lst for col in df.columns if suffix_mapping[col_type](col)]
    # Check if there are any selected columns and concatenate them
    if selected_columns:
        return df[selected_columns].values



def main(
    time = "20250218_115430_test",
    type_test = "blackbox_multi_table",
    model_name = "CatBoost", #or XGBoost or MLP
    features_lst = ["actual", "error", "error_ratio", "accuracy", "prediction"]  # List of features to be used for the classifier
):
    features = ", ".join(features_lst)

    embeddings_extraction_path = EMBEDDINGS_DATA_DIR
    models_path = f"{EVALUATION_DIR}{type_test}/{time}/{model_name}_({features})"
    prediction_folder = f"{DATA_PATH}predictions_ML_new/"
    timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # add end file
    if model_name == "MLP":
        end_file = '.pt'
    elif model_name == "XGBoost" or model_name == "CatBoost":
        end_file = '.json'
    else:
        raise ValueError("Unknown model name.")


    if type_test == "blackbox_single_table":
        base_tabddpm_train_path = f"{models_path}/tabddpm_model{end_file}"
        base_tabsyn_train_path = f"{models_path}/tabsyn_model{end_file}"
        tabddpm_attack_model = get_attack_model(base_tabddpm_train_path)
        tabsyn_attack_model = get_attack_model(base_tabsyn_train_path)
        zip_name = "black_box_single_table_submission.zip"
        DATA_DIR = [TABDDPM_DATA_DIR, TABSYN_DATA_DIR]
        attack_models = [tabddpm_attack_model, tabsyn_attack_model]
    elif type_test == "blackbox_multi_table":
        base_clavaddpm_train_path = f"{models_path}/clavaddpm_model{end_file}"
        clavaddpm_attack_model = get_attack_model(base_clavaddpm_train_path)
        zip_name = "black_box_multi_table_submission.zip"
        DATA_DIR = [CLAVADDPM_MODEL_DIR]
        attack_models = [clavaddpm_attack_model]
    else:
        raise ValueError("Unknown test type.")

    phases = ["train", "dev", "final"]
    for base_dir, attack_model in zip(DATA_DIR, attack_models):
        for phase in phases:
            root = os.path.join(base_dir, phase)
            for model_folder in sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])):
                path = os.path.join(root, model_folder)

                # Load challenge points
                embeddings_path = embeddings_extraction_path + f"/{base_dir.split('/')[-2].split('_')[0]}_black_box/{phase}/{model_folder}/features.csv"
                challenge_points = pd.read_csv(embeddings_path)
                challenge_points = concatenate_data(challenge_points, features.split(","))

                # Convert challenge points to a format compatible with the model
                if isinstance(attack_model, torch.nn.Module):  # PyTorch model
                    challenge_points_tensor = torch.tensor(challenge_points, dtype=torch.float32)
                    attack_model.eval()  # Ensure model is in evaluation mode
                    with torch.no_grad():
                        predictions = attack_model(challenge_points_tensor).squeeze().cpu().numpy()
                elif isinstance(attack_model, XGBClassifier):  # XGBoost or CatBoost model
                    predictions = attack_model.predict_proba(challenge_points)[:, 1]  # Probability of positive class
                elif isinstance(attack_model, CatBoostClassifier):  # MLP model
                    # Convert to numpy array if it's a torch.Tensor
                    if isinstance(challenge_points, torch.Tensor):
                        challenge_points = challenge_points.cpu().numpy()
                    predictions = attack_model.predict_proba(challenge_points)[:, 1]  # Probability of positive class
                else:
                    raise ValueError(f"Unsupported model type: {type(attack_model)}")

                # Validate predictions
                predictions_tensor = torch.tensor(predictions)
                assert torch.all(
                    (0 <= predictions_tensor) & (predictions_tensor <= 1)), "Predictions must be in range [0, 1]"

                # Write predictions to a CSV file
                with open(os.path.join(path, "prediction.csv"), mode="w", newline="") as file:
                    writer = csv.writer(file)
                    for value in predictions_tensor.numpy():
                        writer.writerow([value])

    tpr_at_fpr_list = []
    for base_dir in DATA_DIR:
        predictions = []
        solutions = []
        root = os.path.join(base_dir, "train")
        for model_folder in sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])):
            path = os.path.join(root, model_folder)
            predictions.append(np.loadtxt(os.path.join(path, "prediction.csv")))
            solutions.append(np.loadtxt(os.path.join(path, "challenge_label.csv"), skiprows=1))

        predictions = np.concatenate(predictions)
        solutions = np.concatenate(solutions)

        tpr_at_fpr = get_tpr_at_fpr(solutions, predictions)
        tpr_at_fpr_list.append(tpr_at_fpr)

        print(f"{base_dir.split('_')[0]} Train Attack TPR at FPR==10%: {tpr_at_fpr}")

    final_tpr_at_fpr = max(tpr_at_fpr_list)
    print(f"Final Train Attack TPR at FPR==10%: {final_tpr_at_fpr}")

    final_path_predictions = os.path.join(prediction_folder, type_test, timestemp + f'_{model_name}_{features}')
    os.makedirs(final_path_predictions, exist_ok=True)

    with zipfile.ZipFile(f"{final_path_predictions}/{zip_name}", 'w') as zipf:
        for phase in ["dev", "final"]:
            for base_dir in DATA_DIR:
                root = os.path.join(base_dir, phase)
                for model_folder in sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])):
                    path = os.path.join(root, model_folder)
                    if not os.path.isdir(path): continue

                    file = os.path.join(path, "prediction.csv")
                    if os.path.exists(file):
                        # Use `arcname` to remove the base directory and phase directory from the zip path
                        arcname = os.path.relpath(file, os.path.dirname(DATA_PATH))
                        zipf.write(file, arcname=arcname)
                    else:
                        raise FileNotFoundError(f"`prediction.csv` not found in {path}.")
    print("Submission file created successfully ans saved in:", final_path_predictions)


if __name__ == "__main__":
    args = ArgumentParser().parse()
    main(type_test=args.type_test, model_name=args.classifier_model_name, features_lst=args.columns_lst, time=args.time)