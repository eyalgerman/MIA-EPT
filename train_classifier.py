# from fastai.tabular.all import *
# from torch.utils.data import Dataset
# import argparse
# from datetime import datetime
from argument_parser import ArgumentParser
from data import ChallengeDataset
import pandas as pd
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from data_manager import *


def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) classifier.
    """
    def __init__(self, input_size=100, hidden_size=64, output_size=1):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(X_train, y_train, X_test, y_test, epochs, device, eval):
    """
    Train an MLP classifier and evaluate it on the test set.
    """
    input_size = X_train.shape[1]
    model = MLPClassifier(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    if eval:
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    y_pred, y_proba = None, None
    if eval:
        model.eval()
        with torch.no_grad():
            # Get probabilities
            y_proba = model(X_test).squeeze().cpu().numpy()
            # Convert probabilities to binary predictions
            y_pred = (y_proba > 0.5).astype(float)

    return model, y_pred, y_proba


def save_results(output_dir, model_name, classifier_name, accuracy, auc_roc, tpr_at_fpr, fpr_values, model, args,
                 save_model=True):
    """
    Saves model training results, arguments, and optionally the trained model.
    """
    columns_str = str(args.columns_lst).replace("[", "").replace("]", "").replace("'", "")
    result_dir = Path(output_dir) / f"{classifier_name}_{columns_str}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save model if requested
    if save_model:
        if classifier_name == "MLP":
            model_path = result_dir / f"{model_name}_model.pt"
            torch.save(model.state_dict(), model_path)
        else:
            model_path = result_dir / f"{model_name}_model.json"
            model.save_model(model_path)
        # print(f"Model saved to: {model_path}")
    else:
        print("Model saving skipped.")

    # Save arguments
    args_path = result_dir / "args.txt"
    with open(args_path, "w") as f:
        f.write(str(args))
    # print(f"Arguments saved to: {args_path}")

    if accuracy:
        # Save results
        results_path = result_dir / f"{model_name}_results.txt"
        with open(results_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC-ROC: {auc_roc:.4f}\n")
            for fpr, tpr in zip(fpr_values, tpr_at_fpr.values()):
                f.write(f"TPR at FPR={fpr:.1%}: {tpr:.4f}\n")

        print(f"Results saved to: {results_path}")
        print(f"----- Results for {model_name} -----")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        for fpr, tpr in zip(fpr_values, tpr_at_fpr.values()):
            print(f"TPR at FPR={fpr:.1%}: {tpr:.4f}")


def concatenate_data(features, args):
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
        'prediction': lambda x: x.endswith('prediction'),
    }

    if len(args.columns_lst) == 1 and "," in args.columns_lst[0]:
        args.columns_lst = args.columns_lst[0].split(",")
        args.columns_lst = [col.strip() for col in args.columns_lst]

    # Iterate over each dataframe in features list
    for df in features:
        # Filter columns for each type in args.columns_lst
        selected_columns = [col for col_type in args.columns_lst for col in df.columns if suffix_mapping[col_type](col)]
        # Check if there are any selected columns and concatenate them
        if selected_columns:
            concat_list.append(df[selected_columns].values)

    # Vertically stack all selected columns from all dataframes
    X_train = np.vstack(concat_list)
    return X_train


def main(args):
    """
    Main function to train and evaluate classifiers on embeddings extracted from synthetic data.

    :param args : argument_parser.ArgumentParser
        Command-line arguments containing parameters such as test type, classifier type, dataset paths, and other settings.
    Returns:
        dict: Dictionary containing evaluation metrics such as accuracy, AUC-ROC, and TPR at specific FPR thresholds.
    """

    # Set seed for reproducibility
    set_seed(args.seed)
    max_model_number = 30
    fpr_thresholds = [0.1, 0.01, 0.001]  # 10%, 1%, 0.1%

    # Determine models based on the test type
    if args.type_test == 'blackbox_single_table':
        model_names = ["tabddpm", "tabsyn"]
    elif args.type_test == 'blackbox_multi_table':
        model_names = ['clavaddpm']
    else:
        raise ValueError(f"Unknown test type: {args.type_test}")

    # Initialize results dictionary
    results = {
        "columns_lst": " ".join(args.columns_lst),
        "classifier": args.classifier_name,
    }

    # Iterate through each model for training and evaluation
    for model_name in model_names:
        X_train, y_train = [], []
        X_test, y_test = [], []

        # Load embeddings and labels
        for i in range(1, args.max_index_classifier_train + 1):
            # Set paths
            model_path_outputs = f"{DATA_PATH}{model_name}_black_box/"
            base_dir = os.path.join(model_path_outputs, f"train/{model_name}_{str(i)}")
            # Load datasets
            challenge_ds = ChallengeDataset(base_dir)
            labels = challenge_ds.challenge_labels['is_train'].values

            # Load the embeddings from torch file
            features = pd.read_csv(f"{args.input_embeddings_extraction}/{model_name}_black_box/train/{model_name}_{i}/features.csv")

            X_train.append(features)
            y_train.append(labels)

        X_train = concatenate_data(X_train, args)
        y_train = np.hstack(y_train)

        accuracy, auc_roc, tpr_at_fpr, test_results = None, None, None, None

        if args.max_index_classifier_train < max_model_number: # Training mode
            # Load embeddings and labels
            for i in range(args.max_index_classifier_train + 1, max_model_number + 1):
                # Set paths
                model_path_outputs = f"{DATA_PATH}{model_name}_black_box/"
                base_dir = os.path.join(model_path_outputs, f"train/{model_name}_{str(i)}")
                # Load datasets
                challenge_ds = ChallengeDataset(base_dir)
                labels = challenge_ds.challenge_labels['is_train'].values

                # Load the embeddings from torch file
                features = pd.read_csv(f"{args.input_embeddings_extraction}/{model_name}_black_box/train/{model_name}_{i}/features.csv")

                X_test.append(features)
                y_test.append(labels)

            # Concatenate all data
            X_test = concatenate_data(X_test, args)
            y_test = np.hstack(y_test)

            # Train the classifier
            if args.classifier_name == "XGBoost":
                model = XGBClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            elif args.classifier_name == "CatBoost":
                model = CatBoostClassifier(verbose=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            elif args.classifier_name == "MLP":
                model, y_pred, y_proba = train_mlp(X_train, y_train, X_test, y_test, args.epoch, torch.device("cuda" if torch.cuda.is_available() else "cpu"), eval=True)
            else:
                raise ValueError(f"Unknown classifier: {args.classifier_name}")

            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_roc = auc(fpr, tpr)

            # Compute TPR at specific FPR thresholds
            tpr_at_fpr = {}
            for threshold in fpr_thresholds:
                tpr_at_fpr[threshold] = max(tpr[fpr < threshold])

            # Save per-record predictions for later ROC AUC curve plotting
            if accuracy is not None:
                prediction_df = pd.DataFrame({
                    "y_true": y_test,
                    "y_proba": y_proba,
                    "y_pred": y_pred
                })
                columns_str = str(args.columns_lst).replace("[", "").replace("]", "").replace("'", "")
                result_dir = Path(args.output_dir) / f"{args.classifier_name}_{columns_str}"
                result_dir.mkdir(parents=True, exist_ok=True)
                pred_path = result_dir / f"{model_name}_prediction.csv"
                prediction_df.to_csv(pred_path, index=False)
                print(f"Per-record predictions saved to: {pred_path}")

        else: # Testing mode
            # Train the classifier without evaluation
            if args.classifier_name == "XGBoost":
                model = XGBClassifier()
                model.fit(X_train, y_train)
            elif args.classifier_name == "CatBoost":
                model = CatBoostClassifier(verbose=0)
                model.fit(X_train, y_train)
            elif args.classifier_name == "MLP":
                model, y_pred, y_proba = train_mlp(X_train, y_train, X_test, y_test, args.epoch,
                                                   torch.device("cuda" if torch.cuda.is_available() else "cpu"), eval=False)
            else:
                raise ValueError(f"Unknown classifier: {args.classifier_name}")

        # Store results
        if accuracy:
            results[f"{model_name}_accuracy"] = accuracy
            results[f"{model_name}_auc_roc"] = auc_roc
            for threshold in fpr_thresholds:
                if int(threshold * 100) <= 0:
                    results[f"{model_name}_tpr_fpr_{float(threshold * 100)}"] = tpr_at_fpr.get(threshold, 0.0)
                else:
                    results[f"{model_name}_tpr_fpr_{int(threshold * 100)}"] = tpr_at_fpr.get(threshold, 0.0)

        # Save results
        save_results(
            args.output_dir,
            model_name,
            args.classifier_name,
            accuracy,
            auc_roc,
            tpr_at_fpr,
            fpr_thresholds,
            model,
            args,
            save_model=args.save_model
        )

    # Compute final TPR at 10% FPR across models
    if "tabddpm_tpr_fpr_10" in results and "tabsyn_tpr_fpr_10" in results:
        results["final_tpr_fpr_10"] = max(results["tabddpm_tpr_fpr_10"], results["tabsyn_tpr_fpr_10"])
    elif "clavaddpm_tpr_fpr_10" in results:
        results["final_tpr_fpr_10"] = results["clavaddpm_tpr_fpr_10"]
    return results


if __name__ == '__main__':
    args = ArgumentParser().parse()
    main(args)
