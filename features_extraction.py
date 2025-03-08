import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from argument_parser import ArgumentParser
from data import *
from data_manager import *
import argparse
import torch
import random


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


# def create_output_folder(base_path):
#     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     output_path = os.path.join(base_path, timestamp)
#     os.makedirs(output_path, exist_ok=True)
#     return output_path


def preprocess_and_train(train_points, test_points, target_col, include_account):
    # Combine data for consistent preprocessing
    train_points['is_train'] = 1
    test_points['is_train'] = 0
    # test_points['label'] = test_labels
    combined = pd.concat([train_points, test_points], axis=0)

    # Drop or retain the 'account' column
    if not include_account:
        combined = combined.drop(['account'], axis=1)

    # Identify categorical and numeric columns
    numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns.difference(
        [target_col, 'is_train'])
    categorical_cols = combined.select_dtypes(include=['object']).columns

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Determine if the task is regression or classification
    task_type = "classification" if combined[target_col].nunique() <= 20 else "regression"

    # Define the appropriate model
    model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Split combined back to train/test
    train_data = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
    test_data = combined[combined['is_train'] == 0].drop(['is_train'], axis=1)

    X_train = train_data.drop([target_col], axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Generate predictions
    predictions = model_pipeline.predict(X_test)

    return predictions, y_test, task_type


def main(args):
    # Parse arguments
    index_model = args.index_model
    model_name = args.model_name
    output_dir = args.output_dir
    mode = args.mode
    model_folder = args.model_folder

    challenge_dataset = ChallengeDataset(model_folder)
    synthetic_dataset = SyntheticDataset(model_folder)

    train_points = synthetic_dataset.trans_synthetic_points
    test_points = challenge_dataset.challenge_points
    # test_labels = challenge_dataset.challenge_labels

    # Define target columns (excluding irrelevant ones)
    target_columns = [col for col in train_points.columns if col not in ['label', 'account', 'is_train']]
    features = []
    columns = []

    for target_col in target_columns:
        print(f"Target Column: {target_col}")
        predictions, y_test, task_type = preprocess_and_train(train_points, test_points, target_col, args.include_account)

        features.append(y_test)
        columns.append(target_col)

        if task_type == "classification":
            # Calculate accuracy
            accuracy = (predictions == y_test)
            accuracy = accuracy.astype(int)
            features.append(accuracy)
            columns.append(f"{target_col}_accuracy")
        else:
            # Calculate errors
            errors = np.abs(predictions - y_test)
            # Calculate the ratio of the error
            error_ratio = errors / y_test

            # i want to save the actual, the error and the ratio error
            features.append(errors)
            features.append(error_ratio)
            columns.append(f"{target_col}_error")
            columns.append(f"{target_col}_error_ratio")

    # Create a DataFrame with the results
    results_df = pd.DataFrame(features).T
    results_df.columns = columns

    # Prepare output directory
    final_output_dir = os.path.join(output_dir, f"{model_name}_black_box", mode, f"{model_name}_{index_model}")
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    # Save the results
    results_csv_path = os.path.join(final_output_dir, "features.csv")
    results_df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    args = ArgumentParser().parse()
    set_seed(args.seed)  # Ensure reproducibility
    main(args)
