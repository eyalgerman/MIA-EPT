import os
import subprocess
from datetime import datetime

import features_extraction
from argument_parser import ArgumentParser
from data_manager import *

def run_embeddings_extraction_subprocess(model_folder, model_name, output_dir, index_model, mode, include_account=False):
    """
    Run the features_extraction.py script with the given parameters.

    Args:
        model_name (str): Name of the model (e.g., "tabddpm", "tabsyn").
        output_dir (str): Output directory to save results.
        index_model (int): Index of the model.
        epoch (int): Epoch number.
        mode (str): Mode to run (e.g., "train", "dev", "test").
    """
    code_folder = "/dt/shabtaia/dt-sicpa/daniel/MIDST"
    python_script = os.path.join(code_folder, "features_extraction.py")

    python_path = "/sise/home/samirada/.conda/envs/gpu_env/bin/python3.9"
    if include_account:
        cmd = [
            python_path, python_script,
            "--model_folder", model_folder,
            "--model_name", model_name,
            "--output_dir", output_dir,
            "--index_model", str(index_model),
            "--mode", mode,
            "--include_account"
        ]
    else:
        cmd = [
            python_path, python_script,
            "--model_folder", model_folder,
            "--model_name", model_name,
            "--output_dir", output_dir,
            "--index_model", str(index_model),
            "--mode", mode
        ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_features_extraction(model_folder, model_name, output_dir, index_model, mode, include_account=False):
    """
    Run the features_extraction script directly as a function call.

    Args:
        model_folder (str): Path to the model folder.
        model_name (str): Name of the model (e.g., "tabddpm", "tabsyn").
        output_dir (str): Output directory to save results.
        index_model (int): Index of the model.
        mode (str): Mode to run (e.g., "train", "dev", "test").
        include_account (bool): Whether to include the account column in training.
    """
    args = ArgumentParser().parse()
    args.model_folder = model_folder
    args.model_name = model_name
    args.output_dir = output_dir
    args.index_model = index_model
    args.mode = mode
    args.include_account = include_account

    features_extraction.main(args)


def main(type_test = "blackbox_multi_table"):
    modes = ["train", "dev", "final"]
    base_output_dir = EMBEDDINGS_DATA_DIR

    if type_test == "blackbox_single_table":
        model_names = ["tabddpm", "tabsyn"]
    elif type_test == "blackbox_multi_table":
        model_names = ["clavaddpm"]

    # Nested loops
    for model_name in model_names:
        root = f"{DATA_PATH}{model_name}_black_box/"
        for mode in modes:
            base_dir = os.path.join(root, mode)
            model_folders = [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))]
            for model_folder in sorted(model_folders, key=lambda d: int(d.split('_')[1])):
                index_model = int(model_folder.split('_')[1])
                model_folder = os.path.join(base_dir, model_folder)
                final_output_dir = os.path.join(base_output_dir, f"{model_name}_black_box", mode,
                                                f"{model_name}_{index_model}")
                if os.path.isdir(final_output_dir) and len(os.listdir(final_output_dir)) > 0:
                    print(f"Skipping {model_name}, index_model={index_model}, mode={mode}")
                    continue
                try:
                    run_features_extraction(
                        model_folder=model_folder,
                        model_name=model_name,
                        output_dir=base_output_dir,
                        index_model=index_model,
                        mode=mode,
                        # include_account=include_account
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error running embeddings_extraction for {model_name}, "
                          f"index_model={index_model}, mode={mode}")
                    print(str(e))


if __name__ == "__main__":
    main()
