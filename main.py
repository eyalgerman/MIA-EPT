import pandas as pd

import run_train_classifier
import run_features_extraction


def main(test_type="blackbox_multi_table"):

    print("Starting the pipeline...")
    # Create feature embeddings
    # run_features_extraction.main(test_type)

    # Run the classifier for train on 25 models and test on 5 models
    output_dir_train = run_train_classifier.main(type_test=test_type, max_index_classifier_train=25)

    # Run the classifier for train on 30 models without testing
    output_dir_test = run_train_classifier.main(type_test=test_type, max_index_classifier_train=30)

    train_result_file = output_dir_train / "results_summary.csv"
    train_result_file = pd.read_csv(train_result_file)

    print("Output train directory:", output_dir_train)
    print("Output test directory:", output_dir_test)
    print("Output of the training results:")
    train_result_file.sort_values(by=["final_tpr_fpr_10"], ascending=False, inplace=True)
    print(train_result_file.head(5))


if __name__ == "__main__":
    main()

