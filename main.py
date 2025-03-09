import pandas as pd

import create_predictions_folder
import run_train_classifier
import run_features_extraction


def main(test_type="blackbox_multi_table"):

    print("Starting the pipeline...")
    # Create feature embeddings
    run_features_extraction.main(test_type)

    # Run the classifier for train on 25 models and test on 5 models
    output_dir_train = run_train_classifier.main(type_test=test_type, max_index_classifier_train=25)

    # Run the classifier for train on 30 models without testing
    output_dir_test = run_train_classifier.main(type_test=test_type, max_index_classifier_train=30)

    train_result_file = output_dir_train / "results_summary.csv"
    train_result_file = pd.read_csv(train_result_file)

    print("Output train directory:", output_dir_train)
    print("Output test directory:", output_dir_test)
    test_time = output_dir_test.name
    print("Test timestamp:", test_time)
    print("Output of the training results:")
    train_result_file.sort_values(by=["final_tpr_fpr_10"], ascending=False, inplace=True)

    if test_type == "blackbox_multi_table":
        columns = [
                        "classifier",
                        "columns_lst",
                        "clavaddpm_accuracy",
                        "clavaddpm_auc_roc",
                        "final_tpr_fpr_10",
                    ]
    else:
        columns = [
                        "classifier",
                        "columns_lst",
                        "tabddpm_accuracy",
                        "tabddpm_auc_roc",
                        "tabsyn_accuracy",
                        "tabsyn_auc_roc",
                        "final_tpr_fpr_10",  # Max TPR at FPR=10% across models
                    ]
    print(train_result_file[columns].head(5))

    # Create a submission file with the best model
    row = train_result_file.iloc[0]
    features_lst = row["columns_lst"].split(" ")
    create_predictions_folder.main(time=test_time, type_test=test_type, model_name=row["classifier"], features_lst=features_lst)



if __name__ == "__main__":
    main()

