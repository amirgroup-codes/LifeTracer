import lifetracer

config = {
    "mz_list_path": "demo_data/selected_mz.csv",
    "labels_path": "demo_data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "features_path": "output/features/",
    "peaks_dir_path": "output/peaks/",

    # Directory to save the results of the logistic regression model
    "results_dir": "output/lr_l2_model_results/",

    # Logistic Regression with l2 regularization
    "C": 0.1,
    "seed": 42,
    "lambda1": 5,
    "lambda2": 100,
    "rt1_threshold": 50,
    "rt2_threshold": 0.8,
}

lifetracer.binary_classifier.binary_classifier(config)