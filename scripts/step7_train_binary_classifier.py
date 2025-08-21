import lifetracer

config = {
    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    # Note: If you did not run all the prvious steps, you can download the features and peaks from the huggingface.
    # You can download the features and peaks from the huggingface. Follow the instructions in the notebook to download the data.
    # Download features: https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/features.zip
    # Download peaks: https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/peaks.zip

    "features_path": "output/features/", # Path to features
    "peaks_dir_path": "output/peaks/", # Path to peaks
    "results_dir": "output/lr_l2_results/", # Path to results to be saved

    # Logistic Regression with l2 regularization
    "C": 0.1,

    # Set the seed for reproducibility
    "seed": 42,

    # Note: The best lambda1, lambda2, rt1_threshold, rt2_threshold was obtained from step 3.
    "lambda1": 5,
    "lambda2": 100,
    "rt1_threshold": 50,
    "rt2_threshold": 0.8,
}

lifetracer.binary_classifier.binary_classifier(config)