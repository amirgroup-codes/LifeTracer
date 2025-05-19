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
    "parameters_selection_path":"output/parameters_selection/",

    # Logistic Regression with L2 regularization
    "C": [1e-4,1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3,1e+4],
    "seed": 42,
    "lambda1": [5],
    "lambda2": [100],
    "rt1_threshold": [50],
    "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)],
}

lifetracer.parameters_selection.parameters_selection(config)