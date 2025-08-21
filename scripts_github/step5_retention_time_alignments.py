import lifetracer

config = {
    "parallel_processing": True,
    "number_of_splits": 100,

    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    # Note: If you did not run all the prvious steps, you can download the aligned TIIs from the huggingface or follow the instructions in the notebook to download the data.
    "TII_aligned_dir": "output/TII_aligned/", # Path to aligned TIIs
    "peaks_dir_path": "output/peaks/", # Path to peaks
    "features_path": "output/features/", # Path to features to be saved

    # Note: The best lambda1 and lambda2 was obtained from step 3.
    "lambda1": [5],
    "lambda2": [100],
    "rt1_threshold": [50],
    "rt2_threshold": [0.8]
}

lifetracer.retention_times_alignment.retention_times_alignment(config)