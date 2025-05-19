import lifetracer

config = {
    "parallel_processing": True,
    "number_of_splits": 100,

    "mz_list_path": "demo_data/selected_mz.csv",
    "labels_path": "demo_data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "features_path": "output/features/",
    "output_dir_TII_aligned": "output/TII_aligned/",
    "peaks_dir_path": "output/peaks/",
    "lambda1": [5],
    "lambda2": [100],
    "rt1_threshold": [50],
    "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # "rt2_threshold": [0.9], # 5*0.008 = 0.04 ; 150*0.008 = 1.2
}

lifetracer.retention_times_alignment.retention_times_alignment(config)