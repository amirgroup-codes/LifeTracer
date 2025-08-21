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

    "features_path": "/usr/scratch/LifeTracer/features/",
    "TII_aligned_dir": "/usr/scratch/LifeTracer/TII_aligned/",
    "peaks_dir_path": "/usr/scratch/LifeTracer/peaks/",
    "lambda1": [5],
    "lambda2": [100],
    "rt1_threshold": [50],
    "rt2_threshold": [0.8]
}

lifetracer.retention_times_alignment.retention_times_alignment(config)