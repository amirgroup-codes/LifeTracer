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

    "heatmap_dir": "/usr/scratch/LifeTracer/heatmaps",
    "TII_aligned_dir": "/usr/scratch/LifeTracer/TII_aligned/",
}

lifetracer.TII_alignment.align(config)