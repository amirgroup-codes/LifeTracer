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

    "output_dir_heatmap": "output/heatmaps",

    # Output directory for the aligned heatmaps
    "output_dir_aligned": "output/TII_aligned/",

}

lifetracer.TII_alignment.align(config)