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

    # Note: If you did not run all the steps, you can download the unaligned TIIs (heatmaps) from the huggingface or follow the instructions in the notebook to download the data.
    "heatmap_dir": "output/heatmaps/", # Path to unaligned TIIs
    "TII_aligned_dir": "output/TII_aligned/", # Path to aligned TIIs to be saved
}

lifetracer.TII_alignment.align(config)