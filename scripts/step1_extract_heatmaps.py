import lifetracer

"""
Parameters
-----------
mz_list_path : str
    Path to m/z list csv file.
labels_path : str
    Path to labels csv file.
m_z_column_name : str
    The column name corresponding to M/Z in the raw csv file.
area_column_name : str
    The column name corresponding to Area in the raw csv file.
first_time_column_name : str
    The column name corresponding to the first time column in the raw csv file.
second_time_column_name : str
    The column name corresponding to the second time column in the raw csv file.
csv_file_name_column : str
    The column name corresponding to the csv file name in the labels csv file.
label_column_name : str
    The column name corresponding to the label in the labels csv file.
heatmap_dir : str
    Directory path to save generated heatmaps.
extract_heatmaps : dict
    Dictionary containing parameters for heatmap extraction:
        raw_csv_path : str
            Path to directory containing raw csv files.
        m_z_threshold : float
            Threshold value for m/z filtering.
        parallel_processing : bool
            Flag to indicate whether to use parallel processing.
"""

config = {
    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "heatmap_dir": "output/heatmaps/",
    "extract_heatmaps": {
        "raw_csv_path": "downloads/raw/",
        "m_z_threshold": 0.5,
        "parallel_processing": True
    },
}

lifetracer.extract_heatmap.heatmap_extraction(config)