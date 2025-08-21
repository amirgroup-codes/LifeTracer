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

    "TII_aligned_dir": "/usr/scratch/LifeTracer/TII_aligned/",
    
    "peaks_dir_path": "/usr/scratch/LifeTracer/peaks/",
    "lambda1": 5,
    "lambda2": 100,
    "peak_max_neighbor_distance": 5,
    "strict_noise_filtering": True,

    "enable_noisy_regions": True,
    "noisy_regions": [
        {
            "first_time_start": 0,
            "second_time_start": 0,
            "first_time_end": -1,
            "second_time_end": 1,
            "non_zero_ratio_region_threshold": 1e-3
        },
        {
            "first_time_start": 8700,
            "second_time_start": 1.1,
            "first_time_end": -1,
            "second_time_end": 1.8,
            "non_zero_ratio_region_threshold": 1e-2
        },
        {
            "first_time_start": 8700,
            "second_time_start": 0,
            "first_time_end": -1,
            "second_time_end": -1,
            "non_zero_ratio_region_threshold": 1e-2
        },
        {
            "first_time_start": 8690,
            "second_time_start": 2.2,
            "first_time_end": 8710,
            "second_time_end": 3,
            "non_zero_ratio_region_threshold": 1e-2
        },
        { # 202 EET
            "first_time_start": 5174-50,
            "second_time_start": 0,
            "first_time_end": 5174 + 50,
            "second_time_end": -1,
            "non_zero_ratio_region_threshold": 1e-1
        },

        { # 202 EET
            "first_time_start": 5300-50,
            "second_time_start": 0,
            "first_time_end": 5300 + 50,
            "second_time_end": 1.8,
            "non_zero_ratio_region_threshold": 1e-2
        },

        {
            "first_time_start": 7700-50,
            "second_time_start": 0,
            "first_time_end": 7700 + 50,
            "second_time_end": -1,
            "non_zero_ratio_region_threshold": 1e-1
        },
        {
            "first_time_start": 8700-50,
            "second_time_start": 0,
            "first_time_end": 8700 + 50,
            "second_time_end": -1,
            "non_zero_ratio_region_threshold": 1e-2
        }
    ],
    
    "convolution_filter": {
        "enable": False,
        "lambda3": 1000000,
        "rt1_window_size": 100,
        "rt2_window_size": 0.5,
        "rt1_stride": 20,
        "rt2_stride": 0.5,
        "non_zero_ratio_lambda3_filter": 0.9
    },

    "overall_filter": {
        "enable": True,
        # "lambda": 10,
        "non_zero_ratio_filter": 0.1
    },
}

lifetracer.find_peaks.extract_peaks(config)