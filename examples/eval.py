import lifetracer

config = {
    "parallel_processing": True,
    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "features_path": "/usr/scratch/chromalyzer/features/",
    "peaks_dir_path": "/usr/scratch/chromalyzer/peaks/",
    "eval_path":"/usr/scratch/chromalyzer/eval/NB/",


    "model": "lr_l2",
    "lr_l2": {
        "C": [1e-4,1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3,1e+4],
        "lambda1": [5],
        "lambda2": [100],
        "rt1_threshold": [50],
        "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    }

    # Uncomment the following lines to run the other models

    # "model": "lr_l1",
    # "lr_l1": {
    #     "C": [1e-4,1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3,1e+4],
    #     "lambda1": [5],
    #     "lambda2": [100],
    #     "rt1_threshold": [50],
    #     "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # }

    # "model": "svm",
    # "svm": {
    #     "C": [1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3],
    #     "kernel": ["linear","poly","rbf","sigmoid"],
    #     "lambda1": [5],
    #     "lambda2": [100],
    #     "rt1_threshold": [50],
    #     "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # },

    # "model": "xgboost",
    # "xgboost": {
    #     "n_estimators": [20, 50, 100, 200, 500],
    #     "lambda1": [5],
    #     "lambda2": [100],
    #     "rt1_threshold": [10],
    #     "rt2_threshold": [round(i * 0.1,3) for i in range(3,11)]
    # },

    # "model": "rf",
    # "rf": {
    #     "n_estimators": [20, 50, 100, 200, 500],
    #     "lambda1": [5],
    #     "lambda2": [100],
    #     "rt1_threshold": [50],
    #     "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # }

    # "model": 'NaiveBayes',
    # "NaiveBayes": {
    #     "alpha": [0.01, 0.1, 0.5, 1, 5, 10],
    #     "lambda1": [5],
    #     "lambda2": [100],
    #     "rt1_threshold": [50],
    #     "rt2_threshold": [round(i * 0.1,3) for i in range(5,12)], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # 
    # }
}

lifetracer.evaluation.eval(config)