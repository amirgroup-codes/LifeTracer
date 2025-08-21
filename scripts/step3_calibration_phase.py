import lifetracer

config = {
    "calibration_dataset_path": "data/calibration_dataset.csv",
    "mz_list_path": "data/calibration_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "lambda1s": [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "lambda2s": [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
    "rt1_tol": 50,
    "rt2_tol": 1,
    "accuracy_threshold": 0.9,

    # Note: If you did not run all the prvious steps, you can download the aligned TIIs from the huggingface or follow the instructions in the notebook to download the data.
    "TII_aligned_dir": "output/TII_aligned/", # Path to aligned TIIs
    "calibration_phase_output_dir": "output/calibration_phase/", # Path to calibration phase output
    "best_config_save_path": "output/best_config/",
}

lifetracer.calibration_phase.calibration_phase(config)