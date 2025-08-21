import lifetracer

config = {
    # Path to the CSV file containing the labels
    "labels_path": "data/labels.csv",
    
    # Name of the column in the labels.csv that contains the labels
    "label_column_name": "label",

    # Name of the column in the labels.csv that contains the csv file names.
    "csv_file_name_column": "csv_file_name",
    
    # Note: If you did not run all the steps, you can download the heatmaps from the huggingface.
    # You can download the heatmaps from the huggingface. Follow the instructions in the notebook to download the data.
    """
    Download aligned TIIs:
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-aa
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ab
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ac
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ad
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ae
    https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-af
    """
    
    # Directory where generated heatmaps are stored
    "heatmap_dir": "output/TII_aligned/",
    
    # Directory where generated plots will be saved
    "plot_dir": "output/plots/",
    
    # Boolean flag indicating whether all samples should be processed
    "all_samples": True,
    
    # Name of the sample to be analyzed if all_samples is False
    "sample_name": "230823_01_Atacama_Soil_300uLDCM_100oC24h-001.csv",

    # Change the m/z value to the one you want to plot.
    "m_z": "469"
}

lifetracer.plot_heatmap.plot_heatmap(config)
