# LifeTracer

# Installation Guide

1. Create a new conda environment:
```bash
conda create -n LifeTracer python=3.10.8
```

2. Activate the environment:
```bash
conda activate LifeTracer
```

3. Install the package:
```bash
# Clone the repository (if not already done)
cd LifeTracer

# Install the package in development mode
pip install -e .
```

4. Verify installation:
```bash
python -c "import lifetracer; print(lifetracer.__version__)"
```

## Required Dependencies

The following dependencies will be automatically installed:
- pandas==2.2.3
- numpy==2.2.4
- loguru==0.7.3
- tqdm==4.67.1
- seaborn==0.13.2
- scipy==1.15.2
- scikit-learn==1.6.1
- plotly==6.0.1
- statsmodels==0.14.4
- xgboost==3.0.0
- svgpathtools==1.6.1

# Analysis Pipeline

For the demonstration we’ve included a small subset of 500 GB data (m/z 128 ± 0.5 for every sample). To reproduce the paper’s full results, simply replace the contents of `demo_data/raw/` with the complete raw chromatography files, which are available on request.

## Step 1: Heatmap Extraction
```bash
python examples/step1_extract_heatmaps.py
```

Quantizes GC×GC-HRTOF-MS scan data based on m/z values to create Total Ion Images (TIIs) for each sample. Configure the following parameters in the script:

| Parameter | Description |
|-----------|-------------|
| `mz_list_path` | Path to CSV file with m/z values |
| `labels_path` | Path to CSV file with sample labels and raw data filenames |
| `m_z_column_name` | Column name for M/Z in raw data CSVs |
| `area_column_name` | Column name for Area/Intensity |
| `first_time_column_name` | Column name for first retention time (RT1) |
| `second_time_column_name` | Column name for second retention time (RT2) |
| `csv_file_name_column` | Column in labels file containing raw CSV filenames |
| `label_column_name` | Column in labels file containing sample labels |
| `output_dir_heatmap` | Output directory for TIIs |
| `extract_heatmaps.raw_csv_path` | Directory containing raw data CSV files |
| `extract_heatmaps.m_z_threshold` | Threshold for m/z quantization (default: 0.5) |
| `extract_heatmaps.parallel_processing` | Enable/disable parallel processing |

## Step 2: TII Alignment
```bash
python examples/step2_TII_alignment.py
```

Aligns Total Ion Images to correct misalignments along the RT2 axis. Configure:

| Parameter | Description |
|-----------|-------------|
| `output_dir_heatmap` | Input directory with TIIs from Step 1 |
| `output_dir_aligned` | Output directory for aligned TIIs |

## Step 3: Peak Detection and Filtering
```bash
python examples/step3_find_peaks.py
```

Identifies peaks from aligned TIIs using intensity thresholding and DBSCAN clustering.

| Parameter | Description |
|-----------|-------------|
| `parallel_processing` | Enable/disable parallel processing |
| `number_of_splits` | Number of splits for parallel processing |
| `output_dir_TII_aligned` | Input directory with aligned TIIs |
| `peaks_dir_path` | Output directory for detected peaks |
| `lambda1` | Intensity threshold multiplier (default: 5) |
| `lambda2` | Local intensity filter threshold (default: 100) |
| `peak_max_neighbor_distance` | Max distance parameter for DBSCAN (default: 5) |
| `strict_noise_filtering` | Enable/disable rigorous noise filtering |
| `enable_noisy_regions` | Enable filtering of specific noisy regions |
| `noisy_regions` | List of rectangular regions in (RT1, RT2) space to filter |
| `convolution_filter.enable` | Enable/disable convolution-based filtering |
| `overall_filter.enable` | Enable/disable filtering based on non-zero pixels |
| `overall_filter.non_zero_ratio_filter` | Threshold for non-zero pixel ratio |

## Step 4: Peak Clustering
```bash
python examples/step4_retention_time_alignments.py
```

Groups peaks from different samples that share m/z values and have similar retention times.

| Parameter | Description |
|-----------|-------------|
| `peaks_dir_path` | Input directory with peak data |
| `features_path` | Output directory for feature data |
| `rt1_threshold` | Maximum RT1 difference for clustering (typically 50s) |
| `rt2_threshold` | Maximum RT2 difference for clustering (typically range 0.5-1.1s) |

## Step 5: Hyperparameter Selection
```bash
python examples/step5_parameters_selection.py
```

Determines optimal hyperparameters using cross-validation techniques.

| Parameter | Description |
|-----------|-------------|
| `features_path` | Input directory with feature data |
| `parameters_selection_path` | Output directory for selection results |
| `C` | List of regularization strength values to test |
| `seed` | Random seed for reproducibility |
| `rt2_threshold` | RT2 clustering thresholds to evaluate |

## Step 6: Classifier Training and Evaluation
```bash
python examples/step6_train_binary_classifier.py
```

Trains the final binary classifier using optimal hyperparameters from Step 5.

| Parameter | Description |
|-----------|-------------|
| `features_path` | Input directory with feature data |
| `results_dir` | Output directory for classification results |
| `C` | Optimal regularization strength (0.1 for this dataset) |
| `lambda1` | Intensity threshold (5) |
| `lambda2` | Local intensity filter threshold (100) |
| `rt1_threshold` | RT1 clustering threshold (50s) |
| `rt2_threshold` | RT2 clustering threshold (0.8s) |
| `seed` | Random seed for reproducibility |