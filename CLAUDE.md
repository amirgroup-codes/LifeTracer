# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LifeTracer (lifetracer) is a Python package for 2D gas chromatography analysis and classification. The system processes raw chromatographic data through multiple pipeline stages to identify molecular signatures and classify samples.

## Installation & Setup

```bash
# Install the package in development mode
pip install -e .

# Or install with setup.py
python setup.py install

# Clean build artifacts
python setup.py clean
```

## Running the Application

The main entry point is through the lifetracer command-line interface:

```bash
# Run specific modules
lifetracer --module extract_heatmap --config_path config.json
lifetracer --module find_peaks --config_path config.json
lifetracer --module plot_heatmap --config_path config.json

# Alternative: Run via python module
python -m lifetracer --module extract_heatmap --config_path config.json
```

## Processing Pipeline

The lifetracer workflow follows these sequential steps:

1. **extract_heatmap** - Convert raw CSV chromatographic data into heatmap representations
2. **TII_alignment** - Perform Total Ion Intensity alignment across samples
3. **find_peaks** - Detect and cluster chromatographic peaks using DBSCAN
4. **retention_times_alignment** - Align retention times across samples
5. **parameters_selection** - Select optimal parameters for classification
6. **binary_classifier** - Train logistic regression models for sample classification

### Example Usage

See the `examples/` directory for step-by-step pipeline execution:
- `step1_extract_heatmaps.py` - Extract heatmaps from raw data
- `step2_TII_alignment.py` - Perform TII alignment
- `step3_find_peaks.py` - Find chromatographic peaks
- `step4_retention_time_alignments.py` - Align retention times
- `step5_parameters_selection.py` - Parameter selection
- `step6_train_binary_classifier.py` - Train classification model

## Configuration

The system uses JSON configuration files (see `config.json`) with these key sections:

- **Data paths**: `mz_list_path`, `labels_path`, column names
- **extract_heatmaps**: Raw CSV processing parameters
- **find_peaks**: Peak detection with DBSCAN clustering, noise filtering
- **plot**: Visualization output directories

Important configuration parameters:
- `m_z_threshold`: Mass-to-charge ratio filtering threshold
- `lambda1`, `lambda2`: Peak detection sensitivity parameters
- `area_min_threshold`: Minimum peak area threshold
- `parallel_processing`: Enable/disable parallel processing

## Core Architecture

### Main Modules (`lifetracer/src/`)
- **extract_heatmap.py** - Converts raw chromatographic data to 2D heatmaps
- **find_peaks.py** - DBSCAN-based peak clustering with noise filtering
- **binary_classifier.py** - Logistic regression for sample classification
- **TII_alignment.py** - Total Ion Intensity alignment algorithms
- **retention_times_alignment.py** - Retention time normalization
- **evaluation.py** - Model evaluation and performance metrics

### Utilities (`lifetracer/src/utils/`)
- **heatmap_utils.py** - Heatmap generation and I/O operations
- **plot_utils.py** - Visualization functions (2D/3D plots, PCA, feature analysis)
- **rt_alignment_utils.py** - Retention time alignment utilities
- **misc.py** - General utility functions

### Data Flow
1. Raw CSV files → Heatmap extraction (per m/z value)
2. Heatmaps → Peak detection and clustering
3. Peaks → Feature extraction for classification
4. Features → Binary classification model training
5. Models → Sample prediction and evaluation

## Dependencies

Core dependencies (from setup.py):
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning algorithms
- scipy - Scientific computing
- seaborn, plotly - Data visualization
- loguru - Logging
- tqdm - Progress bars

## Development Notes

- The system processes large chromatographic datasets with parallel processing support
- Peak detection uses DBSCAN clustering with configurable noise filtering
- Classification features are extracted from detected peak regions
- All processing steps are configurable via JSON configuration files
- Visualization utilities support both 2D and interactive 3D plotting