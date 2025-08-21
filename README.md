<div align="center">
  <img src="img/logo_LT.png" width="400px" alt="LifeTracer Logo">
</div>

# LifeTracer: comprehensive Python package for 2D gas chromatography analysis and signature discovery



[![Project Website](https://img.shields.io/badge/Project-Website-orange.svg)](https://life-tracer.github.io/)
[![Version](https://img.shields.io/badge/version-1.0.0.0-blue.svg)](https://github.com/amirgroup-codes/LifeTracer)
[![Python](https://img.shields.io/badge/python-3.10.8-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**LifeTracer** is a comprehensive Python package for 2D gas chromatography analysis and molecular classification. This package provides tools to distinguish between abiotic and biotic organic compounds in meteorite and terrestrial samples using machine learning on mass spectrometry data.

**Project Website**: [https://life-tracer.github.io/](https://life-tracer.github.io/)

## Publication

**Paper Title**: *Discriminating Abiotic and Biotic Organics in Meteorite and Terrestrial Samples Using Machine Learning on Mass Spectrometry Data*

### Abstract
With the upcoming sample return missions to the Solar System where traces of past, extinct, or present life may be found, there is an urgent need to develop unbiased methods that can distinguish molecular distributions of organic compounds synthesized abiotically from those produced biotically but were subsequently altered through diagenetic processes. We conducted untargeted analyses on a collection of meteorite and terrestrial geologic samples using two-dimensional gas chromatography coupled with high-resolution time-of-flight mass spectrometry (GC×GC-HRTOF-MS) and compared their soluble non-polar and semi-polar organic species. To deconvolute the resulting large dataset, we developed LifeTracer, a computational framework for processing and downstream machine learning analysis of mass spectrometry data. LifeTracer identified predictive molecular features that distinguish abiotic from biotic origins and enabled a robust classification of meteorites from terrestrial samples based on the composition of their non-polar soluble organics.
## Tools

- **Data Processing**: Comprehensive preprocessing pipeline for GC×GC-HRTOF-MS data
- **Feature Extraction**: Automated extraction of molecular features from mass spectra
- **Machine Learning**: Built-in classification algorithms optimized for distinguishing abiotic/biotic origins
- **Visualization**: Tools for visualizing chromatographic data and classification results

## Getting Started

To reproduce the results from our research paper, please refer to our comprehensive notebook:
[Getting Started Notebook](https://github.com/amirgroup-codes/LifeTracer/blob/main/Getting_Satrted.ipynb)

This notebook provides a step-by-step guide through the complete pipeline from raw chromatographic data to trained classification models.

## Installation

### Requirements

- Python 3.10.8
- Anaconda or Miniconda

### Linux/macOS Installation

#### Step 1: Create Environment

```bash
conda create -n LifeTracer python=3.10.8
```

#### Step 2: Activate Environment

```bash
conda activate LifeTracer
```

#### Step 3: Install Package

```bash
# Clone the repository
git clone https://github.com/amirgroup-codes/LifeTracer.git

# Navigate to project directory
cd LifeTracer

# Install in development mode
pip install -e .
```

#### Step 4: Verify Installation

```bash
python -c "import lifetracer; print('LifeTracer installed successfully!')"
```

### Windows Installation

The installation process is identical to Linux/macOS. Use Anaconda Prompt or PowerShell:

```powershell
# Create environment
conda create -n LifeTracer python=3.10.8

# Activate environment
conda activate LifeTracer

# Clone and navigate to repository
git clone https://github.com/amirgroup-codes/LifeTracer.git
cd LifeTracer

# Install package
pip install -e .

# Verify installation
python -c "import lifetracer; print('LifeTracer installed successfully!')"
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **Permission Errors** | Use `pip install --user -e .` |
| **Environment Issues** | Run `conda clean --all` and recreate environment |
| **Path Problems** | Ensure you're in the correct project directory |
| **Import Errors** | Check Python version matches 3.10.8 |

# Data

The package works with both raw and processed chromatography data. Data files can be downloaded from hugginface:

## Raw Data (GC×GC-HRTOF-MS CSVs)

| Group     | Sample / Item                           | Link                                                                                                                                                             | Description                                                                        |
| --------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Meteorite | Murchison Pristine 2.0 (replicate -003) | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230823_03_Murchison_Pristine_2.0_300uLDCM_100oC24h-003.csv) | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | EET96029                                | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230830_01_EET96029_300uLDCM_100oC24h.csv)                   | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | Orgueil (replicate -001)                | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230830_02_Orgueil_300uLDCM_100oC24h-001.csv)                | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | ALH83100                                | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230901_06_ALH83100_300uLDCM_100oC24h.csv)                   | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | LON94101                                | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230901_07_LON94101_300uLDCM_100oC24h.csv)                   | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | LEW85311                                | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/230901_08_LEW85311_300uLDCM_100oC24h.csv)                   | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Meteorite | AZ                                      | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/231003_01_AZ_400uLDCM_100oC24h.csv)                         | Raw instrument export; meteorite extract; 400 µL DCM; 100 °C for 24 h.             |
| Meteorite | Jbilet Winselwan                        | [click here to download](https://huggingface.co/datasets/DS-20202/Meteorites_LifeTracer/resolve/main/231003_02_Jbilet_Winselwan_300uLDCM_100oC24h.csv)           | Raw instrument export; meteorite extract; 300 µL DCM; 100 °C for 24 h.             |
| Soil      | Atacama Soil (replicate -001)           | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_01_Atacama_Soil_300uLDCM_100oC24h-001.csv)           | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Rio Tinto Soil                          | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_02_Rio_Tinto_Soil_300uLDCM_100oC24h.csv)             | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Murchison Soil (replicate -001)         | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_04_Murchison_Soil_300uLDCM_100oC24h-001.csv)         | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Antarctica Soil (replicate -001)        | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_05_Antarctica_Soil_300uLDCM_100oC24h-001.csv)        | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Jarosite Soil                           | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_06_Jarosite_Soil_300uLDCM_100oC24h.csv)              | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Green River Shale Soil                  | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230823_07_Green_River_Shale_Soil_500uLDCM_100oC24h.csv)     | Raw instrument export; soil extract; 500 µL DCM; 100 °C for 24 h.                  |
| Soil      | GSFC Soil                               | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230901_05_GSFC_soil_300uLDCM_100oC24h.csv)                  | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Lignite (replicate -001)                | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/230830_03_Lignite_300uLDCM_100oC24h-001.csv)                | Raw instrument export; soil/organic sediment extract; 300 µL DCM; 100 °C for 24 h. |
| Soil      | Utah Soil                               | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/231003_03_Utah_Soil_300uLDCM_100oC24h.csv)                  | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |
| Soil      | Iceland Soil                            | [click here to download](https://huggingface.co/datasets/DS-20202/SoilSample-LifeTracer/resolve/main/231003_04_Iceland_Soil_300uLDCM_100oC24h.csv)               | Raw instrument export; soil extract; 300 µL DCM; 100 °C for 24 h.                  |

## Processed Data

| Set                       | File / Part                 | Link                                                                                                                                 | Description                                                                              |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Unaligned TIIs (heatmaps) | heatmaps.tar.gz.part-aa     | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/heatmaps.tar.gz.part-aa)    | Split archive part; unaligned Total Ion Image heatmaps. Concatenate parts, then extract. |
| Unaligned TIIs (heatmaps) | heatmaps.tar.gz.part-ab     | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/heatmaps.tar.gz.part-ab)    | Split archive part; unaligned TII heatmaps.                                              |
| Unaligned TIIs (heatmaps) | heatmaps.tar.gz.part-ac     | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/heatmaps.tar.gz.part-ac)    | Split archive part; unaligned TII heatmaps.                                              |
| Unaligned TIIs (heatmaps) | heatmaps.tar.gz.part-ad     | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/heatmaps.tar.gz.part-ad)    | Split archive part; unaligned TII heatmaps.                                              |
| Aligned TII               | TII\_aligned.tar.gz.part-aa | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-aa) | Split archive part; aligned Total Ion Images. Concatenate parts, then extract.           |
| Aligned TII               | TII\_aligned.tar.gz.part-ab | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ab) | Split archive part; aligned TIIs.                                                        |
| Aligned TII               | TII\_aligned.tar.gz.part-ac | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ac) | Split archive part; aligned TIIs.                                                        |
| Aligned TII               | TII\_aligned.tar.gz.part-ad | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ad) | Split archive part; aligned TIIs.                                                        |
| Aligned TII               | TII\_aligned.tar.gz.part-ae | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-ae) | Split archive part; aligned TIIs.                                                        |
| Aligned TII               | TII\_aligned.tar.gz.part-af | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/TII_aligned.tar.gz.part-af) | Split archive part; aligned TIIs.                                                        |
| Peaks                     | peaks.zip                   | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/peaks.zip)                  | Detected peak tables per sample (e.g., retention times, intensities).                |
| Features                  | features.zip                | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/features.zip)               | Feature matrix/metadata derived from peaks/TII (aligned features for modeling).          |
| Calibration Phase         | calibration\_phase.zip      | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/calibration_phase.zip)      | Intermediate files for automatic parameter selection                             |
| Parameter Selection       | parameters\_selection.zip   | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/parameters_selection.zip)   | Parameter sweeps/selection results used for final model.                    |
| Final Paper Results       | lr\_l2\_results.zip         | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/lr_l2_results.zip)          | Final result bundle (e.g., logistic-regression L2 results reported in the paper).        |
| Model Evaluations         | eval.zip                    | [click here to download](https://huggingface.co/datasets/DS-20202/LifeTracer-Processed-Data/resolve/main/eval.zip)                   | Evaluation outputs and metrics for trained models.                                       |

*Note:* For any `.tar.gz.part-xx` sets, concatenate parts in order (e.g., `cat file.tar.gz.part-* > file.tar.gz`) before extracting.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please:
- Open an issue on [GitHub](https://github.com/amirgroup-codes/LifeTracer/issues)
- Visit our [project website](https://life-tracer.github.io/)

## Acknowledgments

We thank all contributors and collaborators who have helped develop LifeTracer and the institutions that supported this research. US Antarctic meteorite samples are recovered by the Antarctic Search for Meteorites (ANSMET) program, which has been funded by NSF and NASA and characterized and curated by the Department of Mineral Sciences of the Smithsonian Institution and Astromaterials Curation Office at NASA Johnson Space Center. The authors would like to thank T. McCoy, J. Hoskin, and the Smithsonian National Museum of Natural History - Division of Meteorites; and J.-C. Viennet and the curatorial team at the Muséum National d’Histoire Naturelle for providing the meteorite samples used in this study. This research was supported in part by the Parker H. Petit Institute for Bioengineering and Biosciences (IBB) interdisciplinary seed grant, the Institute of Matter and Systems (IMS) Exponential Electronics seed grant, and the Georgia Institute of Technology start-up funds, and by NASA’s Planetary Science Division Internal Scientist Funding Program through the Fundamental Laboratory Research (FLaRe) Work Package at NASA Goddard Space Flight Center.
