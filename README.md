# LifeTracer

**Processing 2D Gas Chromatography and signature discovery**

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
## Features

- **Data Processing**: Comprehensive preprocessing pipeline for GC×GC-HRTOF-MS data
- **Feature Extraction**: Automated extraction of molecular features from mass spectra
- **Machine Learning**: Built-in classification algorithms optimized for distinguishing abiotic/biotic origins
- **Visualization**: Tools for visualizing chromatographic data and classification results
- **Reproducibility**: Complete workflow for reproducing published results

## Getting Started

To reproduce the results from our research paper, please refer to our comprehensive notebook:
[Getting Started Notebook](https://github.com/amirgroup-codes/LifeTracer/blob/main/Getting_Started.ipynb)

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

## Data

### Download Options

The package works with both raw and processed chromatography data. Data files can be downloaded from hugginface:

1. **Raw Data**: Unprocessed GC×GC-HRTOF-MS output files
2. **Processed Data**: Pre-processed datasets ready for analysis

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please:
- Open an issue on [GitHub](https://github.com/amirgroup-codes/LifeTracer/issues)
- Visit our [project website](https://life-tracer.github.io/)

## Acknowledgments

We thank all contributors and collaborators who have helped develop LifeTracer and the institutions that supported this research.
