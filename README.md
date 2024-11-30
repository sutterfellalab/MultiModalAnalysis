# Multi Modal Analysis

This repository contains a Python script designed for analysis tasks related to MMAnalysis. The script performs various data processing tasks including logging data selection, GIWAXS data selection, and PL data selection. It also allows for peak fitting and generates stacked plots.

## Requirements

Check the file [requirements.txt](requirements.txt) to see which packages are needed. Installing the package using `pip` should already take care of all dependencies.

## Installation instructions

### Create a new virtual environment

Create a new Python environment. (You can also do it in a pre-existing environment, but make sure you don't break something):

```bash
conda create -n mmanalysis python=3.11
conda activate mmanalysis
```

Note that you may need to initialize your shell within conda, e.g., using conda init bash. You will know if the conda environment has been activated when you see that your shell prompt is modified with (`mmanalysis`).

After activating your new (or existing) environment, follow the next steps.

### Install using `pip`

You can simply install the latest release of the package and all dependencies using:

```bash
pip install mmanalysis
```

### Install directly the source code

Alternatively you can obtain `mmanalysis` directly from the repository by following those steps:

Clone the repository in the desired location:

```bash
git clone https://github.com/sutterfellalab/MultiModalAnalysis.git
```

Install the required packages:

```bash
cd MultiModalAnalysis
conda install -c conda-forge --file requirements.txt
```

Install the package with pip:

```bash
pip install .
```

## Features

- **Logging Data Selection**: Automatically suggests start times and plots raw and post-processed log data.
- **GIWAXS Data Selection**: Automatically finds start times, plots raw and post-processed GIWAXS data, and performs peak fitting.
- **PL Data Selection**: Plots raw and post-processed PL data, optimizes data for plotting, and performs peak fitting.
- **Stacked Plots**: Generates stacked plots for combined GIWAXS, PL, and logging data.

## Contact

Feel free to create Merge Requests and Issues on our GitHub page: [https://github.com/sutterfellalab/MultiModalAnalysis](https://github.com/sutterfellalab/MultiModalAnalysis).

If you want to contact the authors, please write to T. Kodalle at <TimKodalle@lbl.gov>.

