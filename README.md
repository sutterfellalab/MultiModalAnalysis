# Multi-Modal Analysis

This repository contains a Python script designed for the analysis of multimodal in-situ data taken at beamline 12.3.2 of the Advanced Lightsource (ALS). The script performs various data processing tasks including timestamp-adjustment, data selection, detector-geometry calibration, diffraction data integration, peak fitting, and various ways of data visualization. It uses in-situ photoluminescence (PL) and (grazing incidence) wide-angle X-ray scattering (GI-WAXS) data as well as logged process parameters as input. MMAnalysis will ask for the GI-WAXS calibration first, which should be done with the pyFAI-GUI unless a local .poni was created before. After the calibration, MMA is going to ask for the calibration.poni file as well as the in situ run files (*.h5).

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
- **GIWAXS Data Selection**: Automatically finds suggested start times, plots raw and post-processed GIWAXS data, and performs peak fitting. Additionally, it gives an option to extract individual frames for x-y-plots.
- **PL Data Selection**: Plots raw and post-processed PL data (PL data have the same timestamp as the logging data), optimizes data for plotting, and performs peak fitting. Additionally, it gives an option to extract individual frames for x-y-plots.
- **Stacked Plots**: Generates stacked plots for combined GIWAXS, PL, and logging data.
- **Output: the script creates a new "output" folder containing all the images displayed during execution as well as all relevant data in .csv files

## Contact

Feel free to create Merge Requests and Issues on our GitHub page: [https://github.com/sutterfellalab/MultiModalAnalysis](https://github.com/sutterfellalab/MultiModalAnalysis).

If you want to contact the authors, please write to T. Kodalle at <TimKodalle@lbl.gov>.

