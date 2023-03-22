# Single-cell imaging-AI based chromatin biomarkers for proton therapy efficacy in tumor patients using liquid biopsies

The repository contains the code used to run the analyses presented in our publication:

> [**Single-cell imaging-AI based chromatin biomarkers for proton therapy efficacy in tumor patients using liquid biopsies (Under review)**]()

<p align="center" width="100%">
  <b>Chromatin organization of PBMCs reflects the presence of tumor signals</b> <br>
    <img width="66%" src="">
</p>

----

# System requirements

The code has been developed and executed on a HP Z4 workstation running Ubuntu 20.04.5 LTS with a Intel(R)
Xeon(R) W-2255 CPU with 3.70 GHz, 128GB RAM. Note that the code can also be run for machines with less available RAM.

## Installation

To install the code, please clone the repository and install the required software libraries and packages listed in
the **requirements.txt** file:

```
git clone https://github.com/GVS-Lab/immune_cell_project.git
conda create --name icp --file requirements.txt
conda activate icp
```

## Data resouces (Optional)

Intermediate results of the analysis can be obtained from our [Google Drive here](https://drive.google.com/drive/folders/1HszNjSRFI2x25mEDQo-a_rKpemwtJZ4C?usp=sharing) but can also be produced using the steps described below to reproduce the results of the paper. If you want to use and/or adapt the code to run another analysis, the data is not required neither.

---

# Reproducing the paper results

## 1. Data preprocessing

The data preprocessing steps quantile-normalize the data, segment individual nuclei and cells as well as measure the
chrometric features described
in [Venkatachalapathy et al. (2020)](https://www.molbiolcell.org/doi/10.1091/mbc.E19-08-0420) for each nucleus and
quantify the associated cellular expression of the proteins stained for in the processed immunofluorescent images. To
preprocess the imaging data for the analysis of the B-cell populations in the germinal centers or the correlation
analysis of the selected microimages please use the notebooks ```notebooks/dataset1/feature_generation.ipynb```
or ```notebooks/dataset3/feature_generation.ipynb``` respectively.

## 2. Reproducing the figure results

To run the analysis regarding the different B-cell populations in the light respectively dark zone of the germinal
centers, please use the code provided in the
notebook ```notebooks/dataset1/light_vs_darkzone_bcells_and_tcell_integration.ipynb```.

---

# How to cite

If you use any of the code or resources provided here please make sure to reference the required software libraries if
needed and also cite our work:

**TO BE ADDED**
---
