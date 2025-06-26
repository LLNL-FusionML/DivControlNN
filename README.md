# DivControlNN and KSTAR_UEDGE_Dataset

## Project Overview

This repository contains two complementary components developed under the DOE FES Long-Pulse Tokamak project:

### 1. DivControlNN

DivControlNN is a machine-learning-based surrogate model for real-time divertor detachment control in magnetic fusion devices. Trained on 75,000 2D drift-included UEDGE simulations from KSTAR discharges, it predicts and regulates detachment states based on plasma boundary conditions for integration with real-time control systems.

### 2. KSTAR_UEDGE_Dataset

This dataset contains edge plasma simulation outputs from UEDGE for the KSTAR tokamak. It includes input boundary conditions, 2D output profiles, and derived physical quantities used to train and validate DivControlNN. The dataset is formatted for direct use with ML frameworks.

## Usage Instructions

### For DivControlNN:
Navigate to the `/DivControlNN` directory for:
- Model training scripts
- Configuration files
- Test modules

### For the dataset:
Navigate to `/KSTAR_UEDGE_Dataset` for:
- Structured HDF5 files
- Parsing and preprocessing scripts

## Licensing

- **DivControlNN Code**  
  Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)  
  LLNL Review & Release Number: **LLNL-CODE-2007259**

- **KSTAR_UEDGE_Dataset**  
  Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
  LLNL Review & Release Number: **LLNL-DATA-2005948**

## Citation

If you use this code or dataset, please cite the following:

### Code

**Citation Key:** `zhu2025_divcontrolnn`  
**Authors:** Zhu, Ben; Zhao, Menglong; Xu, Xueqiao  
**Title:** *DivControlNN: ML-Based Surrogate for Divertor Detachment Control*  
**Year:** 2025  
**Publisher:** Zenodo  
**DOI:** [10.5281/zenodo.15749609](https://doi.org/10.5281/zenodo.15749609)  
**URL:** [https://github.com/xxu/DivControlNN](https://github.com/LLNL-FusionML/DivControlNN)

### Dataset

**Citation Key:** `zhao2025_kstaruedge`  
**Authors:** Zhao, Menglong; Zhu, Ben; Xu, Xueqiao  
**Title:** *KSTAR_UEDGE_Dataset: Edge Plasma Simulations for KSTAR using 2D UEDGE*  
**Year:** 2025  
**Publisher:** Zenodo  
**DOI:** [10.5281/zenodo.TBD2](https://doi.org/10.5281/zenodo.TBD2)  
**URL:** [https://zenodo.org/record/TBD](https://zenodo.org/record/TBD)

## Contact

For questions or collaboration inquiries, please contact:

- Ben Zhu: zhu12@llnl.gov  
- Menglong Zhao: zhao17@llnl.gov  
- Xueqiao Xu: xu2@llnl.gov

## Repository Strategy

We follow a two-tier release strategy:

- **Code (DivControlNN)**  
  Hosted on GitHub → Linked to Zenodo for DOI generation → Includes license and citation metadata.

- **Dataset (KSTAR_UEDGE_Dataset)**  
  Published on Zenodo with an assigned DOI → Referenced within the GitHub repository.

## Badges (to be added after DOI assignment)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
<!--
[![DOI for Code](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD1.svg)](https://doi.org/10.5281/zenodo.TBD1)
[![DOI for Dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD2.svg)](https://doi.org/10.5281/zenodo.TBD2)
-->
