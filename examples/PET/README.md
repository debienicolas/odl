# PET examples

These examples demonstrate the capability of ODL to perform PET (Positron Emission Tomography) forward projections and back-projections using polygon-based scanner geometries. They support both sinogram-based and list-mode data formats, with optional Time-of-Flight (TOF) information.

For examples on how to use the PET operators in inverse problems, see the [examples/solvers](../solvers) folder.

Note that these PET examples require the [parallelproj](https://parallelproj.readthedocs.io/en/stable/) python library to be installed:

    conda install -c conda-forge parallelproj

## Basic usage examples

### Sinogram projectors

Example | Purpose  
------- | ------- 
[`Non_TOF_3d_sinogram_projector.py`](Non_TOF_3d_sinogram_projector.py) | Forward and back-projection in 3D non-TOF sinogram format 
[`TOF_3d_sinogram_projector.py`](TOF_3d_sinogram_projector.py) | Forward and back-projection in 3D TOF sinogram format 

### List-mode projectors

Example | Purpose  
------- | ------- 
[`Non_TOF_3d_listmode_projector.py`](Non_TOF_3d_listmode_projector.py) | Forward and back-projection in 3D non-TOF list-mode format
[`TOF_3d_listmode_projector.py`](TOF_3d_listmode_projector.py) | Forward and back-projection in 3D TOF list-mode format
