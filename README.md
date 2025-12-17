# OpenMM Chemical Shift Restraint Plugin
This is a plugin for OpenMM which implements a restraining force derived from NMR Chemical Shift (CS) data. This is achieved via the NapShift neural network model, which predicts NMR CS from simulated protein structures. Predicted simulated CS are compared with experimentally measured values, and a restraining force is applied to minimise the calculated discrepancy. 

The main contribution of this repository is the `NapShiftForce` class, which derives from the OpenMM `Force` class, and provides implements the aforementioned restraining force. Additionally, we provide several utility functions for parsing raw CS data into a simulation-ready format, and for setting-up NapShiftForce objects to apply an input set of CS restraints to a given system.

# Installation
## Installing with conda
## Building from source
1. Install dependencies
This project depends on the OpenMM and LibTorch libraries. A CUDA compiler is also required. OpenMM is available via conda, and instructions for LibToch installation can be found at https://pytorch.org. Alternatively, an environment file containing all required dependencies to build OpenMMNapShift can be found at [environments/build.yml](environments/build.yml):
```
cd environments
conda env create -f build.yml
conda activate BuildOpenMMNapShift
cd ..
```
2. Make build directory
Create an empty directory to contain build outputs
```
mkdir build
cd build
```
3. Run CMake
Run CMake and specify where to source the OpenMM and LibTorch libraries, and where to find the CUDA compiler. 
```
cmake -DOPENMM_DIR <path to OpenMM> -DPYTORCH_DIR <path to LibTorch> -DCMAKE_CUDA_COMPILER <path to CUDA compiler> ..
```
If you installed dependencies using `build.yml`, you should be able to use:
```
<path to OpenMM>: <path to BuildOpenMMNapShift environment>
<path to LibTorch>: <path to BuildOpenMMNapShift environment>/lib/<python version>/site-packages/torch
<path to CUDA compiler>: <path to BuildOpenMMNapShift environment>/bin/nvcc
```
4. Run build targets
```
make install
make PythonInstall
```
5. Install the newly built package
```
cd python
python -m pip install .
```

# Using the OpenMM NapShift plugin
CS restraints can be applied to atomistic-resolution simulations, simulations with the Martini3 forcefield, and simulations with forcefields which use only CA atoms. Tutorials for each of these are provided:


