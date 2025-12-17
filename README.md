# OpenMM Chemical Shift Restraint Plugin
This is a plugin for OpenMM which implements a restraining force derived from NMR Chemical Shift (CS) data. This is achieved via the NapShift neural network model, which predicts NMR CS from simulated protein structures. Predicted simulated CS are compared with experimentally measured values, and a restraining force is applied to minimise the calculated discrepancy. 

The main contribution of this repository is the `NapShiftForce` class, which derives from the OpenMM `Force` class, and provides implements the aforementioned restraining force. Additionally, we provide several utility functions for parsing raw CS data into a simulation-ready format, and for setting-up NapShiftForce objects to apply an input set of CS restraints to a given system.

# Installation
## Installing with conda
⚠️ TEMPORARY ⚠️

We are still in the process of submitting this project to conda-forge, so for now it is not installable via conda. In the meantime, this functionality is provided by the package file available in [environments](environments). 
```
cd environments
# conda environment containing all dependencies required to run the tutorials in this project
conda env create -f run.yml 
conda activate OpenMMNapShift
conda install openmmnapshift-1.0.0-py_0.conda
 # additional dependency not yet published to conda-forge, required to compute random coil chemical shifts 
conda install pycamcoil-1.0.0-py_0.conda
```

## Building from source
1. Install dependencies:
   
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
CS restraints can be applied to atomistic-resolution simulations, simulations with the Martini3 forcefield, and simulations with dramatically coarse-grained forcefields which represent only CA atoms. Tutorials for each of these are provided:
- [Creating Chemical Shift input files](tutorials/create_input_files.ipynb)
- [Running all-atomistic simulations with Chemical Shift restraints](tutorials/all_atom_tutorial.ipynb)
- [Running Martini3 simulations with Chemical Shift restraints](tutorials/martini_tutorial.ipynb)
- [Running dramatically coarse-grained (e.g. CALVADOS) simulations with Chemical Shift restraints](tutorials/CA_tutorial.ipyn)








