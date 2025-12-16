# OpenMM Chemical Shift Restraint Plugin
This is a plugin for OpenMM which implements a restraining force derived from NMR Chemical Shift (CS) data. This is achieved via the NapShift neural network model, which predicts NMR CS from simulated protein structures. Predicted simulated CS are compared with experimentally measured values, and a restraining force is applied to minimise the calculated discrepancy. 

The main contribution of this repository is the `NapShiftForce` class, which derives from the OpenMM `Force` class, and provides implements the aforementioned restraining force. Additionally, we provide several utility functions for parsing raw CS data into a simulation-ready format, and for setting-up NapShiftForce objects to apply an input set of CS restraints to a given system.

# Installation
## Installing with conda
## Building from Source

# Using the OpenMM NapShift plugin
CS restraints can be applied to atomistic-resolution simulations, simulations with the Martini3 forcefield, and simulations with forcefields which use only CA atoms. Tutorials for each of these are provided:

