# active-csp

Welcome to the GitHub repository for our software package designed for active learning with neural network ensembles in the context of crystal structure search. This repository contains the codebase used in our recent publication, which leverages high-performance computing resources for efficient and effective crystal structure prediction.

## Overview

This software package facilitates the active learning process for crystal structure search by utilizing neural network ensembles. The main job script, `active_csp_run.py`, orchestrates the submission of various worker jobs responsible for:

- Training machine learning force fields
- Computing reference data using DFT
- Running structure relaxations
- Validating final structures with DFT

Each worker job is associated with a separate script that is installed alongside the main package.

## Installation

To install the package and the corresponding scripts, navigate to the repository root directory and run:

```bash
python setup.py install
```

## Running the Code

To test the installation, start a simulation on Si2 with the following command:

```bash
python src/scripts/active_csp_run.py experiment=si2
```

This script is designed to run on our SLURM-based high-performance computing (HPC) cluster due to the significant computational power required for the tasks. Adapting the code to run on other systems may necessitate writing your own hardware interface. As a reference, you can modify the example provided in `src/active_csp/jobs/hardware_interface`.

## Configuration

We utilize Hydra for configuring our experiments. The settings used in our publication are available in `src/scripts/configs/experiments`. Hydra allows for flexible and hierarchical configuration management, making it easier to reproduce and tweak experiments.

## Dependencies

Currently, the package supports the following interfaces:

- **DFT**: Quantum Espresso
- **Machine learning forces fields**: SchNetPack

Additional tools for labeling data and force field models can be integrated by implementing the necessary interfaces.

## Citation

If you use this code in your research, please cite our publication:

```
@article{wip}
```
