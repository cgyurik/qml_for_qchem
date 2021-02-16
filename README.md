# QML for QChem
All code for the "Quantum Machine Learning for Quantum Chemistry" project. 


## This repository

Contents of the code folder `gs_energy-prediction`:
- `run.py`: executable python file that performs an groundstate energy prediction experiment.
- `tfq_model.py`: python file containing the main class for the QML model in the groundstate energy prediction experiment.
- `notebooks`: directory containing jupyter-notebooks explaining the data and benchmarking experiments with classical models.
- `utils` directory containing files for generating/loading data, vqe optimization and pqc ansatzes.
- `data` directory containing all data of generated molecules.

Contents of the code folder `vqe-surrogate`:
- `main.py`: executable python file that performs an vqe surrogate experiment.
- `vqe.py`: python file containing the main class for the vqe with surrogate cost function.
- `qml_model`: directory containing python files for the QML model that implements the surrogate vqe cost function.
- `molecules`: directory containing a family of H4 molecules to use for the vqe with surrogate experiments.
- `vqe_utils` directory containing python files for vqe functionalities.
- `results` directory containing all results from previous experiments.


Additionally, at the root level of this repository, we find:
- `figures` directory: figures for use in this README file etc.
- setup code for continuous integration


---

## Definition of the groundstate energy prediction problem:

### Initial Goal: 
For a family of chemical Hamiltonians (i.e., molecules), use a QML model to **predict some property of the molecules**. 
The QML model takes as an input *K* copies of a quantum state *ρ* (encoding some properties of the molecule) and some classical information on the molecule, and outputs the best estimate of property to predict.

![structure -> property diagram](figures/diagram.png)

### Definition of the problem to be learned:
**Input**: 
- quantum state(s) {ρ} (quantum data encoding the molecule)
- structure/parameters of the molecule ω (classical data, size O(N)).

**Output**: 
- Estimate of the property to predict.

### QML model:
The quantum state *ρ* is fed into a PQC whose parameters are obtained by preprocessing the classical data via a neural network.
Local measurements are taken on the output state, and post-processed by a neural network to yield an estimate of the property we wish to predict.

![generic model drawing](figures/generic_model.png)

## Serial and Parallel PQCs.

The PQC can reupload different copies of the groundstate both serial or in parallel:

![serial model drawing](figures/serial_model.png)

![parallel model drawing](figures/parallel_model.png)

---

## Simple first test case:

As a first test case, we try to predict the ground state energy *E_GS*, given *K* copies of the ground state |*GS*> and some information on the molecule, for the moelcule family H4.

### Instance system
A molecule in the H4 family, with the following restrictions on the geometry:
- no pair of atoms farther than 0.4 Angstrom (avoid exaggerate overlaps)
- no pair of consecutive atoms farther than 1.5 Angstrom (avoid complete dissociation)

**System modelling**:
The space is parametrized by STO-3G atomic orbitals (for each H atom A single spherically-symmetric orbital, i.e. 2 spin-orbitals).
The fermionic states are represented in the canonical orbital basis, and mapped to the qubit register by the Jordan-Wigner transform.

### QML model I/O

**quantum input of our PQC**: 
- The ground state of the molecule
    
**classical input of NNs**:
- The geometry of the molecule
- The orbital energies (single-particle energies for each canonical orbital)
- The canonical orbital matrix, which indicates which linear combination of the atomic orbitals (STO-3G basis functions) construct the molecular orbitals

**Model output**:
- Estimate of the ground state energy of the instance molecule.

