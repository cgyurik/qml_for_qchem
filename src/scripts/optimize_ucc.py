"""Script to perform a UCC optimization given a filename."""

import sys
import os
import time
import json

from typing import Generator, Tuple, Union

import numpy
import scipy.optimize

import openfermion
import cirq

# Relative imports from the package. This code is needed for the imports to
# work even if this file is run as a script.

# pylint: disable=wrong-import-position
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.ansatz_functions.ucc_functions import (
    generate_ucc_amplitudes,
    generate_circuit_from_pauli_string,
    generate_ucc_operators)
from src.vqe_functions.vqe_optimize_functions import (
    circuit_state_fidelity,
    circuit_state_expval)
from src.utils import (
    load_data,
    load_ucc_data,
    MOLECULES_DIR,
    UCC_DIR,
    encode_complex_and_array)

# pylint: enable=wrong-import-position


def load_molecule(filename) -> openfermion.MolecularData:
    """Load the molecule given the filename"""
    molecule = openfermion.MolecularData(filename=MOLECULES_DIR + filename)
    molecule.load()
    return molecule


def singlet_hf_generator(n_electrons: int,
                         n_orbitals: int) -> Generator:
    """
    Add X gate to qubits 0 to n_electrons.
    """
    for i in range(n_electrons):
        yield cirq.X(cirq.GridQubit(1, i))
    for i in range(n_electrons, 2 * n_orbitals):
        yield cirq.I(cirq.GridQubit(1, i))


def triplet_hf_generator(n_electrons: int,
                         n_orbitals: int) -> Generator:
    """
    Add X gate to qubits 0 to n_electrons.
    """
    for i in range(n_electrons - 1):
        yield cirq.X.on_each(cirq.GridQubit(1, i))
    yield cirq.I.on(cirq.GridQubit(1, n_electrons - 1))
    yield cirq.X.on(cirq.GridQubit(1, n_electrons))
    for i in range(n_electrons + 1, 2 * n_orbitals):
        yield cirq.I.on_each(cirq.GridQubit(1, i))


def optimize_ucc(
    filename: str,
    initial_params: Union[Tuple[float, ...], None] = None,
    maxiter: int = 10000
) -> dict:
    """
    Run the optimization sequence

    Args:
        filename: string representing the molecule geometry (see
            `notebooks/Guide_to_data`), without any extension or directory.
        initial_params: initial UCC optimization params. If None (default),
            parameters are all set to zero (HF state).
        maxiter: the maximum number of iterations for the COBYLA optimizer

    Returns:
        dictionary containing optimized parameters and some benchmark data,
        namely the dictionary keys are:
            - 'params': optimized parameters
            - 'infidelity': final state infidelity with ground-subspace
            - 'energy_expval': final state energy expectation value
            - 'energy_error': final state error in energy
            - 'optimizer_success': whether optimization succeeded
            - 'optimizer_nfev': number of required function evaluations
    """

    molecule = load_molecule(filename)
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(
            molecule.get_molecular_hamiltonian()))

    data_dict = load_data(filename)
    ground_states = data_dict['ground_states']

    singles, doubles = generate_ucc_amplitudes(molecule.n_electrons,
                                               2 * molecule.n_orbitals)
    ucc_ferop = generate_ucc_operators(singles, doubles)

    simulator = cirq.Simulator()
    parameter_dict = {}
    circuit = cirq.Circuit(
        singlet_hf_generator(molecule.n_electrons, molecule.n_orbitals)
        if molecule.multiplicity == 1 else
        triplet_hf_generator(molecule.n_electrons, molecule.n_orbitals)
    )

    for i, op in enumerate(ucc_ferop):
        parameter_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))

    if initial_params is None:
        initial_params = numpy.zeros(len(parameter_dict))

    def cost_function(params):
        return 1 - circuit_state_fidelity(params, circuit, parameter_dict,
                                          ground_states, simulator)

    stopwatch = time.time()
    print('Starting optimization.')
    result = scipy.optimize.minimize(
        cost_function,
        x0=(initial_params),
        options={'maxiter': maxiter},
        method='COBYLA')
    print(result)
    print('Optimization time: {} minutes \n'
          .format((time.time() - stopwatch) / 60))

    optimized_energy = circuit_state_expval(
        result['x'], circuit, parameter_dict,
        qubit_hamiltonian, simulator)

    return dict(
        params=result['x'],
        infidelity=result['fun'],
        energy_expval=optimized_energy,
        energy_error=optimized_energy - data_dict['exact_energy'],
        optimizer_success=result['success'],
        optimizer_nfev=result['nfev']
    )


# ***  Main script code  ***
if __name__ == '__main__':

    usage = 'Usage: python {} <filename>'.format(sys.argv[0])
    if len(sys.argv) is not 2:
        print(usage)
        raise Exception('wrong usage')
    if not isinstance(sys.argv[1], str):
        print(usage)
        raise TypeError('The first argument is not a string')
    filename = sys.argv[1]

    existing_ucc_files = os.listdir(UCC_DIR)
    if (filename + '.json' in existing_ucc_files):
        print('The file data/ucc/{}.json exists already. loading file.')
        ucc_dict = load_ucc_data(filename)
    else:
        ucc_dict = optimize_ucc(filename)

        print('saving data to file.')
        with open(UCC_DIR + filename + '.json', 'wt') as f:
            json.dump(ucc_dict, f, default=encode_complex_and_array)

    print(*((k, v) for k, v in ucc_dict.items()), sep='\n')
