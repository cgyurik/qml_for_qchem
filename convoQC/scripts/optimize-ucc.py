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

# pylint: disable=wrong-import-position
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from ansatz_functions.ucc_functions import (  # type: ignore
    generate_ucc_amplitudes,
    generate_circuit_from_pauli_string,
    generate_ucc_operator)
from vqe_functions.vqe_optimize_functions import (  # type: ignore
    overlap_with_circuit_state,
    expectation_value_with_circuit_state)
from utils.load_lib import (  # type: ignore
    load_data,
    load_ucc_data,
    MOLECULES_DIR,
    UCC_DIR)
from utils.generic import encode_complex_and_array  # type: ignore
# pylint: enable=wrong-import-position


def load_molecule(filename) -> openfermion.MolecularData:
    """Load the molecule given the filename"""
    molecule = openfermion.MolecularData(filename=MOLECULES_DIR + filename)
    molecule.load()
    return molecule


def initialize_hf_state(n_electrons: int) -> Generator:
    """Add X gate to qubits 0 to n_electrons."""
    yield cirq.X.on_each(cirq.LineQubit.range(n_electrons))


def optimize_ucc(
    filename: str,
    initial_params: Union[Tuple[float, ...], None] = None
) -> dict:
    """
    Run the optimization sequence

    Args:
        filename: string representing the molecule geometry (see
            `notebooks/Guide_to_data`), without any extension or directory.
        initial_params: initial UCC optimization params. If None (default),
            parameters are randomly extracted.

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
    ucc_ferop = generate_ucc_operator(singles, doubles)

    simulator = cirq.Simulator()
    parameter_dict = {}
    circuit = cirq.Circuit()
    circuit.append(initialize_hf_state(molecule.n_electrons))

    for i, op in enumerate(ucc_ferop):
        parameter_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))

    if initial_params is None:
        rng = numpy.random.default_rng()
        initial_params = rng.uniform(-numpy.pi, numpy.pi, len(parameter_dict))

    stopwatch = time.time()
    print('Starting optimization.')
    result = scipy.optimize.minimize(
        overlap_with_circuit_state,
        x0=(initial_params),
        args=(circuit, parameter_dict, ground_states, simulator, False),
        options={'maxiter': 2000},
        method='COBYLA')
    print(result)
    print('Optimization time: {} minutes \n'
          .format((time.time() - stopwatch) / 60))

    optimized_energy = expectation_value_with_circuit_state(
        result['x'], circuit, parameter_dict,
        qubit_hamiltonian, simulator)

    return dict(
        params=result['x'],
        overlap=result['fun'],
        energy_expval=optimized_energy,
        energy_error=optimized_energy - data_dict['exact_energy'],
        optimizer_success=result['success'],
        optimizer_nfev=result['nfev']
    )


# ***  Main script code  ***

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
    print(type(ucc_dict['optimizer_success']))

    print('saving data to file.')
    with open(UCC_DIR + filename + '.json', 'wt') as f:
        json.dump(ucc_dict, f, default=encode_complex_and_array)

print(*((k, v) for k, v in ucc_dict.items()), sep='\n')
