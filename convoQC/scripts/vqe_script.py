"""Script to perform a VQE with H4 family molecules."""

import sys
import os
from typing import Generator
import argparse
import numpy
from scipy.optimize import minimize, OptimizeResult

import openfermion
import cirq

#  Clash between local import and external one.
#  This might be easy to solve but not sure how.
try:
    from convoQC.ansatz_functions.ucc_functions import (
        generate_ucc_amplitudes,
        generate_circuit_from_pauli_string,
        generate_ucc_operator)
    from convoQC.vqe_functions.vqe_optimize_functions import (
        overlap_with_circuit_state)
except Exception:
    sys.path.insert(0, os.getcwd() + '/../')
    from ansatz_functions.ucc_functions import (  # type: ignore
        generate_ucc_amplitudes,
        generate_circuit_from_pauli_string,
        generate_ucc_operator)
    from vqe_functions.vqe_optimize_functions import (  # type: ignore
        overlap_with_circuit_state,
        expectation_value_with_circuit_state)


def get_molecule_data():
    """Obtain molecule data."""
    #  Xavi: this function must return the
    #  necessary information of the molecule
    geometry = [('H', (0., 0., 0.)),
                ('H', (0., 0., 0.7414))]
    basis = 'sto-3g'
    multiplicity = 1
    description = format(0.7414)

    molecule = openfermion.MolecularData(
        geometry,
        basis,
        multiplicity,
        description=description)

    molecule.load()

    return molecule


def initialize_hf_state(n_electrons: int) -> Generator:
    """Add X gate to qubits 0 to n_electrons."""
    yield cirq.X.on_each(cirq.LineQubit.range(n_electrons))


def parse_arguments(args):
    """Helper function to parse arguments to main."""
    parser = argparse.ArgumentParser('QCNN-VQE state preparation.')

    parser.add_argument('--n_electrons',
                        type=int,
                        help='Number of electrons.'
                        )
    parser.add_argument('--n_orbitals',
                        type=int,
                        help='Number of orbitals.'
                        )
    return vars(parser.parse_args(args))


def main(*, n_electrons: int, n_orbitals: int) -> OptimizeResult:
    """Run script."""
    #  Here it should import the data from molecules.
    molecule = get_molecule_data()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(
            molecule.get_molecular_hamiltonian()))
    #  pylint: disable = unused-variable
    eigvals, eigvecs = numpy.linalg.eigh(
        openfermion.qubit_operator_sparse(qubit_hamiltonian).toarray())
    # pylint: enable = unused-variable
    ground_state = eigvecs[:, 0]

    singles, doubles = generate_ucc_amplitudes(
        n_electrons, n_orbitals)
    ucc_ferop = generate_ucc_operator(singles, doubles)

    simulator = cirq.Simulator()
    parameter_dict = {}
    circuit = cirq.Circuit()
    circuit.append(initialize_hf_state(n_electrons))
    for i, op in enumerate(ucc_ferop):
        parameter_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))
    start_parameters = numpy.random.uniform(-numpy.pi, numpy.pi,
                                            len(parameter_dict))

    result = minimize(
        overlap_with_circuit_state,
        x0=(start_parameters),
        args=(circuit, parameter_dict, ground_state, simulator, False),
        options={'maxiter': 2000},
        method='COBYLA')

    optimized_energy = expectation_value_with_circuit_state(
        result['x'], circuit, parameter_dict,
        qubit_hamiltonian, simulator)
    return result, optimized_energy


if __name__ == '__main__':
    print('Starting optimization.')
    result, energy = main(**parse_arguments(sys.argv[1:]))
    print(result)
    print('Optimized energy {}'.format(numpy.real(energy)))
    molecule = get_molecule_data()
    print('Exact energy     {}'.format(molecule.fci_energy))
