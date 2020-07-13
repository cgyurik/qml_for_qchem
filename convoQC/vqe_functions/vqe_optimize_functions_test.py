"""Test functions for vqe optimization."""

import numpy
import pytest

import cirq
import openfermion

from convoQC.ansatz_functions.ucc_functions import (
    generate_ucc_amplitudes,
    generate_ucc_operator,
    generate_circuit_from_pauli_string
)
from .vqe_optimize_functions import (
    overlap_with_circuit_state,
    expectation_value_with_circuit_state)


def get_molecule():
    """Get molecule."""
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


def test_raises():
    """Test raises errors in functions."""
    molecule = get_molecule()
    mol_ham = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(mol_ham))

    # pylint: disable = unused-variable
    eigvals, eigvecs = numpy.linalg.eigh(
        openfermion.qubit_operator_sparse(
            qubit_hamiltonian).toarray())
    # pylint: enable = unused-variable

    ground_state = eigvecs[:, 0]

    sing, doubs = generate_ucc_amplitudes(
        n_electrons=2, n_orbitals=4)
    ucc_ops = generate_ucc_operator(sing, doubs)

    simulator = cirq.Simulator()
    circuit = cirq.Circuit()
    param_dict = {}
    for i, op in enumerate(ucc_ops):
        param_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))
    circuit.insert(0, cirq.X(cirq.LineQubit(0)))
    circuit.insert(0, cirq.X(cirq.LineQubit(1)))

    start_parameters = [1.0, 0.0]

    with pytest.raises(ValueError):
        overlap_with_circuit_state(
            start_parameters, circuit, param_dict,
            ground_state, simulator)
    with pytest.raises(ValueError):
        expectation_value_with_circuit_state(
            start_parameters, circuit, param_dict,
            qubit_hamiltonian, simulator)
    new_start_params = numpy.zeros(len(param_dict))
    with pytest.raises(TypeError):
        expectation_value_with_circuit_state(
            new_start_params, circuit, param_dict,
            mol_ham, simulator)


def test_overlap_function():
    """Test for overlap circuit state."""
    molecule = get_molecule()
    mol_ham = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(mol_ham))

    # pylint: disable = unused-variable
    eigvals, eigvecs = numpy.linalg.eigh(
        openfermion.qubit_operator_sparse(
            qubit_hamiltonian).toarray())
    # pylint: enable = unused-variable

    ground_state = eigvecs[:, 0]
    hf_state = numpy.zeros(2**4)
    hf_state[12] = 1.0

    sing, doubs = generate_ucc_amplitudes(
        n_electrons=2, n_orbitals=4)
    ucc_ops = generate_ucc_operator(sing, doubs)

    simulator = cirq.Simulator()
    circuit = cirq.Circuit()
    param_dict = {}
    for i, op in enumerate(ucc_ops):
        param_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))
    circuit.insert(0, cirq.X(cirq.GridQubit(1, 0)))
    circuit.insert(0, cirq.X(cirq.GridQubit(1, 1)))

    start_parameters = numpy.zeros(len(param_dict))

    assert overlap_with_circuit_state(
        start_parameters, circuit, param_dict, hf_state,
        simulator, True) == 1.0
    assert overlap_with_circuit_state(
        start_parameters, circuit, param_dict, hf_state,
        simulator, False) == 0.0

    gs_overlap = overlap_with_circuit_state(
        start_parameters, circuit, param_dict, ground_state,
        simulator, True)
    numpy.testing.assert_approx_equal(gs_overlap, 0.99, 2)
    numpy.testing.assert_approx_equal(1 - gs_overlap, 0.006, 1)


def test_expectation_value_function():
    """Test for expectation value from circuit state."""
    molecule = get_molecule()
    mol_ham = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(mol_ham))

    hf_state = numpy.zeros(2**4)
    hf_state[12] = 1.0

    sing, doubs = generate_ucc_amplitudes(
        n_electrons=2, n_orbitals=4)
    ucc_ops = generate_ucc_operator(sing, doubs)

    simulator = cirq.Simulator()
    circuit = cirq.Circuit()
    param_dict = {}
    for i, op in enumerate(ucc_ops):
        param_dict['theta_' + str(i)] = 0.0
        circuit.append(cirq.Circuit(
            generate_circuit_from_pauli_string(
                op, parameter_name='theta_' + str(i))))
    circuit.insert(0, cirq.X(cirq.GridQubit(1, 0)))
    circuit.insert(0, cirq.X(cirq.GridQubit(1, 1)))

    start_parameters = numpy.zeros(len(param_dict))
    hf_expectation = expectation_value_with_circuit_state(
        start_parameters, circuit, param_dict, qubit_hamiltonian, simulator)

    numpy.testing.assert_approx_equal(molecule.hf_energy, hf_expectation, 7)

    start_parameters = numpy.random.uniform(-numpy.pi,
                                            numpy.pi, len(param_dict))
    expectation_value = expectation_value_with_circuit_state(
        start_parameters, circuit, param_dict, qubit_hamiltonian, simulator)
    assert expectation_value >= molecule.fci_energy
