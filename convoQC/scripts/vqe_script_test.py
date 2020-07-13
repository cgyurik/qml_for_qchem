"""Test vqe script functions."""

import pytest
import numpy

import openfermion
import cirq
from .vqe_script import (
    get_molecule_data,
    singlet_hf_generator,
    parse_arguments)


def test_get_molecule():
    """Test correct molecule."""
    molecule = get_molecule_data()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(
            molecule.get_molecular_hamiltonian()))

    spectrum = openfermion.eigenspectrum(qubit_hamiltonian)
    numpy.testing.assert_approx_equal(molecule.fci_energy,
                                      spectrum[0], 7)

    assert molecule.hf_energy > spectrum[0]


@pytest.mark.parametrize('n_electrons', [2, 4, 6])
def test_initialize_hf(n_electrons):
    """Test correct HF state circuit."""
    circuit = cirq.Circuit(singlet_hf_generator(n_electrons))

    assert len(circuit.all_qubits()) == n_electrons

    list_of_xgates = []
    for operation in circuit.findall_operations_with_gate_type(
            cirq.XPowGate):
        list_of_xgates.append(operation[-1])

    assert len(list_of_xgates) == n_electrons


def test_parse_arguments():
    """Test parsing arguments."""
    arguments = parse_arguments(['--n_electrons=4', '--n_orbitals=6'])

    assert len(arguments) == 2
