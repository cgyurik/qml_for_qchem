"""Test functions for vqe optimization."""

import numpy
import pytest

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


def test_overlap_raises():
    """Test raises errors in overlap function."""
    molecule = get_molecule()
    mol_ham = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(mol_ham))
    pass


def test_expectation_raise():
    """Test raises errors in expeectation function."""
    molecule = get_molecule()
    mol_ham = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = openfermion.jordan_wigner(
        openfermion.get_fermion_operator(mol_ham))
    pass
