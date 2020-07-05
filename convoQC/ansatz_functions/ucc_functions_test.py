"""Test for UCC functions."""

import scipy
import pytest

import cirq
import openfermion

from .ucc_functions import (
    generate_ucc_amplitudes,
    generate_ucc_amplitudes_spin_conserved,
    generate_ucc_operator,
    generate_circuit_from_pauli_string)


def test_raises_ucc():
    """Test raise errors in ucc amplitudes."""
    with pytest.raises(TypeError):
        generate_ucc_amplitudes([10], 2)
        generate_ucc_amplitudes_spin_conserved([10], 2)
    with pytest.raises(TypeError):
        generate_ucc_amplitudes(10, [-3])
        generate_ucc_amplitudes_spin_conserved(10, [-3])
    with pytest.raises(ValueError):
        generate_ucc_amplitudes(10, -3)
        generate_ucc_amplitudes_spin_conserved(10, -3)


@pytest.mark.parametrize('n_elec, n_orbs',
                         [(2, 4), (2, 8), (4, 8)])
def test_correct_number_amplitudes(n_elec, n_orbs):
    """Test correct number of amplitudes generated."""
    num_singles = n_elec * (n_orbs - n_elec)
    num_doubles = (scipy.special.comb(n_elec, 2) *
                   scipy.special.comb((n_orbs - n_elec), 2))
    singles, doubles = generate_ucc_amplitudes(n_elec, n_orbs)
    (singles_spin,
     doubles_spin) = generate_ucc_amplitudes_spin_conserved(n_elec, n_orbs)

    assert len(singles) == num_singles
    assert len(doubles) == num_doubles
    assert len(singles_spin) == int(num_singles / 2)
    assert len(doubles) >= len(doubles_spin)


def test_generate_ucc_operator():
    """Test generate ucc operator function."""
    singles, doubles = generate_ucc_amplitudes(2, 4)
    ucc_ops_list = generate_ucc_operator(singles, doubles)

    assert len(ucc_ops_list[3].terms) == 2
    assert len(ucc_ops_list) == (len(singles) + len(doubles))


def test_generate_circuit_raises():
    """Test raising errors in circuit generator."""
    with pytest.raises(TypeError):
        cirq.Circuit(generate_circuit_from_pauli_string(0, 'theta_0'))
    with pytest.raises(TypeError):
        cirq.Circuit(generate_circuit_from_pauli_string(
            openfermion.FermionOperator('4^ 0 3^ 1', 1.0),
            10))


def test_correct_number_moment():
    """Test circuit has correct number of moments."""
    singles, doubles = generate_ucc_amplitudes_spin_conserved(2, 4)
    ucc_ops_list = generate_ucc_operator(singles, doubles)

    circuit_single = cirq.Circuit(
        generate_circuit_from_pauli_string(ucc_ops_list[0],
                                           'theta_single'))
    circuit_double = cirq.Circuit(
        generate_circuit_from_pauli_string(ucc_ops_list[-1],
                                           'theta_double'))

    assert len(circuit_single.moments) == 2
    assert len(circuit_double.moments) == 8
