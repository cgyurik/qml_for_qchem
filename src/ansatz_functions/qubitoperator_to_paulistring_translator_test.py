"""Test functions QubitOperator to PauliString translator."""

import numpy
import openfermion
import cirq
from openfermion import QubitOperator

import pytest

from .qubitoperator_to_paulistring_translator import (
    qubitoperator_to_pauli_string,
    qubitoperator_to_pauli_sum)


def test_function_raises():
    """Test function raises."""
    qop = QubitOperator('X0 Y1', 1.0) + QubitOperator('Z0 Z1', -0.5)
    with pytest.raises(TypeError):
        qubitoperator_to_pauli_string(1.0)
    with pytest.raises(ValueError):
        qubitoperator_to_pauli_string(qop)
    with pytest.raises(TypeError):
        qubitoperator_to_pauli_sum([5.0])


def test_identity():
    """Test correct hanlding of Identity."""
    pau_from_qop = qubitoperator_to_pauli_string(QubitOperator(' ', -0.5))
    pauli_str = cirq.PauliString() * (-0.5)

    assert pauli_str == pau_from_qop


def test_parameter_input():
    """Test input parameter."""
    qop = openfermion.QubitOperator('X0 X1 Y2 Y3', -1.0j)
    pstring1 = qubitoperator_to_pauli_string(qop)
    pstring2 = qubitoperator_to_pauli_string(qop, 0.05)

    assert pstring2.coefficient != pstring1.coefficient
    assert pstring2.coefficient == 0.05


@pytest.mark.parametrize('qubitop, state_binary',
                         [(QubitOperator('Z0 Z1', -1.0), '00'),
                          (QubitOperator('X0 Y1', 1.0), '10')])
def test_expectation_values(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.GridQubit(1, i): i for i in range(n_qubits)}

    pauli_str = qubitoperator_to_pauli_string(qubitop)
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_wavefunction(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)


@pytest.mark.parametrize(
    'qubitop, state_binary',
    [(QubitOperator('Z0 Z1 Z2 Z3', -1.0) + QubitOperator('X0 Y1 Y2 X3', 1.0),
      '1100'),
     (QubitOperator('X0 X3', -1.0) + QubitOperator('Y1 Y2', 1.0), '0000')])
def test_expectation_values_paulisum(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.GridQubit(1, i): i for i in range(n_qubits)}

    pauli_str = qubitoperator_to_pauli_sum(qubitop)
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_wavefunction(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)
