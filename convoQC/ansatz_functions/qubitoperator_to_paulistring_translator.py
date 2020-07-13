"""Functions to translate QubitOperators to PauliString."""
import cirq
from openfermion import QubitOperator


def qubitoperator_to_pauli_string(
        qubit_op: QubitOperator,
        parameter: float=None) -> cirq.PauliString:
    """
    Convert QubitOperator to Pauli String.

    Args:
        qubit_op (QubitOperator): operator to convert.

    Returns:
        pauli_string (PauliString): cirq PauliString object.

    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
        ValueError: if qubit_op has more than one Pauli string.
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')
    if len(qubit_op.terms) > 1:
        raise ValueError('Input has more than one Pauli string.')

    pauli_string = cirq.PauliString()

    if parameter is None:
        parameter = list(qubit_op.terms.values())[0]

    for ind_ops in qubit_op.terms.keys():

        if ind_ops == ():
            return pauli_string * parameter

        else:
            for ind, op in ind_ops:
                if op == 'X':
                    op = cirq.X
                elif op == 'Y':
                    op = cirq.Y
                elif op == 'Z':
                    op = cirq.Z

                pauli_string *= op(cirq.GridQubit(1, ind))

    return pauli_string * parameter


def qubitoperator_to_pauli_sum(qubit_op: QubitOperator) -> cirq.PauliSum:
    """
    Convert QubitOperator to PauliSum object.

    Args:
        qubit_op (QubitOperator): operator to convert.

    Returns:
        pauli_sum (PauliSum): cirq PauliSum object.

    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')

    pauli_sum = cirq.PauliSum()
    for pauli in qubit_op:
        pauli_sum += qubitoperator_to_pauli_string(pauli)

    return pauli_sum
