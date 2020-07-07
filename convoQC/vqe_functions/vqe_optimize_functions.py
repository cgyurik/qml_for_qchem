"""Functions to run a VQE."""
from typing import Dict, Union, Sequence

import numpy

import openfermion
import cirq

Hamiltonian = Union[
    openfermion.FermionOperator,
    openfermion.QubitOperator,
    openfermion.InteractionOperator,
    openfermion.DiagonalCoulombHamiltonian,
    openfermion.PolynomialTensor,
    openfermion.BosonOperator,
    openfermion.QuadOperator]


def circuit_state_fidelity(parameters: Sequence,
                           circuit: cirq.Circuit,
                           parameters_dict: Dict,
                           target_states: numpy.ndarray,
                           simulator: cirq.Simulator) -> float:
    """
    Returns the fidelity of a state prepated by `circuit` to a target state or
    a target subspace.

    Args:
        parameters (List): parameterst to update the circuit.
        circuit (cirq.Circuit): Circuit object to prepare state.
        parameters_dict (Dict): Dictionary of parameters needed to resolve the
            circuit.
        target_states (numpy.ndarray): Array representating the target
            state(s), with which the overlap is computed. If target_states is a
            two-dimensional ndarray, the column vectors are interpreted as
            (orthonormal) basis vectors of the target subspace.
    """
    if len(parameters) != len(parameters_dict):
        raise ValueError('Number of parameters and dictionary do not match.')

    if not isinstance(target_states, numpy.ndarray):
        raise TypeError(
            '`target_states` should be a numpy.ndarray, not {}'
            .format(type(target_states)))
    if len(target_states.shape) == 1:
        target_states = numpy.reshape(target_states, (-1, 1))
    if len(target_states.shape) != 2:
        raise TypeError(
            '`target_states` should be a one- or two-dimensional np.array')

    for val, key in zip(parameters, parameters_dict.keys()):
        parameters_dict[key] = val

    simulated_circuit = simulator.simulate(
        cirq.resolve_parameters(circuit,
                                cirq.ParamResolver(parameters_dict)))

    fidelity = sum(numpy.abs(
        numpy.dot(simulated_circuit.final_state.conj(),  # type: ignore
                  state))**2 for state in target_states.T)

    return fidelity


def expectation_value_with_circuit_state(parameters: Sequence,
                                         circuit: cirq.Circuit,
                                         parameters_dict: Dict,
                                         hamiltonian: Hamiltonian,
                                         simulator: cirq.Simulator) -> float:
    """
    Compute energy expectation value without sampling.

    Args:
        circuit: Circuit object to prepare state.
        parameters: parameterst to update the circuit.
        parameters_dict: Dictionary of parameters.
        hamiltonian: OpenFermion object that can be converted to sparse matrix.

    Returns:
        expectation value of Hamiltonina without sampling.
    """
    if len(parameters) != len(parameters_dict):
        raise ValueError('Number of parameters and dictionary do not match.')
    if not isinstance(hamiltonian, openfermion.QubitOperator):
        raise TypeError('Hamiltonian must be a QubitOperator.')

    for val, key in zip(parameters, parameters_dict.keys()):
        parameters_dict[key] = val

    simulated_circuit = simulator.simulate(
        cirq.resolve_parameters(circuit,
                                cirq.ParamResolver(parameters_dict)))

    return openfermion.expectation(
        openfermion.get_sparse_operator(hamiltonian),
        simulated_circuit.final_state)  # type: ignore
