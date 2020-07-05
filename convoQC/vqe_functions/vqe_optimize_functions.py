"""Functions to run a VQE."""
from typing import Dict, Union, Sequence, Optional

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


def overlap_with_circuit_state(parameters: Sequence,
                               circuit: cirq.Circuit,
                               parameters_dict: Dict,
                               state: Sequence,
                               simulator: cirq.Simulator,
                               overlap_bool: Optional[bool] = True) -> float:
    """
    Compute state overlap.

    Calculate state overlap between state and circuit state.

    Args:
        parameters (List): parameterst to update the circuit.
        circuit (cirq.Circuit): Circuit object to prepare state.
        parameters_dict (Dict): Dictionary of parameters.
        state (Array): Array representating the state.
        overlap_bool (Boolean): Wheter to return the overlap
            or 1-overlap

    Return:
        A float if the overlap or 1-overlap.

    """
    if len(parameters) != len(parameters_dict):
        raise ValueError('Number of parameters and dictionary do not match.')

    for val, key in zip(parameters, parameters_dict.keys()):
        parameters_dict[key] = val

    simulated_circuit = simulator.simulate(
        cirq.resolve_parameters(circuit,
                                cirq.ParamResolver(parameters_dict)))

    overlap = numpy.abs(
        numpy.dot(simulated_circuit.final_state.conj(),  # type: ignore
                  state))
    if overlap:
        return overlap
    else:
        return (1.0 - overlap)


def expectation_value_with_circuit_state(circuit: cirq.Circuit,
                                         parameters: Sequence,
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
