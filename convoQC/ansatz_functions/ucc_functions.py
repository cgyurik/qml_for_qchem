"""Functions to generate UCC amplitudes and operators."""

from typing import Union, Sequence, Generator, List, Any
import numpy

import sympy
import openfermion
import cirq
from openfermioncirq.utils.qubitoperator_translator import (  # type: ignore
    _qubitoperator_to_pauli_string)


def generate_ucc_amplitudes(n_electrons: Union[int, float],
                            n_orbitals: Union[int, float],) -> Sequence:
    """
    Create lists of amplidues to generate UCC operators.

    This function does not enforce spin-conservation in the
    excitation operators.

    Args:
        n_electrons (int): Number of electrons.
        n_orbitals (int): Number of spin-orbitals,
            equivalent to number of qubits.

    Returns:
        single_amplitudes, double_amplitudes (list): List of single
            and double amplitudes as [[i,j], t_ij, [k,l], t_kl]

    Raises:
        TypeError: if n_electrons or n_orbitals is not integer or float.
        ValueError: if n_electrons is greater than n_orbitals.

    Notes:
        Assigns value 1 to each amplitude.
    """
    if isinstance(n_electrons, (int, float)):
        n_electrons = int(n_electrons)
    else:
        raise TypeError('Electrons must be a number.')
    if isinstance(n_orbitals, (int, float)):
        n_orbitals = int(n_orbitals)
    else:
        raise TypeError('Orbitals must be a number.')
    if n_electrons > n_orbitals:
        raise ValueError(
            'Number of electrons can not be greater than orbitals.')

    single_amplitudes = []
    double_amplitudes = []

    for one_el in range(0, n_electrons):
        for unocc_orb in range(n_electrons, n_orbitals):
            single_amplitudes.append([[unocc_orb, one_el], 1.0])

            for two_el in range(one_el, n_electrons):
                for two_unocc_orb in range(unocc_orb, n_orbitals):
                    if ((two_unocc_orb != unocc_orb) and
                            (two_el != one_el)):
                        double_amplitudes.append(
                            [[two_unocc_orb, two_el, unocc_orb, one_el], 1.0])

    return single_amplitudes, double_amplitudes


def generate_ucc_amplitudes_spin_conserved(
        n_electrons: Union[int, float],
        n_orbitals: Union[int, float]) -> Sequence:
    """
    Create list of amplitudes to generate UCC operators.

    Currently this function only allows for even n_electrons,
    and n_orbitals!

    Args:
        n_electrons (int): Number of electrons.
        n_orbitals (int): Number of spin-orbitals,
             equivalent to number of qubits.

    Returns:
        single_amplitudes, double_amplitudes (list): List of single
            and double amplitudes as [[i,j], t_ij, [k,l], t_kl]

    Raises:
        TypeError: if n_electrons or n_orbitals is not integer or float.
        ValueError: if n_electrons is greater than n_orbitals.

    Notes:
        Assigns value 1 to each amplitude.
    """
    if isinstance(n_electrons, (int, float)):
        n_electrons = int(n_electrons)
    else:
        raise TypeError('Electrons must be a number.')
    if isinstance(n_orbitals, (int, float)):
        n_orbitals = int(n_orbitals)
    else:
        raise TypeError('Orbitals must be a number.')
    if n_electrons > n_orbitals:
        raise ValueError(
            'Number of electrons can not be greater than orbitals.')

    occ_orbs_even = range(0, n_electrons, 2)
    unocc_orbs_even = range(n_electrons, n_orbitals, 2)
    occ_orbs_odd = range(1, n_electrons + 1, 2)
    unocc_orbs_odd = range(n_electrons + 1, n_orbitals + 1, 2)

    single_amplitudes = []
    double_amplitudes = []

    even_singles = numpy.array(numpy.meshgrid(numpy.array(occ_orbs_even),
                                              numpy.array(unocc_orbs_even))).T
    odd_singles = numpy.array(numpy.meshgrid(numpy.array(occ_orbs_odd),
                                             numpy.array(unocc_orbs_odd))).T

    even_singles = even_singles.reshape(-1, 2)
    odd_singles = odd_singles.reshape(-1, 2)

    for e, o in zip(even_singles.tolist(), odd_singles.tolist()):
        single_amplitudes.append([e, 1.0])
        single_amplitudes.append([o, 1.0])

    for e in even_singles.tolist():
        for o in odd_singles.tolist():
            double_amplitudes.append([e + o, 1.0])

    for i in range(int(len(even_singles) / 2)):
        for j in range(i + int(len(even_singles) / 2) + 1, len(even_singles)):
            double_amplitudes.append([list(even_singles[i]) +
                                      list(even_singles[j]), 1.0])
            double_amplitudes.append([list(odd_singles[i]) +
                                      list(odd_singles[j]), 1.0])

    return single_amplitudes, double_amplitudes


def generate_ucc_operator(
        single_amplitudes: Sequence,
        double_amplitudes: Sequence) -> List[Any]:
    """
    Create UCC Fermionic Operator from amplitude list.

    Args:
        single_amplitudes (List): List of Fermion interaction and amplitude
            for single exctitations of the form [[i,j], t_ij]
        double_amplitudes (List): List of Fermionic interaction and amplitudes
            for double excitations of the form [[i,j,k,l], t_ijkl]

    Return:
        ucc_fermion_ops_list (List): List of Fermionic Operators for single
            and doubles.

    Raises:
        TypeError if incorrect formatting of the single or double amplitudes
            list.

    Notes:
        This function returns a list of FermionOperator each one corresponding
        to aan amplitude of the single and double amplitudes list.
        It uses the default openfermion UCC generetor but for each amplitude
        individually.
    """
    ucc_fermion_ops_list = []
    for amplitude in single_amplitudes:
        ucc_fermion_ops_list.append(openfermion.normal_ordered(
            openfermion.uccsd_generator([amplitude], [])))
    for amplitude in double_amplitudes:
        ucc_fermion_ops_list.append(openfermion.normal_ordered(
            openfermion.uccsd_generator([], [amplitude])))

    return ucc_fermion_ops_list


def generate_circuit_from_pauli_string(
        operator: openfermion.FermionOperator,
        parameter_name: str,
        transformation=openfermion.jordan_wigner) -> Generator:
    """
    Create a cirq.Circuit object from the operator.

    This function uses PauliString and PauliStringPhasor objects
    to generate a cirq.Circuit from the operator.
    Makes a circuit with a parametrized gate named after the input
    parameter_name.

    Args:
        operator (FermionOperator): Fermionic operator to translate
            to a circuit.
        parameter_name (str): Name to use for the sympy parameter.
        transformation: Optional fermion to qubit transformation.
            It uses Jordan-Wigner by default.

    Yields:
        cirq.Circuit objects from PauliStringPhasor.

    Raises:
        TypeError: if operator is not a FermionOperator.

    Notes:
        If this function is used to generate a concatenation of circuits
        be sure that the parameter name is the same or different.
    """
    if not isinstance(operator, openfermion.FermionOperator):
        raise TypeError('Operator must be a FermionOperator object.')
    if not isinstance(parameter_name, str):
        raise TypeError('Parameter name must be a string.')

    qubit_op = transformation(operator)

    for op, val in qubit_op.terms.items():
        pauli_string = _qubitoperator_to_pauli_string(
            openfermion.QubitOperator(op, numpy.sign(val)))
        yield cirq.Circuit(
            cirq.PauliStringPhasor(pauli_string,
                                   exponent_neg=sympy.Symbol(parameter_name)))
