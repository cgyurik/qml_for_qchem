"""Functions to generate UCC amplitudes and operators."""
## general tools
import numpy
import sympy
import openfermion
import cirq

def qubit_operator_to_pauli_string(qubit_op, qubits, parameter=None):
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
    if not isinstance(qubit_op, openfermion.QubitOperator):
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

                pauli_string *= op(qubits[ind])

    return pauli_string * parameter

def generate_ucc_amplitudes(n_electrons, n_spin_orbitals):
    """
    Create lists of amplidues to generate UCC operators.
    This function does not enforce spin-conservation in the
    excitation operators.
    Args:
        n_electrons (int): Number of electrons.
        n_spin_orbitals (int): Number of spin-orbitals,
            equivalent to number of qubits.
    Returns:
        single_amplitudes, double_amplitudes (list): List of single
            and double amplitudes as [[i,j], t_ij, [k,l], t_kl]
    Raises:
        TypeError: if n_electrons or n_spin_orbitals is not integer or float.
        ValueError: if n_electrons is greater than n_spin_orbitals.
    Notes:
        Assigns value 1 to each amplitude.
    """
    if isinstance(n_electrons, (int, float)):
        n_electrons = int(n_electrons)
    else:
        raise TypeError('Electrons must be a number.')
    if isinstance(n_spin_orbitals, (int, float)):
        n_spin_orbitals = int(n_spin_orbitals)
    else:
        raise TypeError('Orbitals must be a number.')
    if n_electrons > n_spin_orbitals:
        raise ValueError(
            'Number of electrons can not be greater than orbitals.')

    single_amplitudes = []
    double_amplitudes = []

    for one_el in range(0, n_electrons):
        for unocc_orb in range(n_electrons, n_spin_orbitals):
            single_amplitudes.append([[unocc_orb, one_el], 1.0])

            for two_el in range(one_el, n_electrons):
                for two_unocc_orb in range(unocc_orb, n_spin_orbitals):
                    if ((two_unocc_orb != unocc_orb) and (two_el != one_el)):
                        double_amplitudes.append(
                            [[two_unocc_orb, two_el, unocc_orb, one_el], 1.0])

    return single_amplitudes, double_amplitudes

def generate_ucc_operators(single_amplitudes, double_amplitudes):
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
        It uses the default openfermion UCC generator but for each amplitude
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

def generate_circuit_from_pauli_string(operator, parameter, qubits, transformation=openfermion.jordan_wigner):
    """
    Create a cirq.Circuit object from the operator.
    This function uses PauliString and PauliStringPhasor objects
    to generate a cirq.Circuit from the operator.
    Makes a circuit with a parametrized gate named after the input
    parameter_name.
    Args:
        operator (FermionOperator): Fermionic operator to translate
            to a circuit.
        parameter (sympy.Symbol): Name to use for the sympy parameter.
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
    if not isinstance(parameter, sympy.Symbol):
        raise TypeError('Parameter name must be a sympy.Symbol.')

    qubit_op = transformation(operator)

    for op, val in qubit_op.terms.items():
        pauli_string = qubit_operator_to_pauli_string(openfermion.QubitOperator(op, numpy.sign(val)), qubits)
        yield cirq.PauliStringPhasor(pauli_string,
                                     exponent_neg=parameter)

def singlet_hf_generator(n_electrons, n_orbitals, qubits):
    """
    Add X gate to qubits 0 to n_electrons.
    """
    for i in range(n_electrons):
        yield cirq.X(qubits[i])
    for i in range(n_electrons, 2 * n_orbitals):
        yield cirq.I(qubits[i])

