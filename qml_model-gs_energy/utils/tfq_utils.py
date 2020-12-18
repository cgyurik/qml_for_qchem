"""Utility functions related to tensorflow_quantum."""
import numpy as np
from openfermion import transforms, ops
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq  # type: ignore
from utils import load_ucc_data, load_data
from utils.ansatz_functions import (generate_ucc_amplitudes,
                                generate_ucc_operators)
from utils.ansatz_functions.qubitoperator_to_paulistring_translator \
    import qubitoperator_to_pauli_string
from scripts.optimize_ucc import (singlet_hf_generator,
                                    triplet_hf_generator)

def tensorable_ucc_circuit(filename: str) -> cirq.Circuit:
    '''construct a UCC circuit compatible with TFQ, from data files'''
    params = load_ucc_data(filename)['params']
    multiplicity = load_data(filename)['multiplicity']

    # create UCC operators and amplitudes
    ucc_amplitudes = generate_ucc_amplitudes(4, 8)
    ucc_operators = generate_ucc_operators(*ucc_amplitudes)

    # generate a list of pauli strings with the correct coefficients
    pauli_str_list = []
    for (fop, param) in zip(ucc_operators, params):
        qop = transforms.jordan_wigner(fop)
        for ps, val in qop.terms.items():
            pauli_str_list.append(
                qubitoperator_to_pauli_string(
                    ops.QubitOperator(ps, np.pi / 2 * param * np.sign(val))))

    # HF state preparation
    c = cirq.Circuit(singlet_hf_generator(4, 4)
                     if multiplicity == 1 else
                     triplet_hf_generator(4, 4))

    # append tfq-compatible circuit from pauli strings
    c.append(tfq.util.exponential(pauli_str_list))

    return c
