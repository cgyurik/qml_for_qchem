## os/sys tools
import os, sys
# disable terminal warning tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
## general tools
import numpy as np
import cirq
import sympy
## vqe/qml tools.
import openfermion
import tensorflow_quantum as tfq
## vqe util tools
from vqe_utils.uccsd_utils import singlet_hf_generator
from vqe_utils.uccsd_utils import generate_ucc_amplitudes
from vqe_utils.uccsd_utils import generate_ucc_operators
from vqe_utils.uccsd_utils import qubit_operator_to_pauli_string
from vqe_utils.uccsd_utils import generate_circuit_from_pauli_string


class UCCSDAnsatz():

    def __init__(self, molecule, qubits):
        self.molecule = molecule
        singles, doubles = generate_ucc_amplitudes(self.molecule.n_electrons, 2 * self.molecule.n_orbitals)
        self.ucc_ferop = generate_ucc_operators(singles, doubles)
        self.symbols = [sympy.Symbol('theta_' + str(i)) for i in range(len(self.ucc_ferop))]
        self.qubits = qubits
        self.circuit = cirq.Circuit(singlet_hf_generator(self.molecule.n_electrons, self.molecule.n_orbitals,
                                                            self.qubits))
        self.circuit += cirq.Circuit(self.operations(self.qubits), strategy=cirq.InsertStrategy.EARLIEST)

    def params(self):
        return self.symbols

    def operations(self, qubits):
        for param, op in zip(self.symbols, self.ucc_ferop):
            yield generate_circuit_from_pauli_string(op, param, qubits)


    def tensorable_ucc_circuit(self, params, qubits):
        # Generate a list of pauli strings with the correct coefficients
        pauli_str_list = []
        for (fop, param) in zip(self.ucc_ferop, params):
            qop = openfermion.transforms.jordan_wigner(fop)
            for ps, val in qop.terms.items():
                pauli_str_list.append(qubit_operator_to_pauli_string(
                                            openfermion.ops.QubitOperator(ps, np.pi/2*param*np.sign(val)),
                                                                    qubits)
                                     )
        # HF state preparation
        c = cirq.Circuit(singlet_hf_generator(self.molecule.n_electrons, self.molecule.n_orbitals, qubits))
        # Appending variational part
        c.append(tfq.util.exponential(pauli_str_list))


        return c
