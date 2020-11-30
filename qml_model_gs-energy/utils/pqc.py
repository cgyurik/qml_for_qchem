import numpy as np
import cirq, sympy, openfermion, openfermioncirq

"""
Make a Cirq circuit for the Given layer/ansatz.
"""
def parametrized_givens_ansatz(qubits, thetas, spin_basis=False):
    if spin_basis:
        N = len(qubits) // 2
        if len(thetas) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')
        def gen():
            for i, theta in zip(
                (list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N,
                thetas):
                for sigma in [0, 1]:
                    yield openfermioncirq.Ryxxy(-theta).on(qubits[2*i+sigma], 
                                           qubits[2*i+2+sigma])
                 # TODO: this is not a FSWAP under JW.
    else:
        N = len(qubits)
        if len(thetas) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')
        def gen():
            for i, theta in zip(
                (list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N,
                thetas):
                yield openfermioncirq.Ryxxy(-theta).on(qubits[i], qubits[i+1])
    return gen()
  
"""
Make a Cirq circuit for spin conserving Given layer/ansatz.
[TODO] Make it work with symbols to integrate with TFQ.
"""  
def parametrized_spin_conserving_givens_ansatz(qubits, thetas, spinless=False):
    if spinless:
        N = len(qubits)
        if len(thetas) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')

        matr = np.eye(N)
        for i, theta in zip(
            (list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N,
            thetas):
            givens = np.eye(N)
            givens[i:i+2, i:i+2] = [[ np.cos(-theta), np.sin(-theta)], 
                                    [-np.sin(-theta), np.cos(-theta)]]
            matr = givens @ matr
        #print(matr.round(3)) # TESTING
    else:
        N = len(qubits) // 2 # number of spatial orbitals
        if len(thetas) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')

        matr = np.eye(2*N)
        for i, theta in zip(
            (list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N,
            thetas):
            givens = np.eye(2*N)
            for sigma in [0, 1]:
                i0 = 2*i+sigma
                i1 = 2*(i+1)+sigma
                givens[i0, i0] = np.cos(-theta)
                givens[i0, i1] = np.sin(-theta) 
                givens[i1, i0] = -np.sin(-theta)
                givens[i1, i1] = np.cos(-theta)
            matr = givens @ matr
        #print(matr.round(3)) # TESTING
    return openfermion.optimal_givens_decomposition(qubits, matr)

"""
Layer of single qubit z-rotations
"""
def rot_z_layer(qubits, parameters):
    for i in range(len(qubits)):
        yield cirq.rz(parameters[i])(qubits[i])

"""
Layer of single qubit y-rotations
"""
def rot_y_layer(qubits, parameters):
    for i in range(len(qubits)):
        yield cirq.ry(parameters[i])(qubits[i])

"""
Layer of entangling CZ(i,i+1 % n_qubits) gates.
"""
def entangling_layer_linear(qubits):
    if len(qubits) == 2:
        yield cirq.CZ(qubits[0], qubits[1])
        return
    for i in range(len(qubits)):
        yield cirq.CZ(qubits[i], qubits[(i+1) % len(qubits)])

"""    
Construct variational part of the PQC.
"""
def variational_circuit(qubits, symbols, depth=2):
    if len(symbols) != 2 * len(qubits) * (depth + 1):
        raise ValueError("Symbols should be of dimension 2 * n_qubits * (var_depth + 1).")
    for d in range(depth):
        # Layer of single qubit z/y-rotations
        yield rot_z_layer(qubits, symbols[d * 2 * len(qubits)
                                        : (d+1) * 2 * len(qubits) : 2])
        yield rot_y_layer(qubits, symbols[d * 2 * len(qubits) + 1
                                              : (d+1) * 2 * len(qubits) + 1 : 2])
        yield entangling_layer_linear(qubits)
  
    # Final layer of single qubit z/y-rotations
    yield rot_z_layer(qubits, symbols[depth * 2 * len(qubits)
                                           : (depth+1) * 2 * len(qubits) : 2])
    yield rot_y_layer(qubits, symbols[depth * 2 * len(qubits) + 1
                                            : (depth+1) * 2 * len(qubits) + 1 : 2])

"""
Make a Cirq circuit enacting a rotation of the bloch sphere about 
the X, Y and Z axis, that depends on the values in `symbols`.

def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])

------

Make a Cirq circuit to do a parameterized 'pooling' operation, which
attempts to reduce entanglement down from two qubits to just one.
* uses a fixed number of symbols (i.e., 6). 

def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append( sink_basis_selector )
    pool_circuit.append( source_basis_selector )
    pool_circuit.append( cirq.CNOT(control=source_qubit, target=sink_qubit) )
    pool_circuit.append( sink_basis_selector ** (-1) )
    return pool_circuit

------
 
A layer that specifies a quantum pooling operation.
Tries to learn to pool relevant information from two qubits onto 1.

def pool_circuit(source_bits, sink_bits, symbols):
    circuit = cirq.Circuit()
    i = 0
    for source, sink in zip(source_bits, sink_bits):
        two_qubit_symbols = symbols[i * 6 : (i + 1) * 6]
        i += 1
        circuit += two_qubit_pool(source, sink, two_qubit_symbols)
    return circuit
"""
