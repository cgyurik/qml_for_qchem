import numpy as np
import cirq, sympy

"""
Make a Cirq circuit enacting a rotation of the bloch sphere about 
the X, Y and Z axis, that depends on the values in `symbols`.
"""
def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


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
Make a Cirq circuit to do a parameterized 'pooling' operation, which
attempts to reduce entanglement down from two qubits to just one.
* uses a fixed number of symbols (i.e., 6). 
"""
def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append( sink_basis_selector )
    pool_circuit.append( source_basis_selector )
    pool_circuit.append( cirq.CNOT(control=source_qubit, target=sink_qubit) )
    pool_circuit.append( sink_basis_selector ** (-1) )
    return pool_circuit

"""
A layer that specifies a quantum pooling operation.
Tries to learn to pool relevant information from two qubits onto 1.
"""    
def pool_circuit(source_bits, sink_bits, symbols):
    circuit = cirq.Circuit()
    i = 0
    for source, sink in zip(source_bits, sink_bits):
        two_qubit_symbols = symbols[i * 6 : (i + 1) * 6]
        i += 1
        circuit += two_qubit_pool(source, sink, two_qubit_symbols)
    return circuit
