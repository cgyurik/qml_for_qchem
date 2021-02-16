import numpy as np
import cirq, sympy, openfermion, openfermioncirq

"""
Generator of a Cirq circuit for the Given layer/ansatz.
"""
def givens_generator(qubits, params, spin_basis=False):
    if spin_basis:
        N = len(qubits) // 2
        if len(params) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')
        def gen():
            for i, theta in zip((list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N, params):
                for sigma in [0, 1]:
                    yield openfermioncirq.Ryxxy(-theta).on(qubits[2*i+sigma], qubits[2*i+2+sigma])
                 # TODO: this is not a FSWAP under JW.
    else:
        N = len(qubits)
        if len(params) != (N * (N-1))/2:
            raise ValueError('Wrong number of parameters or qubits.')
        def gen():
            for i, theta in zip((list(range(0, N-1, 2)) + list(range(1, N-1, 2)))*N, params):
                yield openfermioncirq.Ryxxy(-theta).on(qubits[i], qubits[i+1])
    return gen()
 
"""
Simple wrapper to return circuit instead of generator
"""
def givens_ansatz(qubits, params, spin_basis=False):
    return cirq.Circuit(givens_generator(qubits, params, spin_basis=spin_basis))
  
'''
Generator of a Cirq circuit for the spinconserving Given layer/ansatz.
where m = n * (n - 1)`, and n = len(qubits) // 2.
'''
def spinconserving_givens_generator(qubits):
    n_qubits = len(qubits)
    if n_qubits % 2:
        raise ValueError('The number of qubits should be even')
    params = []
    j = 0
    for i in range((n_qubits*(n_qubits-1))//2):
        if i < n_qubits//2:
            params.append(0)
            continue
        p = i // (n_qubits - 1)
        q = (2 * i) % (n_qubits - 1)
        if (q < 2*p) != ((n_qubits - (q + 2)) < 2*p):
            params.append(sympy.Symbol(f'θ{j}'))
            j+=1
            continue
        params.append(np.pi/2)
    return givens_generator(qubits, params)

"""
Simple wrapper to return circuit instead of generator
"""
def spinconserving_givens_ansatz(qubits):
    return cirq.Circuit(spinconserving_givens_generator(qubits))
  

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
def hardware_efficient_generator(qubits, depth=2):
    n_symbols = 2 * len(qubits) * (depth + 1)
    symbols = sympy.symbols('pqc0:' + str(n_symbols))
    
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
Simple wrapper to return circuit instead of generator
"""
def hardware_efficient_ansatz(qubits, depth=2):
    return cirq.Circuit(hardware_efficient_generator(qubits, depth=depth))

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
