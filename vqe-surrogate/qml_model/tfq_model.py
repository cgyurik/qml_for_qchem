## os/sys tools
import os, sys
# disable terminal warning tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
## general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq, sympy
import qml_model.qml_utils.pqc as pqc
import scipy, random, pickle
from itertools import combinations
## visualization tools
import matplotlib.pyplot as plt

"""
[Research]
    - 'pool-circuit' architecture.
    - 'controller_nn' architecture.
    - 'postprocess_nn' architecture.
"""

class tfq_model():
    """
    Attributes: 
    - n_aux_qubits: number of ancilla qubits of the variational circuit.
    - var_depth: number of repetitions of single-qubit rotations & entangling layer in variational circuit.
    - n_uploads: number of groundstates (i.e., quantum input) fed to the qml model.
    - intermediate_readouts: allow readouts after each reupload (i.e., parallel or serial pqcs).
    """
    def __init__(self, qubits, readouts=None, n_uploads=1, n_aux_qubits=0, ansatz='hwe', var_depth=1, 
                        print_circuit=False, print_summary=False, plot=False):
        ## Setting hyperparameters.
        self.n_uploads = n_uploads
        self.n_aux_qubits = n_aux_qubits
        self.var_depth=var_depth

        ## Initializing qubits and observables.
        self.n_ham_qubits = 8
        self.n_qubits = self.n_ham_qubits + self.n_aux_qubits     
        self.qubits = qubits
        
        ## Initializing readout operators.
        if readouts is None:
            # one-body measurements.
            self.readouts = [cirq.Z(i) for i in self.qubits]
            # two-body correlators.
            self.readouts += [cirq.PauliString([cirq.Z(i), cirq.Z(j)]) 
                                            for (i, j) in combinations(self.qubits,2)]
        else:
            self.readouts = readouts

        ## Initializing components of the model.    
        print("Setting up components of the model.")
        print("  - pqc.")        
        self.pqc = self.create_model_circuit(ansatz=ansatz, print_circuit=print_circuit)
        #print("  - postprocess nn.")  
        #self.postprocess_nn = self.create_postprocess_nn()
        print("Connecting components of the model.")
        self.tfq_model = self.create_tfq_model(print_summary=print_summary, plot=plot)

    """
    Create the final circuit of the model.
    """
    def create_model_circuit(self, ansatz='hwe', print_circuit=False):
        """
        # Creating the (parallel) model circuits
        model_circuits = []
        for i in range(self.n_uploads):
            if ansatz == 'hwe':
                ith_circuit = pqc.hardware_efficient_ansatz(self.qubits, depth=self.var_depth)
            elif ansatz == 'givens':
                ith_circuit = pqc.spinconserving_givens_ansatz(self.qubits)
           
            model_circuits.append(ith_circuit)

        ## Printing the circuit(s).
        if print_circuit:   
            print(model_circuits[0].to_text_diagram(transpose=True))
            
        return model_circuits
        """
        
        return pqc.hardware_efficient_ansatz(self.qubits, depth=self.var_depth)
        

    """
    Create NN that postprocesses outcome of PQC.
    """
    def create_postprocess_nn(self):  
        # Setting input_shape of expectations & classical_input of postprocess_nn.
        input_shape = (len(self.readouts) * self.n_uploads, )
        
        # Setting-up postprocess_nn
        postprocess_nn = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=input_shape)],
                                                                                    name='postprocess_nn')
        return postprocess_nn

    """
    Create the hybrid model.
    """
    def create_tfq_model(self, print_summary=False, plot=False):
        ## Setting up input layer for the quantum input.
        quantum_input = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_input')

        """
        ## Setting up each parallel pqc.
        pqc_layers = []
        for i in range(self.n_uploads):
            pqc_id = 'pqc'+str(i)
            pqc_layers.append(
                tfq.layers.PQC(self.pqc[i], operators=self.readouts, name=pqc_id)(quantum_input)
                                )
        
        ## If multiple reuploads, concatenate outcomes.
        if self.n_uploads > 1:
            pqc_expectation = tf.keras.layers.concatenate(pqc_layers, name='readout_concatenate')
        else:
            pqc_expectation = pqc_layers[0]
        
    
        ## Connecting PQC to 'postprocess NN'
        postprocess_nn = self.postprocess_nn(pqc_expectation)    
        """    
        
        self.pqc_layer = tfq.layers.PQC(self.pqc, operators=self.readouts, name="pqc")
        pqc_expectation = self.pqc_layer(quantum_input)
        
        ## Build full keras model from the layers
        # fix: Testing diagonal observable with 1 upload, normally outputs = [postprocess_nn].
        model = tf.keras.Model(inputs=quantum_input, outputs=pqc_expectation, name="surrogate_model")    
    
        ## Print summary of the model.
        if print_summary:
            model.summary()
        
        ## Show the keras plot of the model
        if plot:
            tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=70)
        
        return model

    """
    Load VQE cost function evaluations dataset (if not split into train/test yet)
    """
    def load_data(self, data, vqe_ansatz, split=0.6667):
        ## Creating state prep. circuits from params
        processed_data = []
        for i in range(len(data)):
            resolved_ansatz = vqe_ansatz.tensorable_ucc_circuit(data[i]['params'], self.qubits)
            processed_data.append({"circuit": resolved_ansatz, "energy": data[i]["energy"], 
                                                                "params": data[i]['params']})

        data = processed_data    

        ## Dividing into training and test.
        #random.shuffle(data)
        split_ind = int(len(data) * split)
        train_data = data[:split_ind]
        test_data = data[split_ind:]
        
        # Parsing labels and params.
        self.train_labels = np.array([train_data[j]['energy'] for j in range(len(train_data))])
        self.test_labels = np.array([test_data[j]['energy'] for j in range(len(test_data))])
        self.train_params = [train_data[j]['params'] for j in range(len(train_data))]
        self.test_params = [test_data[j]['params'] for j in range(len(test_data))]

        # Converting to tensor.
        print('    * converting circuits to tensors.')
        train_vqe_circuits = [train_data[j]['circuit'] for j in range(len(train_data))]
        test_vqe_circuits = [test_data[j]['circuit'] for j in range(len(test_data))]
        self.train_states = tfq.convert_to_tensor(train_vqe_circuits)
        self.test_states = tfq.convert_to_tensor(test_vqe_circuits)

        return
        
    """
    Load VQE cost function evaluations dataset (if already split in train/test)
    """
    def load_presplit_data(self, params, labels, vqe_ansatz, split=0.6667):
        ## Reading out the split.
        train_params =  params[0]
        test_params = params[1]
        train_labels = labels[0]
        test_labels = labels[1] 
        
        ## Creating state prep. circuits from params
        print('    * processing presplit data.')
        train_data = []
        for i in range(len(train_params)):
            resolved_ansatz = vqe_ansatz.tensorable_ucc_circuit(train_params[i], self.qubits)
            train_data.append({"circuit": resolved_ansatz, "energy": train_labels[i], 
                                                "params": train_params[i]})
        test_data = []
        for i in range(len(test_params)):
            resolved_ansatz = vqe_ansatz.tensorable_ucc_circuit(test_params[i], self.qubits)
            test_data.append({"circuit": resolved_ansatz, "energy": test_labels[i], 
                                                "params": test_params[i]}) 

        # Parsing labels and params.
        self.train_labels = np.array([train_data[j]['energy'] for j in range(len(train_data))])
        self.test_labels = np.array([test_data[j]['energy'] for j in range(len(test_data))])
        self.train_params = [train_data[j]['params'] for j in range(len(train_data))]
        self.test_params = [test_data[j]['params'] for j in range(len(test_data))]

        # Converting to tensor.
        print('    * converting circuits to tensors.')
        train_vqe_circuits = [train_data[j]['circuit'] for j in range(len(train_data))]
        test_vqe_circuits = [test_data[j]['circuit'] for j in range(len(test_data))]
        self.train_states = tfq.convert_to_tensor(train_vqe_circuits)
        self.test_states = tfq.convert_to_tensor(test_vqe_circuits)

        return  

