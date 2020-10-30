## os/sys tools
import os, sys
from functools import partial
# disable terminal warning tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq, sympy
import utils.pqc as pqc
import scipy, random, pickle
# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
# data loading
from utils.data_utils.load_lib import load_data, JSON_DIR
from utils.tfq_utils import *


"""
[Research]
    - 'pool-circuit' architecture.
    - prune/sparsify 'controller_nn'.
    - prune/sparsify 'postprocess_nn'.
"""

"""
[TFQ meeting]
    - Contribute H4 dataset?
    - ...
"""

"""
[TODO]    
    - Re-add serial model.
    - Reinitialize ham_qubits in groundstate when reuploading (i.e., use fewer qubits in serial model).
"""

class tfq_model():
    """
    Attributes: 
    - n_aux_qubits: number of ancilla qubits of the variational circuit.
    - var_depth: number of repetitions of single-qubit rotations & entangling layer in variational circuit.
    - n_gs_uploads: number of groundstates (i.e., quantum input) fed to the qml model.
    - intermediate_readouts: allow readouts after each reupload (i.e., parallel or serial pqcs).
    """
    def __init__(self, n_gs_uploads=2, n_aux_qubits=2, var_depth=2,
                dir_path=None, print_circuit=False, print_summary=False, plot_to_file=False,
                processed_data=None):
        ## Setting hyperparameters.
        self.n_gs_uploads = n_gs_uploads
        self.n_aux_qubits = n_aux_qubits
        self.var_depth = var_depth

        ## Initializing qubits and observables.
        self.n_ham_qubits = 8
        self.n_qubits = self.n_ham_qubits + self.n_aux_qubits     
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.readouts = [cirq.Z(bit) for bit in self.qubits]       
        
        ## Reading H4 data
        print("Loading data.")
        dataset = self.load_dataset(processed_data=processed_data)
        self.train_input = [dataset[0], dataset[1][0], dataset[1][1], dataset[1][2], dataset[1][3]] 
        self.test_input = [dataset[2], dataset[3][0], dataset[3][1], dataset[3][2], dataset[3][3]]
        self.train_labels = dataset[4]
        self.test_labels = dataset[5]
        
        ## Setting dir_path for reporting output
        self.dir_path = dir_path

        ## Initializing components of the model.    
        print("Setting up components of the model.")
        print("  - pqc.")        
        self.pqc = self.create_model_circuit(print_circuit=print_circuit)
        print("  - controller nn.")        
        self.controller_nn = self.create_controller_nn()
        print("  - postprocess nn.")  
        self.postprocess_nn = self.create_postprocess_nn()
        print("Connecting components of the model.")
        self.tfq_model = self.create_tfq_model(print_summary=print_summary, plot_to_file=plot_to_file)


    """
    Create the final circuit of the model.
    """
    def create_model_circuit(self, print_circuit=False):
        # Creating symbols.
        n_var_symbols_layer = 2 * self.n_qubits * (self.var_depth + 1)
        self.var_symbols = sympy.symbols('pqc0:' + str(n_var_symbols_layer * self.n_gs_uploads))
    
        # Creating the (parallel) model circuits
        model_circuits = []
        for i in range(self.n_gs_uploads):
            ith_model_circuit = cirq.Circuit()
            var_symbols_layer = self.var_symbols[i * n_var_symbols_layer : (i+1) * n_var_symbols_layer]
            ith_model_circuit += pqc.variational_circuit(self.qubits, var_symbols_layer,
                                                                                    depth=self.var_depth)    
            model_circuits.append(ith_model_circuit)

        ## Printing the circuit(s).
        if print_circuit:   
            # Checking if dir_path is specified, otherwise print to terminal.
            if self.dir_path is None:
                print("-----Encoding circuit-----")
                print(self.n_gs_uploads, "parallel copies of the circuit:")
                print(model_circuits[0].to_text_diagram(transpose=True))
            else:
                with open(self.dir_path + '/txt/encoding_circuit.txt', 'w') as f:
                    print(self.n_gs_uploads, "parallel copies of the circuit:", file=f)
                    print(model_circuits[0].to_text_diagram(transpose=True), file=f)
        
        return model_circuits
        
    """
    Create NN(s) that controlls parameters of encoding circuit(s).
    """
    def create_controller_nn(self):
        n_enc_params = ( 2 * (self.n_aux_qubits + self.n_ham_qubits) * (self.var_depth + 1) ) 
        controllers = []
        for i in range(self.n_gs_uploads):
            ith_controller = tf.keras.Sequential(
                                [#tfmot.sparsity.keras.prune_low_magnitude(
                                    tf.keras.layers.Dense(n_enc_params, input_shape=(7,))
                                #)
                                ],
                            name='controller_nn_' + str(i))
            controllers.append(ith_controller)
               
        return controllers

    """
    Create NN that postprocesses outcome of PQC.
    """
    def create_postprocess_nn(self):  
        # Computing the right input shape
        q_shape = (len(self.readouts) * self.n_gs_uploads, )
        c_shape = (7,)
        input_shape = tuple(map(sum, zip(q_shape, c_shape))) # (x, ) , (y, ) -> (x + y, )
        
        # Setting-up postprocess_nn
        postprocess_nn = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=input_shape)],
                                                name='postprocess_nn')
        return postprocess_nn

    """
    Create the hybrid model.
    """
    def create_tfq_model(self, print_summary=False, plot_to_file=False):
        ## Setting up input layer for the quantum input.
        quantum_input = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_input')

        ## Setting up input layer for each of the classical parameters.        
        classical_input = []
        for i in range(3):
            classical_input.append(tf.keras.Input(shape=(3,), dtype=tf.dtypes.float32, 
                                                    name='geometry_'+str(i)))
        classical_input.append(tf.keras.Input(shape=(4,), dtype=tf.dtypes.float32, 
                                                    name='orbital_energies'))

        ## Setting up seperate NN to preprocess each of the geometry triplets.
        preprocessed_geometries = []
        for i in range(3):
            preprocessed_geometries.append(
                tf.keras.layers.Dense(1, input_shape=(3,),name='geometry_nn_' + str(i))(classical_input[i])
                )
        geometries = tf.keras.layers.concatenate(preprocessed_geometries, name='processed_geometries')
        processed_classical_input = tf.keras.layers.concatenate([geometries, classical_input[3]],
                                                                      name='processed_classical_input')

        ## Setting up controller nn(s) & controlled encoding_circuits(s) and connecting them to input layer.
        preprocess_nn = [self.controller_nn[i](processed_classical_input) for i in range(self.n_gs_uploads)]
        
        ## Setting up the controlledPQCs
        pqc_layers = []
        # connecting each controller nn & quantum input to the corresponding pqc.
        for i in range(self.n_gs_uploads):
            pqc_id = 'pqc'+str(i)
            pqc_layers.append(
                tfq.layers.ControlledPQC(self.pqc[i], operators=self.readouts, 
                                            name=pqc_id)([quantum_input, preprocess_nn[i]])
                            )
        # if multiple reuploads, concatenate outcomes.
        if self.n_gs_uploads > 1:
            pqc_expectation = tf.keras.layers.concatenate(pqc_layers, name='readout_concatenate')
        else:
            pqc_expectation = pqc_layers[0]
        
        ## Connecting controlledPQC to 'postprocess NN'
        postprocess_input = tf.keras.layers.concatenate([pqc_expectation, processed_classical_input],
                                                           name='postprocess_input')
        postprocess_nn = self.postprocess_nn(postprocess_input)
            
        ## Build full keras model from the layers
        model = tf.keras.Model(inputs=[quantum_input, classical_input], outputs=postprocess_nn,
                                name="QML_model")    
    
        ## Print summary of the model.
        if print_summary:
            # Checking if dir_path is specified, otherwise print to terminal.
            if self.dir_path is None: 
                model.summary()
            else:
                with open(self.dir_path + '/txt/summary.txt', 'w') as f:      
                    fn = partial(print, file=f)
                    model.summary(print_fn=fn)
        ## Show the keras plot of the model
        if plot_to_file:
            path = self.dir_path + '/img/model.png'
            tf.keras.utils.plot_model(model, to_file=path, show_shapes=True, show_layer_names=True, dpi=70)
        
        return model

    """
    Generate H4 dataset
    """
    def load_dataset(self, processed_data=None):
        ## Loading generated dataset.
        if processed_data is not None:
            print("  - reading directly from processed dataset pickle.")
            path = './data/'+processed_data+'.p'
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
                self.classical_input_shape = dataset[1][0].shape
                return dataset[0], dataset[1], \
                    dataset[2], dataset[3], \
                    dataset[4], dataset[5]
                           
        ## Reading .json files.
        dataset = []
        print("  - reading .json files.")
        for filename in os.listdir(JSON_DIR):
            if filename.endswith('.json'):
                datapoint = load_data(filename)
                gs_circuit = tensorable_ucc_circuit(filename)
                datapoint.update({'gs_circuit' : gs_circuit})
                dataset.append(datapoint) 
                print("    * read molecule", len(dataset), ".")
                if len(dataset) >= 3:
                    break

        ## Dividing into training and test.
        random.shuffle(dataset)
        split_ind = int(len(dataset) * 0.7)
        train_data = dataset[:split_ind]
        test_data = dataset[split_ind:]
        
        ## Training data
        print("  - processing training data.")
        # Parsing classical input.
        train_geom1 = []
        train_geom2 = []
        train_geom3 = []
        train_oe = []
        print('    * loading classical training data.')
        for i in range(len(train_data)):
            # Only include molecules with groundstate degeneracy 1.
            if train_data[i]['multiplicity'] == 3:
                continue

            # Reading data from dict.
            geometry = np.transpose(np.array([train_data[i]['geometry'][j][1] for j in range(1, 4)]))
            canonical_orbitals = np.array(train_data[i]['canonical_orbitals'])
            orbital_energies = np.array(train_data[i]['orbital_energies'])
            canonical_to_oao = np.array(train_data[i]['canonical_to_oao'])  
            
            # Puting geometry & orbital energies in classical input
            train_geom1.append(geometry[0])
            train_geom2.append(geometry[1])
            train_geom3.append(geometry[2])
            train_oe.append(orbital_energies)
        train_classical_inputs = [np.array(train_geom1), np.array(train_geom2), np.array(train_geom3),
                                    np.array(train_oe)]

        # Parsing quantum input.
        train_gs_circuits = []            
        for i in range(len(train_data)):
            print('    * loading training circuit', i+1, '/', len(train_data))            

	        # Only include molecules with groundstate degeneracy 1.
            if train_data[i]['multiplicity'] == 3:
                continue

            # Reading circuit.
            train_gs_circuit = cirq.Circuit()
            for op in train_data[i]['gs_circuit'].all_operations():
                if len(op.qubits) == 1:
                    qubit_id = op.qubits[0].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        train_gs_circuit += op.with_qubits(self.qubits[qubit_id])
                    # else, apply on qubit corresponding to upload. 
                    else:
                        for i in range(self.n_gs_uploads):
                            train_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id])
                elif len(op.qubits) == 2:
                    qubit_id0 = op.qubits[0].col
                    qubit_id1 = op.qubits[1].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        train_gs_circuit += op.with_qubits(self.qubits[qubit_id0], self.qubits[qubit_id1])
                    # else, apply on qubit corresponding to upload.
                    else:
                        for i in range(self.n_gs_uploads):
                            train_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id0],
                                                                self.qubits[i*self.n_ham_qubits + qubit_id1])
                else:
                    print("Encountered >=3-qubit gate, error!")
                    exit()
            train_gs_circuits.append(train_gs_circuit)

        ## Testing data
        print("  - processing validation data.")
        # Parsing classical input.
        test_classical_inputs = []
        test_geom1 = []
        test_geom2 = []
        test_geom3 = []
        test_oe = []
        print('    * loading classical validation data.')
        for i in range(len(test_data)):
            # Only include molecules with groundstate degeneracy 1.
            if test_data[i]['multiplicity'] == 3:
                continue

            # Reading data from dict.            
            geometry = np.transpose(np.array([test_data[i]['geometry'][j][1] for j in range(1, 4)]))
            canonical_orbitals = np.array(test_data[i]['canonical_orbitals'])
            orbital_energies = np.array(test_data[i]['orbital_energies'])
            canonical_to_oao = np.array(test_data[i]['canonical_to_oao'])    
            
            # Puting geometry & orbital energies in classical input
            test_geom1.append(geometry[0])
            test_geom2.append(geometry[1])
            test_geom3.append(geometry[2])
            test_oe.append(orbital_energies)        
        test_classical_inputs = [np.array(test_geom1), np.array(test_geom2), np.array(test_geom3), 
                                    np.array(test_oe)]
           
        # Parsing quantum input.
        test_gs_circuits = []
        for i in range(len(test_data)):
            print('    * loading validation circuit', i+1, '/', len(test_data))

            # Only include molecules with groundstate degeneracy 1.
            if test_data[i]['multiplicity'] == 3:
                continue
            
            # Reading circuit.
            test_gs_circuit = cirq.Circuit()
            for op in test_data[i]['gs_circuit'].all_operations():
                if len(op.qubits) == 1:
                    qubit_id = op.qubits[0].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        test_gs_circuit += op.with_qubits(self.qubits[qubit_id])
                    # else, apply on qubit corresponding to upload.
                    else:
                        for i in range(self.n_gs_uploads):
                            test_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id])
                elif len(op.qubits) == 2:
                    qubit_id0 = op.qubits[0].col
                    qubit_id1 = op.qubits[1].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        test_gs_circuit += op.with_qubits(self.qubits[qubit_id0], self.qubits[qubit_id1])
                    # else, apply on qubit corresponding to upload.
                    else:
                        for i in range(self.n_gs_uploads):
                            test_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id0],
                                                                self.qubits[i*self.n_ham_qubits + qubit_id1])
                else:
                    print("Encountered >=3-qubit gate, error!")
                    exit()
            test_gs_circuits.append(test_gs_circuit)

        # Parsing labels.
        train_labels = [train_data[j]['exact_energy'] 
                            for j in range(len(train_data)) if train_data[j]['multiplicity'] == 1]
        test_labels = [test_data[j]['exact_energy'] 
                            for j in range(len(test_data)) if test_data[j]['multiplicity'] == 1]

        # Converting to tensor.
        print('  - converting circuits to tensors.')
        train_gs_tensors = tfq.convert_to_tensor(train_gs_circuits)
        test_gs_tensors = tfq.convert_to_tensor(test_gs_circuits)

        # Pickling for next time.
        processed_dataset = [train_gs_tensors, train_classical_inputs, test_gs_tensors,
                                test_classical_inputs, np.array(train_labels), np.array(test_labels)]
        
        # Pickling the dict.
        print("Creating pickle of processed data for quicker loading.")
        data_id = 'H4_processed_'
        pickle_path = "./data/" + data_id + '.p' 
        with open(pickle_path, 'wb') as f:      
            pickle.dump(processed_dataset, f)    

        return [train_gs_tensors, train_classical_inputs, \
                test_gs_tensors, test_classical_inputs, \
                np.array(train_labels), np.array(test_labels)]

