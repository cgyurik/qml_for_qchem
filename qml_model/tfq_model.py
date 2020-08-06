# disable terminal warning tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.path.abspath('..'))
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
from utils import load_data, JSON_DIR
from utils.tfq_utils import tensorable_ucc_circuit


"""
[Research]
    - 'pool-circuit' architecture.
    - 'controller_nn' architecture.
    - 'postprocess_nn' architecture.
"""

"""
[TFQ meeting]
    - Contribute H4 dataset?
    - ...
"""

"""
[TODO]    
    - Reinitialize ham_qubits in groundstate when reuploading (i.e., use fewer qubits in serial model).
"""

class tfq_model():
    """
    Attributes: 
    - n_var_qubits: number of ancilla qubits of the variational circuit.
    - var_depth: number of repetitions of single-qubit rotations & entangling layer in variational circuit.
    - n_reuploads: number of groundstates (i.e., quantum input) fed to the qml model.
    - intermediate_readouts: allow readouts after each reupload (i.e., parallel or serial pqcs).
    """
    def __init__(self, n_var_qubits=2, var_depth=2, n_reuploads=2, intermediate_readouts=False,
                  processed_data=None, print_circuit=False, print_summary=False, plot_to_file=False):
        ## Setting hyperparameters.
        self.n_var_qubits = n_var_qubits
        self.var_depth = var_depth
        self.n_reuploads = n_reuploads
        self.intermediate_readouts = intermediate_readouts       

        ## Initializing qubits and observables.
        self.n_ham_qubits = 8
        # if parallel, tfq copies qubits -> no need for different qubits per groundstate.
        if self.intermediate_readouts:
            total_n_qubits = self.n_ham_qubits + self.n_var_qubits     
        # else, initialize enough qubits to load all groundstates on.   
        else:
            total_n_qubits = ( self.n_reuploads * self.n_ham_qubits ) + self.n_var_qubits
        self.qubits = cirq.GridQubit.rect(1, total_n_qubits)
        self.readouts = [cirq.Z(bit) for bit in self.qubits[-self.n_var_qubits:]]       
        
        ## Reading H4 data
        print("Loading data.")
        self.train_groundstates, self.train_classical_inputs, \
        self.test_groundstates, self.test_classical_inputs, \
        self.train_labels, self.test_labels = self.load_dataset(processed_data=processed_data)
        
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
    Create the pqc of the model.
    """
    def create_model_circuit(self, print_circuit=False):
        ## Initializing the symbols of the circuit(s).
        n_qubits = self.n_var_qubits + self.n_ham_qubits  
        n_var_layer_symbols = 2 * n_qubits * (self.var_depth + 1)
        n_pool_layer_symbols = 6 * min(self.n_var_qubits, self.n_ham_qubits) 
        if self.intermediate_readouts: 
            var_symbols = sympy.symbols('pqc0:' + str(n_var_layer_symbols))
            pool_symbols = sympy.symbols('pool0:' + str(n_pool_layer_symbols))
            model_circuit = []
        else:
            var_symbols = sympy.symbols('pqc0:' + str( n_var_layer_symbols * self.n_reuploads ))
            pool_symbols = sympy.symbols('pool0:' + str( n_pool_layer_symbols * self.n_reuploads ))
            model_circuit = cirq.Circuit()   
          
        ## Constructing the circuit(s).
        for i in range(self.n_reuploads):
            # parallel layers.
            if self.intermediate_readouts:
                ith_pqc_layer = self.pqc_layer(self.qubits, var_symbols, pool_symbols)
                model_circuit.append(ith_pqc_layer)
            # or, serial layers.
            else:
                # qubits of the ith upload of groundstate.
                layer_qubits = self.qubits[i * self.n_ham_qubits : (i + 1) * self.n_ham_qubits]
                # ancilla qubits at the end.
                layer_qubits += self.qubits[-self.n_var_qubits:]
                layer_var_symbols = var_symbols[i * n_var_layer_symbols : (i + 1) * n_var_layer_symbols]
                layer_pool_symbols = pool_symbols[i * n_pool_layer_symbols : (i + 1) * n_pool_layer_symbols]
                ith_pqc_layer = self.pqc_layer(layer_qubits, layer_var_symbols, layer_pool_symbols)
                model_circuit += ith_pqc_layer
        
        ## Printing the circuit(s).
        if print_circuit:   
            if self.intermediate_readouts:
                print(self.n_reuploads, "parallel copies of the circuit:")
                print(model_circuits[0].to_text_diagram(transpose=True))
            else:
                print(model_circuit.to_text_diagram(transpose=True))

        return model_circuit

    """
    Construct ith layer of the pqc.
    """
    def pqc_layer(self, layer_qubits, var_symbols, pool_symbols):
        # Initialize circuit.
        pqc_layer = cirq.Circuit()
        # Append variational circuit.
        pqc_layer += pqc.variational_circuit(layer_qubits, var_symbols, depth=self.var_depth)
        # Append pool circuit.
        pqc_layer += pqc.pool_circuit(layer_qubits[:self.n_ham_qubits], layer_qubits[-self.n_var_qubits:],
                                        pool_symbols)
        return pqc_layer

    """
    Create NN(s) that controlls parameters of PQC(s).
    """
    def create_controller_nn(self):
        n_var_params = ( 2 * (self.n_var_qubits + self.n_ham_qubits) * (self.var_depth + 1) ) 
        n_pool_params = 6 * min(self.n_var_qubits, self.n_ham_qubits)
        # If intermediate_readouts, construct a controller for each parallel pqc.
        if self.intermediate_readouts:
            n_total_params = n_var_params + n_pool_params 
            controller = []
            for i in range(self.n_reuploads):
                ith_controller = tf.keras.Sequential([
                                    tf.keras.layers.Dense(3, 
                                                        input_shape=self.classical_input_shape, 
                                                        activation='elu'),
                                    tf.keras.layers.Dense(n_total_params)],
                                    name='controller_nn_' + str(i)
                                    )
                controller.append(ith_controller)
        # Else, construct single controller for the serial pqc.
        else:        
            n_total_params = ( n_var_params + n_pool_params ) * self.n_reuploads 
            controller = tf.keras.Sequential([
                tf.keras.layers.Dense(3, input_shape=self.classical_input_shape, activation='elu'),
                #tf.keras.layers.Reshape((10, )),
                tf.keras.layers.Dense(n_total_params)],
                name='controller_nn')
    
        return controller

    """
    Create NN that postprocesses outcome of PQC.
    """
    def create_postprocess_nn(self):  
        # Setting input_shape of expectations & classical_input of postprocess_nn.
        if self.intermediate_readouts:                
            q_shape = (len(self.readouts) * self.n_reuploads, )
            c_shape = self.classical_input_shape
            input_shape = tuple(map(sum, zip(q_shape, c_shape))) # (x, ) , (y, ) -> (x + y, )
        else:
            q_shape = (len(self.readouts), )
            c_shape = self.classical_input_shape 
            input_shape = tuple(map(sum, zip(q_shape, c_shape))) # (x, ) , (y, ) -> (x + y, )

        # Setting-up postprocess_nn
        postprocess_nn = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=input_shape, activation='elu'),
            tf.keras.layers.Dense(1)],
            name='postprocess_nn')
        return postprocess_nn

    """
    Create the hybrid model.
    """
    def create_tfq_model(self, print_summary=False, plot_to_file=False):
        ## Setting up input layer.
        quantum_input = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_input')
        classical_input = tf.keras.Input(shape=self.classical_input_shape, dtype=tf.dtypes.float32, 
                                             name='classical_input')

        ## Setting up controller nn(s) & controlled pqc(s) and connecting them to input layer.
        if self.intermediate_readouts:
            # setting up 'controller nn' for each parallel pqc.
            preprocess_nn = [self.controller_nn[i](classical_input) for i in range(self.n_reuploads)]
            pqc_layers = []
            # connecting each controller nn & quantum input to the corresponding pqc.
            for i in range(self.n_reuploads):
                pqc_id = 'pqc'+str(i)
                pqc_layers.append(
                    tfq.layers.ControlledPQC(self.pqc[i], operators=self.readouts, 
                                                name=pqc_id)([quantum_input, preprocess_nn[i]])
                )
            # If multiple reuploads, concatenate outcomes.
            if self.n_reuploads > 1:
                pqc_expectation = tf.keras.layers.concatenate(pqc_layers, name='readout_concatenate')
            else:
                pqc_expectation = pqc_layers[0]
        else:
            # setting up single controller nn for the serial pqc.
            preprocess_nn = self.controller_nn(classical_input)
            # connecting controller nn & quantum input to the serial pqc.
            pqc_layer = tfq.layers.ControlledPQC(self.pqc, operators=self.readouts, name='pqc')
            pqc_expectation = pqc_layer([quantum_input, preprocess_nn])

        ## Connecting PQC to 'postprocess NN'
        postprocess_input = tf.keras.layers.concatenate([pqc_expectation, classical_input],
                                                            name='postprocess_input')
        postprocess_nn = self.postprocess_nn(postprocess_input)
        
        ## Build full keras model from the layers
        model = tf.keras.Model(inputs=[quantum_input, classical_input], outputs=postprocess_nn,
                                name="QML_model")    
    
        ## Print summary of the model.
        if print_summary:
            model.summary()
        ## Show the keras plot of the model
        if plot_to_file:
            path = './img/model_'
            path += 'v-qubits:' + str(self.n_var_qubits)
            path += '_v-depth:' + str(self.var_depth) 
            path += '_reuploads:' + str(self.n_reuploads)
            path += '_intermediate_readouts:' + str(self.intermediate_readouts)
            path += '.png'             
            tf.keras.utils.plot_model(model,
                                      to_file=path,
                                      show_shapes=True,
                                      show_layer_names=True,
                                      dpi=70)
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

        ## Dividing into training and test.
        # Shuffeling dataset
        random.shuffle(dataset)
        # Spliting
        split_ind = int(len(dataset) * 0.7)
        train_data = dataset[:split_ind]
        test_data = dataset[split_ind:]
        
        ## Training data
        print("  - processing training data.")
        # Parsing classical input.
        train_classical_inputs = []
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
            # classical_input = np.concatenate((geometry, orbital_energies), axis=None)
            train_classical_inputs.append(geometry.flatten()) # 2nd experiment: train only on geometry
        self.classical_input_shape = train_classical_inputs[0].shape  

        # Parsing quantum input.
        train_gs_circuits = []            
        for i in range(len(train_data)):
            print('    * loading training circuit', i, '/', len(train_data))            

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
                        for i in range(self.n_reuploads):
                            train_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id])
                elif len(op.qubits) == 2:
                    qubit_id0 = op.qubits[0].col
                    qubit_id1 = op.qubits[1].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        train_gs_circuit += op.with_qubits(self.qubits[qubit_id0], self.qubits[qubit_id1])
                    # else, apply on qubit corresponding to upload.
                    else:
                        for i in range(self.n_reuploads):
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
            # classical_input = np.concatenate((geometry, orbital_energies), axis=None)
            test_classical_inputs.append(geometry.flatten()) # 2nd experiment: train only on geometry
            
        # Parsing quantum input.
        test_gs_circuits = []
        for i in range(len(test_data)):
            print('    * loading validation circuit', i, '/', len(test_data))

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
                        for i in range(self.n_reuploads):
                            test_gs_circuit += op.with_qubits(self.qubits[i*self.n_ham_qubits + qubit_id])
                elif len(op.qubits) == 2:
                    qubit_id0 = op.qubits[0].col
                    qubit_id1 = op.qubits[1].col
                    # if parallel, apply on same qubit.  
                    if self.intermediate_readouts:
                        test_gs_circuit += op.with_qubits(self.qubits[qubit_id0], self.qubits[qubit_id1])
                    # else, apply on qubit corresponding to upload.
                    else:
                        for i in range(self.n_reuploads):
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
        processed_dataset = [train_gs_tensors, np.array(train_classical_inputs), test_gs_tensors,
                    np.array(test_classical_inputs), np.array(train_labels), np.array(test_labels)]
        
        # Pickling the dict.
        print("Creating pickle of processed data for quicker loading.")
        data_id = 'H4_dataset_processed_' + str(len(dataset))
        if self.intermediate_readouts:
            data_id += '_parallel_only-geometry'
        else:
            data_id += '_serial_' + str(self.n_reuploads)
        pickle_path = "./data/" + data_id + '.p' 
        with open(pickle_path, 'wb') as f:      
            pickle.dump(processed_dataset, f)    
      
        return train_gs_tensors, np.array(train_classical_inputs), \
                test_gs_tensors, np.array(test_classical_inputs), \
                np.array(train_labels), np.array(test_labels)

