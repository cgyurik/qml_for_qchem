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
from utils.tfq_utils import tensorable_ucc_circuit


"""
[Research]
    - fewer parameters (?)
"""

"""
[TFQ meeting]
    - Contribute H4 dataset?

"""

"""
[TODO]    
    - Givens rotation ansatz.
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
    def __init__(self, n_gs_uploads=2, n_aux_qubits=2, var_depth=2, normalize_data=False,
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
        self.train_input, self.test_input, \
        self.train_labels, self.test_labels, \
        self.test_hfe = self.load_dataset(normalize=normalize_data, processed_data=processed_data)
        
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
        #n_var_symbols_upload = 2 * self.n_qubits * (self.var_depth + 1)
        #self.var_symbols = sympy.symbols('pqc0:' + str(n_var_symbols_layer * self.n_gs_uploads))
        n_var_symbols_upload = ( (self.n_qubits) * ( (self.n_qubits) - 1) ) // 2
        self.var_symbols = sympy.symbols('pqc0:' + str(n_var_symbols_upload * self.n_gs_uploads))
    
        # Creating the (parallel) model circuits
        model_circuits = []
        for i in range(self.n_gs_uploads):
            ith_model_circuit = cirq.Circuit()
            var_symbols_upload = self.var_symbols[i * n_var_symbols_upload : (i+1) * n_var_symbols_upload]
            #ith_model_circuit += pqc.variational_circuit(self.qubits, var_symbols_layer,
            #                                                                        depth=self.var_depth)    
            ith_model_circuit = cirq.Circuit(pqc.parametrized_givens_ansatz(self.qubits, var_symbols_upload))
            #ith_model_circuit = cirq.Circuit(pqc.parametrized_spin_conserving_givens_ansatz(
            #                                                                            self.qubits,
            #                                                                            var_symbols_upload
            #                                                                                ))
            model_circuits.append(ith_model_circuit)

        ## Printing the circuit(s).
        if print_circuit:   
            # Checking if dir_path is specified, otherwise print to terminal.
            if self.dir_path is None:
                print(model_circuits[0].to_text_diagram(transpose=True))
            else:
                with open(self.dir_path + '/txt/encoding_circuit.txt', 'w') as f:
                    print(model_circuits[0].to_text_diagram(transpose=True), file=f)
        
        return model_circuits
        
    """
    Create NN(s) that controlls parameters of encoding circuit(s).
    """
    def create_controller_nn(self):
        #n_params = ( 2 * (self.n_aux_qubits + self.n_ham_qubits) * (self.var_depth + 1) ) 
        n_params = ( self.n_qubits * (self.n_qubits - 1) ) // 2
        controllers = []
        for i in range(self.n_gs_uploads):
            ith_controller = tf.keras.Sequential(
                                [#tfmot.sparsity.keras.prune_low_magnitude(
                                    tf.keras.layers.Dense(n_params, input_shape=(7,))
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
    def load_dataset(self, normalize=False, processed_data=None):
        ## Loading generated dataset.
        if processed_data is not None:
            print("  - reading directly from processed dataset pickle.")
            path = './data/'+processed_data+'.p'
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
                return dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]
                           
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
                if len(dataset) == 2:
                    break
                
        ## Removing degenerate molecules
        dataset = [dataset[j] for j in range(len(dataset)) if dataset[j]['multiplicity'] == 1]

        ## Processing dataset
        print("  - processing dataset.")
        geometry_1 = []
        geometry_2 = []
        geometry_3 = []
        orbital_energies = []
        groundstate_circuits = [] 
        print('    * loading classical data.')
        for i in range(len(dataset)):
            # Only include molecules with groundstate degeneracy 1.
            if dataset[i]['multiplicity'] == 3:
                continue

            # Reading data from dict.
            geometry = np.transpose(np.array([dataset[i]['geometry'][j][1] for j in range(1, 4)]))
            canonical_orbitals = np.array(dataset[i]['canonical_orbitals'])
            orbital_energy = np.array(dataset[i]['orbital_energies'])
            canonical_to_oao = np.array(dataset[i]['canonical_to_oao'])  
            
            # Puting geometry & orbital energies in classical input
            geometry_1.append(geometry[0])
            geometry_2.append(geometry[1])
            geometry_3.append(geometry[2])
            orbital_energies.append(orbital_energy)
            groundstate_circuits.append(dataset[i]['gs_circuit'])

        ## Normalize input
        if normalize:
            print('    * normalizing the input.')
            geometry_mean = np.mean(np.concatenate((geometry_1, geometry_2, geometry_3)), axis=0)
            geometry_std = np.std(np.concatenate((geometry_1, geometry_2, geometry_3)), axis=0)
            geometry_1 = np.array(geometry_1 - geometry_mean)/geometry_std
            geometry_2 = np.array(geometry_2 - geometry_mean)/geometry_std
            geometry_3 = np.array(geometry_3 - geometry_mean)/geometry_std
            orbital_energies = np.array((orbital_energies - np.mean(orbital_energies, axis=0))
                                                            /np.std(orbital_energies, axis=0))
            # checking dividing by zero.
            if any(v == 0 for v in geometry_std) or any(v == 0 for v in np.std(orbital_energies, axis=0)):
                print("Dividing by standard deviation==zero; Aborting!")
                exit()    
              
        # Collecting the processed data.
        inputs = [np.array(geometry_1), np.array(geometry_2), np.array(geometry_3), 
                    np.array(orbital_energies)]
        #print(inputs)
        labels = [dataset[j]['exact_energy'] for j in range(len(dataset))]
        hf_energies = [dataset[j]['hf_energy'] for j in range(len(dataset))]

        ## Dividing into training and test.
        print('    * extracting output and input, and test/train splitting.')
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split_ind = int(len(dataset) * 0.7)
        # Converting circuits to tensors.
        print('    * converting circuits to tensors.')
        train_gs_tensors = tfq.convert_to_tensor([groundstate_circuits[i] for i in indices[:split_ind]])
        test_gs_tensors = tfq.convert_to_tensor([groundstate_circuits[i] for i in indices[split_ind:]])
        # Collecting the inputs.
        train_inputs = [train_gs_tensors, 
                        np.array([inputs[0][i] for i in indices[:split_ind]]),
                        np.array([inputs[1][i] for i in indices[:split_ind]]),
                        np.array([inputs[2][i] for i in indices[:split_ind]]),
                        np.array([inputs[3][i] for i in indices[:split_ind]])]
        train_labels = [labels[i] for i in indices[:split_ind]]
        test_inputs = [test_gs_tensors, 
                        np.array([inputs[0][i] for i in indices[split_ind:]]),
                        np.array([inputs[1][i] for i in indices[split_ind:]]),
                        np.array([inputs[2][i] for i in indices[split_ind:]]),
                        np.array([inputs[3][i] for i in indices[split_ind:]])]
        test_labels = [labels[i] for i in indices[split_ind:]]
        test_hf_energies = [hf_energies[i] for i in indices[split_ind:]]
        print('    * processing done!')
        
        """
        ## Pickling for next time.
        processed_dataset = [train_inputs, test_inputs, train_labels, test_labels, test_hf_energies]
        print("  - creating pickle of processed data for quicker loading.")
        data_id = 'H4_processed_' + str(len(dataset))
        pickle_path = "./data/" + data_id + '.p' 
        with open(pickle_path, 'wb') as f:      
            pickle.dump(processed_dataset, f)    
        """

        return train_inputs, test_inputs, np.array(train_labels), np.array(test_labels), test_hf_energies

if __name__ == "__main__":
    test_model = tfq_model(n_gs_uploads=2, n_aux_qubits=1, var_depth=2, normalize_data=True,
                            print_circuit=True, print_summary=True)
