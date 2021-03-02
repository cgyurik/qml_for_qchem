## os/sys tools
import os, sys
# disable terminal warning tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
## general tools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cirq
from scipy.optimize import minimize
## vqe/qml tools.
import openfermion
import tensorflow_quantum as tfq
from vqe import *
from vqe_utils.uccsd import UCCSDAnsatz
from qml_model.tfq_model import tfq_model
## visualization tools
import matplotlib.pyplot as plt

    
""" 
Train surrogate VQE cost function on randomized evaluations of VQE cost function.
"""
def train_surrogate_random(test_vqe, radius=0.1, n_samples=100, epochs=15):
    print("-----Training and Testing surrogate on random parameters in neighbourhood-----")
    ## Generating random centre of sphere.
    print("Generating training/validation data.")
    center = np.random.uniform(0, 2 * np.pi, len(test_vqe.ansatz.symbols))
    ## Generating random data in radius around center.
    random_data = []
    for i in range(n_samples):
        if radius == 0:
            params = np.random.uniform(0, 2 * np.pi, len(test_vqe.ansatz.symbols))
        else:
            params = np.zeros(len(test_vqe.ansatz.symbols))
            for i in range(len(test_vqe.ansatz.symbols)):
                params[i] = np.random.uniform(center[i] - radius, center[i] + radius)
        random_data.append({"params": params, "energy": test_vqe.vqe_cost(params)})
        
    ## Training QML model (i.e., the surrogate)    
    print("Training the surrogate.")
    print("  - loading the data.")
    test_vqe.surrogate.load_data(random_data, test_vqe.ansatz)
    print("  - fitting the QML model.")
    history = test_vqe.surrogate.tfq_model.fit(x=test_vqe.surrogate.train_states, 
                                                    y=test_vqe.surrogate.train_labels,
                                                    batch_size=32,
                                                    epochs=epochs,
                                                    verbose=1,
                                                    validation_data=(test_vqe.surrogate.test_states, 
                                                                    test_vqe.surrogate.test_labels))
                             
    return history    
        
            
""" 
Hyperparameter sweep for the hardware-efficient ansatz.
"""  
def hwe_experiment(molecule, depths=[3, 7, 13], radii=[0.01, 0.05, 0.1, 0.25, 0.5], n_samples=75, epochs=200):
    ##Setting up directory
    print("-----Setting up directories-----")
    dir_path = './results/experiment_02-03'
    if os.path.exists(dir_path):
        print("Directory already exists; Aborting!")
        exit()
    # Creating subdirectories.
    os.mkdir(dir_path)
    os.mkdir(dir_path + '/data')
    os.mkdir(dir_path + '/weights')
    os.mkdir(dir_path + '/loss')
    print("Success!")
    ## Hyperparameter-sweep
    for depth in depths:
        for radius in radii:
            print("===== depth", depth, "and radius", radius, "=====")      
            # Setting up VQE.
            current_vqe = vqe(molecule, n_uploads=1, var_depth=depth)
            # Training vqe surrogate on random data.
            history = train_surrogate_random(current_vqe, radius=radius, n_samples=n_samples, epochs=epochs)
            ## Saving results
            print("Saving the results.")
            # Training data.
            filename =  '/data/train_params-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:
                pickle.dump(current_vqe.surrogate.train_params, f)
            filename =  '/data/train_labels-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:
                pickle.dump(current_vqe.surrogate.train_labels, f)
            # Validation data.
            filename =  '/data/test_params-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:
                pickle.dump(current_vqe.surrogate.test_params, f)
            filename =  '/data/test_labels-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:
                pickle.dump(current_vqe.surrogate.test_labels, f)
            # Trained weights.
            trained_weights = current_vqe.surrogate.pqc_layer.symbol_values()
            filename =  '/weights/trained_weights-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:
                pickle.dump(trained_weights, f)
            # Training loss.
            train_loss = history.history['loss'] 
            filename =  '/loss/train_loss-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:      
                pickle.dump(train_loss, f)
            # Validation loss.
            val_loss = history.history['val_loss']
            filename =  '/loss/val_loss-' + str(depth) + '_' + str(radius) + '.p'
            with open(dir_path + filename, 'wb') as f:   
                pickle.dump(val_loss, f)
            # Saving final losses
            filename =  '/final_loss-' + str(depth) + '_' + str(radius) + '.txt'
            with open(dir_path + filename, 'w') as f: 
                txt = "Training loss:" + str(train_loss[-1]) + " and Validation loss:" + str(val_loss[-1])
                print(txt, file=f)
                    
        
            
if __name__ == "__main__":    
    molecule = "./molecules/molecule1"
    hwe_experiment(molecule)
    #vqe_test = vqe(filename, n_uploads=1, var_depth=1, verbose=True)
    #train_surrogate_random(vqe_test, n_samples=3, epochs=1)
    
    
"""                                                         
## Bug-fixing
# PQC setup.
resolver = cirq.ParamResolver(vqe.surrogate.pqc_layer.symbol_values())
resolved_pqc = cirq.resolve_parameters(vqe.surrogate.pqc, resolver)    

# VQE setup.
for params in vqe.surrogate.train_params:
    resolved_vqe = vqe.ansatz.tensorable_ucc_circuit(params, vqe.qubits)
    print("Resolved the circuits.")
    final_circuit = resolved_vqe + resolved_pqc
    print("Generating final state.")
    final_state = cirq.final_state_vector(final_circuit)
    qubit_map = {}
    for i in range(len(vqe.qubits)):
        qubit_map[vqe.qubits[i]] = i
    energy = vqe.surrogate.readouts.expectation_from_state_vector(
                                                final_state/np.linalg.norm(final_state),qubit_map).real
    print(energy, vqe.surrogate_cost(params))
"""

"""
## Train surrogate VQE cost function on evaluations following VQE optimization trajectory.
def train_surrogate_trajectory(vqe, maxfev=100, epochs=5):
    print("----- Testing surrogate on parameters following optimization trajectory-----")

    ## Generating data following VQE cost function optimization trajectory.
    print("Generating VQE cost function optimization trajectory for training.")
    vqe.params = minimize(vqe.vqe_cost, x0=vqe.params, method="Nelder-Mead", options={'maxfev':maxfev}).x
    n_train_evals = len(vqe.eval_history)

    ## Training the surrogate.
    print("Training the surrogate.")
    print("  - loading the data, consisting of", len(vqe.eval_history), "evaluations.")
    vqe.surrogate.load_data(vqe.eval_history, vqe.ansatz, split=1)
    history = vqe.surrogate.tfq_model.fit(x=vqe.surrogate.train_states, 
                                    y=vqe.surrogate.train_labels,
                                    batch_size=32,
                                    epochs=epochs,
                                    verbose=1)
    
    ## Testing surrogate in new evalutions on optimization trajectory.
    print("Generating VQE cost function optimization trajectory for validation.")
    vqe.params = minimize(vqe.vqe_cost, x0=vqe.params, method="Nelder-Mead", options={'maxfev':maxfev}).x
    validation_evals = vqe.eval_history[n_train_evals:]
    mse_surrogate_prediction = []
    print("Computing mse of the surrogate on validation part of trajectory.")
    for i in range(len(validation_evals)):
        true_energy = validation_evals[i]['energy']
        surrogate_prediction = vqe.surrogate_cost(validation_evals[i]['params'])
        mse_surrogate_prediction.append( (true_energy - surrogate_prediction)**2 )
    avg_mse_surrogate_prediction = np.sum(mse_surrogate_prediction) / len(mse_surrogate_prediction)
    
    ## Plotting results.
    print("Reporting results.")
    with open('./trajectory_losses.txt', 'w') as f:
        print("    * final training loss:", history.history['loss'][-1], file=f)
        print("    * average loss over new part trajectory:", avg_mse_surrogate_prediction, file=f)
    plt.plot(history.history['loss'], label = 'Training loss.')
    plt.xlabel('Epoch.')
    plt.ylabel('MSE of the predicted energy.')
    plt.savefig('./trajectory_training_loss.png')
    plt.close()
    print(mse_surrogate_prediction)
    plt.plot(mse_surrogate_prediction, label='Validation loss over new part trajectory')
    plt.xlabel('Iteration of optimizer.')
    plt.ylabel('MSE of the predicted energy.')
    plt.savefig('./trajectory_validation_loss.png')    
"""

