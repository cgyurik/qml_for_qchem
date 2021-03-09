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
Plotting losses
"""
def plot_losses(train_losses, val_losses, radius, depths=[3, 7, 13], dir_path='./results/experiment_02-03'):
    ## Training loss.
    plt.title('Training loss for radius ' + str(radius) + '.')
    for i in range(len(train_losses)):
        label = 'depth '+ str(depths[i])
        plt.plot(train_losses[i], label=label)
    plt.xlabel('Epoch.')
    plt.ylabel('MSE of predicted energy.')
    plt.legend()
    filename = dir_path + '/train_loss-' + str(radius) + '.png'
    plt.savefig(filename)
    plt.close()
    ## Validation loss.
    plt.title('Validation loss for radius ' + str(radius) + '.')
    for i in range(len(val_losses)):
        label = 'depth '+ str(depths[i])
        plt.plot(val_losses[i], label=label)
    plt.xlabel('Epoch.')
    plt.ylabel('MSE of predicted energy.')
    plt.legend()
    filename = dir_path + '/val_loss-' + str(radius) + '.png'
    plt.savefig(filename)
    plt.close()

"""
Comparing against classical model.
"""
def compare_classical(params, labels, cur_vqe, pqc_weights, model_id, layers=[5, 5, 5], 
                        dir_path='./results/experiment_02-03'):
    ## Constructing classical model.
    input_layer = tf.keras.Input(shape=(len(params[0][0]), ))
    output = tf.keras.layers.Dense(layers[0])(input_layer)
    for i in range(1, len(layers)):
        output = tf.keras.layers.Dense(layers[i], activation='relu')(output)
    output = tf.keras.layers.Dense(1)(output)
    ## Training classical model.
    c_model = tf.keras.Model(inputs=input_layer, outputs=output, name="c_model")
    c_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
    print("  - fitting classical model.")
    history = c_model.fit(x=np.array(params[0]), y=labels[0], epochs=250, verbose=0)    
    ## Evaluating classical models on test sets.
    print("  - evaluating classical model.")
    c_test_predictions = c_model.predict(np.array(params[1])).flatten()
    ## Evaluating quantum model on test sets.
    print("  - evaluating quantum model.")
    q_test_predictions = []
    resolver = cirq.ParamResolver(pqc_weights)
    resolved_pqc = cirq.resolve_parameters(cur_vqe.surrogate.pqc, resolver)    
    for param in params[1]:
        resolved_vqe = cur_vqe.ansatz.tensorable_ucc_circuit(param, cur_vqe.qubits)
        final_circuit = resolved_vqe + resolved_pqc
        final_state = cirq.final_state_vector(final_circuit)
        qubit_map = {}
        for i in range(len(cur_vqe.qubits)):
            qubit_map[cur_vqe.qubits[i]] = i
        prediction = cur_vqe.surrogate.readouts.expectation_from_state_vector(
                                                    final_state/np.linalg.norm(final_state),qubit_map).real
        q_test_predictions.append(prediction)
    print(q_test_predictions, model_id)
    
    """    
    ## Plotting comparisson.
    print("  - plotting results.")
    a = plt.axes(aspect='equal')
    plt.scatter(labels[1], q_test_predictions, label = 'Predicted by QML model')
    plt.scatter(labels[1], c_test_predictions, label = 'Predicted by CML model')
    plt.xlabel('True energies [Ha]')
    plt.ylabel('Energy by other method [Ha]')
    lims = plt.gca().get_ylim()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.legend()
    filename = dir_path + '/comparison-' + model_id + '.png'
    plt.savefig(filename)
    plt.close()
    """
    
if __name__ == "__main__":
    ## Location of current results
    dir_path = './results/experiment_02-03/folds'
    ## Hyperparameters
    radii = [0.01, 0.05, 0.1, 0.25, 0.5]
    #radii = [0.5]
    # quantum
    depths = [3, 7, 13]
    #depths = [3, 7]
    # classical
    nn_layers = [[2], [3], [4, 4]]
    ## Representative folds per depth
    folds = [[1, 1, 1, 0, 0], [1, 0, 2, 1, 0], [2, 2, 2, 1, None]]
    #folds = [[0], [0]]
    
    ## Going over all hyperparameter configurations
    for i in range(len(radii)):    
        train_losses = []
        val_losses = []
        for j in range(len(depths)):
            best_fold = folds[j][i]
            # skip if experiment not finished
            if best_fold is None:
                break
            depth = str(depths[j])
            radius = str(radii[i])
            ## Collecting all the losses
            dir_path_fold = dir_path + '/fold_' + str(best_fold)
            dir_path_temp = dir_path_fold + '/loss/train_loss-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                train_losses.append(pickle.load(openfile))
            dir_path_temp = dir_path_fold + '/loss/val_loss-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                val_losses.append(pickle.load(openfile))
            
            ## Collecting all train/test parameters
            dir_path_temp = dir_path_fold + '/data/train_params-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                train_params = pickle.load(openfile)
            dir_path_temp = dir_path_fold + '/data/test_params-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                test_params = pickle.load(openfile)
            params = [train_params, test_params]
            
            ## Collecting all train/test labels
            dir_path_temp = dir_path_fold + '/data/train_labels-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                train_labels = pickle.load(openfile)
            dir_path_temp = dir_path_fold + '/data/test_labels-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                test_labels = pickle.load(openfile)
            labels = [train_labels, test_labels]
     
            ## Setting up vqe.
            print("Comparing quantum with classical model for depth", depths[j], "and radius", radii[i])
            cur_vqe = vqe('./molecules/molecule1', n_uploads=1, var_depth=depths[j])       
            model_id = depth + '_' + radius
            dir_path_temp = dir_path_fold + '/weights/trained_weights-' + depth + '_' + radius + '.p'
            with (open(dir_path_temp, 'rb')) as openfile:
                pqc_weights = pickle.load(openfile) 
            compare_classical(params, labels, cur_vqe, pqc_weights, model_id, layers=nn_layers[j])
        ## Plotting the losses
        plot_losses(train_losses, val_losses, radii[i])    
