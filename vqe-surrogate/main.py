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
Observations first experiments:
- Surrogate training on ~100 samples seems to plateau after 50 epochs.
"""


""" 
Train surrogate VQE cost function on randomized evaluations of VQE cost function.
"""
def train_surrogate_random(vqe, n_samples=100, epochs=15):
    print("-----Training and Testing surrogate on random parameters-----")
    ## Generating random data.
    print("Generating training/validation data.")
    random_data = []
    for i in range(n_samples):
        params = np.random.uniform(0, 2 * np.pi, len(vqe.ansatz.symbols))
        random_data.append({"params": params, "energy": vqe.vqe_cost(params)})
    
    ## Training QML model (i.e., the surrogate)    
    print("Training the surrogate.")
    print("  - loading the data.")
    vqe.surrogate.load_data(random_data, vqe.ansatz)
    print("  - fitting the QML model.")
    history = vqe.surrogate.tfq_model.fit(x=vqe.surrogate.train_states, 
                                            y=vqe.surrogate.train_labels,
                                            batch_size=32,
                                            epochs=epochs,
                                            verbose=1,
                                            validation_data=(vqe.surrogate.test_states, 
                                                             vqe.surrogate.test_labels))
                             
    return history
    
        
""" 
Train surrogate VQE cost function on evaluations following VQE optimization trajectory.
"""
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
Benchmark (i.e., cross-validate and compare to classical model) surrogate on random evaluations.
"""  
def benchmark_random_experiment(filename, n_folds=3):
    ## Trying out multiple depths of the hwe ansatz.
    depths = [5, 15, 30]
    for depth in depths:
        ## Setting up cross-validation.
        histories = []
        best_val_loss = float('inf')
        for fold in range(n_folds): 
            print("+++++ fold", fold, ". +++++")  
            # Setting up VQE.
            cur_vqe = vqe(filename, n_uploads=1, var_depth=depth)
            # 250 training, 50 testing samples per depth increase.
            n_samples = 300 * (depths.index(depth) + 1)
            history = train_surrogate_random(cur_vqe, n_samples=n_samples, epochs=50)
            histories.append(history)

            # Saving best fold to compare with classical model.
            if history.history['val_loss'] < best_val_loss:
                train_labels = cur_vqe.surrogate.train_labels
                train_params = cur_vqe.surrogate.train_params
                test_params = cur_vqe.surrogate.test_params
                q_train_predictions = cur_vqe.surrogate.tfq_model.predict(cur_vqe.surrogate.train_states,
                                                                            verbose=1).flatten()
                q_test_predictions = cur_vqe.surrogate.tfq_model.predict(cur_vqe.surrogate.test_states,
                                                                            verbose=1).flatten()
                best_val_loss = history.history['val_loss']
                
                
        ## Reporting results.
        print("Reporting results.")
        # Writing average final losses to .txt
        txt_name = './avg_losses_' + str(n_samples) + '_' + str(epochs) + '.txt'
        avg_train_loss = np.mean([history.history['loss'][-1] for history in histories])
        avg_val_loss = np.mean([history.history['val_loss'][-1] for history in histories])
        with open(txt_name, 'w') as f:
            print("- Average final training loss:", avg_train_loss, file=f)
            print("- Average final validation loss:", avg_val_loss, file=f)
        # Plotting training losses
        for i in range(n_folds):
            label = 'Training loss fold' + str(i) + '.'
            plt.plot(histories[i].history['loss'], label=label)
        plt.xlabel('Epoch.')
        plt.ylabel('MSE of predicted energy.')
        filename = './train_loss_' + str(depth) + '.png'
        plt.savefig(filename)
        plt.close()
        # Plotting validation losses
        for i in range(n_folds):
            label = 'Validation loss fold' + str(i) + '.'
            plt.plot(histories[i].history['val_loss'], label=label)
        plt.xlabel('Epoch.')
        filename = './val_loss_' + str(depth) + '.png'
        plt.savefig(filename)
        plt.close()

        ## Comparing best fold against classical model.
        print("Comparing best fold against classical model.")
        layer_size = int(depth / 3) 
        input_layer = tf.keras.Input(shape=(len(best_vqe.ansatz.symbols), ))
        output = tf.keras.layers.Dense(layer_size)(input_layer)
        output = tf.keras.layers.Dense(layer_size, activation='relu')(output)
        output = tf.keras.layers.Dense(1)(output)
        c_model = tf.keras.Model(inputs=input_layer, outputs=output, name="c_model")
        c_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
        print("  - fitting classical model.")
        history = c_model.fit(x=train_params, y=train_labels, epochs=epochs, verbose=1)    
    
        # Evaluating models on test sets
        print("  - evaluating classical model.")
        c_test_predictions = c_model.predict(test_params, verbose=1).flatten()
        c_train_predictions = c_model.predict(train_params, verbose=1).flatten()
     
        ## Plotting comparisson.
        # Training.
        a = plt.axes(aspect='equal')
        plt.scatter(best_vqe.surrogate.test_labels, q_test_predictions, label = 'Predicted by QML model')
        plt.scatter(best_vqe.surrogate.test_labels, c_test_predictions, label = 'Predicted by CML model')
        plt.xlabel('True energies [Ha]')
        plt.ylabel('Energy by other method [Ha]')
        lims = plt.gca().get_ylim()
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.legend()
        path = './comparisson_' + str(n_samples) + '_' + str(epochs) + '.png'
        plt.savefig(path)
        plt.close()
        
        # Testing.
        a = plt.axes(aspect='equal')
        plt.scatter(best_vqe.surrogate.test_labels, q_test_predictions, label = 'Predicted by QML model')
        plt.scatter(best_vqe.surrogate.test_labels, c_test_predictions, label = 'Predicted by CML model')
        plt.xlabel('True energies [Ha]')
        plt.ylabel('Energy by other method [Ha]')
        lims = plt.gca().get_ylim()
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.legend()
        path = './comparisson_' + str(n_samples) + '_' + str(epochs) + '.png'
        plt.savefig(path)
        plt.close()
        
            
if __name__ == "__main__":    
    filename = "./molecules/molecule1"
    benchmark_random_experiment(filename)
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
