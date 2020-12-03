# disable terminal warning tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tfq_model import *
from utils.benchmark import *
# general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from datetime import datetime
# visualization tools
import matplotlib.pyplot as plt

"""
Run QML model experiments with given hyperparameters
"""
def experiment(n_gs_uploads, n_aux_qubits, var_depths, dir_prefix, epochs=250, processed_data=None):
    for i in range(len(n_gs_uploads)):
        """ 
        Setting up directory
        """
        print("-----Setting up directories-----")
        #dir_path = './results/' + dt_string
        dir_path = './results/' + dir_prefix + '-' + str(i)
        if os.path.exists(dir_path):
            print("Directory already exists; Aborting!")
            exit()
        # Creating subdirectories.
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/img')
        os.mkdir(dir_path + '/txt')
        os.mkdir(dir_path + '/checkpoints')
        os.mkdir(dir_path + '/loss_pickles')
        # Reporting hyperparameters.
        with open(dir_path + '/txt/hyperparams.txt', 'w') as f:
            hyperparams_txt = "n_gs_uploads: " + str(n_gs_uploads[i])
            hyperparams_txt += ", n_aux_qubits: " + str(n_aux_qubits[i]) 
            hyperparams_txt += " and var_depth: " + str(var_depths[i])
            print(hyperparams_txt, file=f)
        print("Success!")

        print("-----Setting up model-----")
        qml_model = tfq_model(n_gs_uploads=n_gs_uploads[i], n_aux_qubits=n_aux_qubits[i],
                                var_depth=var_depths[i], normalize_data=True, 
                                dir_path=dir_path, print_summary=True,              
                                processed_data=processed_data)

        print("Compiling model.")
        qml_model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)

        ## Setting up callback to save during training.
        checkpoint_path = dir_path + "/checkpoints/cp-{epoch:03d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True) 

        ## Loading weights from previous experiment.
        print("Loading weights")
        qml_model.tfq_model.load_weights("./results/givens_experiment-0/final_weights")

        print("-----Fitting quantum model.------")
        history = qml_model.tfq_model.fit(x=qml_model.train_input,
                                            y=qml_model.train_labels,
                                            epochs=epochs, 
                                            verbose=1,
                                            callbacks=[cp_callback],
                                            validation_data=(qml_model.test_input, qml_model.test_labels)
                                            )
        print("Success!")
        print("------Saving results.-----")
        save_results(qml_model, history, dir_path)
        benchmark_model(qml_model, dir_path, c_layer_sizes=[10, 8])
        
if __name__ == "__main__":
    """
    Hyperparameters
    """
    n_gs_uploads = [1] 
    n_aux_qubits= [0]
    var_depths= [0]

    experiment(n_gs_uploads, n_aux_qubits, var_depths, 'givens_experiment_extended')
