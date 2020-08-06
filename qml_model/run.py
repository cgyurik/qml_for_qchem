# disable terminal warning tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tfq_model import *
# general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from datetime import datetime
# visualization tools
import matplotlib.pyplot as plt

# Hyperparameters
reups = [3, 1, 2, 3, 4, 5]
qubits = [2, 2, 2, 1, 1, 1]
depths = [3, 7, 3, 2, 1, 1]


# Looping over all experiments
for i in range(len(reups)):
    # Setting up directory
    print("-----Setting up directories-----")
    #dt_string = datetime.now().strftime('%H%M%S')
    dt_string = datetime.now().strftime('%d-%m_%H%M')
    dir_path = './results/experiment_' + str(i) + '_'+ dt_string
    if os.path.exists(dir_path):
        print("Directory already exists; Aborting!")
        exit()
    os.mkdir(dir_path)
    os.mkdir(dir_path + '/img')
    os.mkdir(dir_path + '/txt')
    os.mkdir(dir_path + '/checkpoints')
    os.mkdir(dir_path + '/loss')
    with open(dir_path + '/txt/hyperparams.txt', 'w') as f:
        hyperparams = "reuploads:" + str(reups[i]) + 
                        ", qubits:" + str(qubits[i]) + 
                        "and depth:" + str(depths[i])
        print(hyperparams, file=f)
    print("Success!")

    # Setting up the model.
    print("-----Setting up model-----")
    model = tfq_model(n_var_qubits=2, var_depth=3, n_reuploads=3,
                        intermediate_readouts=True,
                        dir_path = dir_path, print_summary=True, plot_to_file=True,
                        processed_data='H4_processed_parallel_501'
                    )
    
    print("Compiling model.")
    model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)

    # Setting up callback to save during training.
    checkpoint_path = dir_path + "/checkpoints/cp-{epoch:02d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        save_weights_only=True, 
                                                        verbose=1)
    callbacks = [cp_callback]


    print("Fitting model.")
    history = model.tfq_model.fit(x=[model.train_groundstates, 
                                        model.train_classical_inputs[0],
                                        model.train_classical_inputs[1], 
                                        model.train_classical_inputs[2], 
                                        model.train_classical_inputs[3]],
                                    y=model.train_labels,
                                    batch_size=32, #default
                                    epochs=50, #seems enough from prior experiments
                                    verbose=1,
                                    callbacks=callbacks,
                                    validation_data=([model.test_groundstates, 
                                                        model.test_classical_inputs[0], 
                                                        model.test_classical_inputs[1],
                                                        model.test_classical_inputs[2], 
                                                        model.test_classical_inputs[3]], 
                                                        model.test_labels))

    # Saving results.
    print("Saving results trained model.")
    train_loss = history.history['loss'] 
    val_loss = history.history['val_loss']
    pickle_path = dir_path + '/loss/'
    with open(pickle_path + 'train_loss.p', 'wb') as f:      
            pickle.dump(train_loss, f)
    with open(pickle_path + 'val_loss.p', 'wb') as f:      
            pickle.dump(val_loss, f)   
    model_path = dir_path + '/final_weights'
    model.tfq_model.save_weights(model_path)

    # Plotting results.
    print("Plotting validation/training accuracy")
    plt.plot(history.history['loss'], label='qml_model')
    plt.title('QML model performance')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    path = dir_path + '/img/train_loss'
    plt.savefig(path)
    plt.close()
    plt.plot(history.history['val_loss'], label='qml_model')
    plt.title('QML model performance')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    path = dir_path + '/img/val_loss'
    plt.savefig(path)
    plt.close()

