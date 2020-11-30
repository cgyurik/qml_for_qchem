# disable terminal warning tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.path.abspath('..'))
from tfq_model import *
# general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import pickle
# visualization tools
import matplotlib.pyplot as plt


"""
Benchmark trained model against HF and classical model.
"""
def benchmark_model(qml_model, dir_path, tfq_weights=None, c_layer_sizes=[8], epochs=500):
    ## Setting up classical model.
    print("  - setting up classical model.")
    # processings multiple inputs.
    inputs = [tf.keras.Input(shape=(3, )), tf.keras.Input(shape=(3, )), tf.keras.Input(shape=(3, )),
                                                                        tf.keras.Input(shape=(4, ))]
    merged_inputs = tf.keras.layers.Concatenate(axis=1)(inputs)
    # adding dense NN layers.
    output = tf.keras.layers.Dense(c_layer_sizes[0])(merged_inputs)
    for i in range(len(c_layer_sizes)-1):
        output = tf.keras.layers.Dense(c_layer_sizes[i+1], activation='relu')(output)
        #c_layers.append(c_layer)
    output = tf.keras.layers.Dense(1)(output)
    c_model = tf.keras.Model(inputs=inputs, outputs=output, name="c_model")
    c_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
    c_model.summary()
    print("  - fitting classical model.")
    history = c_model.fit(qml_model.train_input[1:5], qml_model.train_labels, epochs=epochs, verbose=1)
    
    ## Loading weights.
    if tfq_weights is not None:
        q_model.tfq_model.load_weights(tfq_weights)

    ## Evaluating models on test sets
    print("  - evaluating models.")
    print("    * quantum.")
    q_test_predictions = qml_model.tfq_model.predict(qml_model.test_input, verbose=1)
    print("    * classical.")
    c_test_predictions = c_model.predict(qml_model.test_input[1:5], verbose=1)
 
    ## Plotting comparisson.
    a = plt.axes(aspect='equal')
    plt.scatter(qml_model.test_labels, q_test_predictions, label = 'Predicted by QML model')
    plt.scatter(qml_model.test_labels, qml_model.test_hfe, label = 'HF energy', marker='x')
    plt.scatter(qml_model.test_labels, c_test_predictions, label = 'Predicted by CML model')
    plt.xlabel('True GS energies (FCI) [Ha]')
    plt.ylabel('Energy by other method [Ha]')
    lims = plt.gca().get_ylim()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.legend()
    #plt.show()
    path = dir_path + '/img/comparisson.png'
    plt.savefig(path)

"""
Gather results of trained QML model.
"""
def save_results(qml_model, history, dir_path):
    ## Saving train/val loss and final weights.
    train_loss = history.history['loss'] 
    val_loss = history.history['val_loss']
    pickle_path = dir_path + '/loss_pickles/'
    with open(pickle_path + 'train_loss.p', 'wb') as f:      
            pickle.dump(train_loss, f)
    with open(pickle_path + 'val_loss.p', 'wb') as f:      
            pickle.dump(val_loss, f)
    with open(dir_path + '/txt/final_losses.txt', 'w') as f:
            loss_txt = "Training loss:" + str(train_loss[-1]) + ", Validation loss:" + str(val_loss[-1])
            print(loss_txt, file=f)
    model_path = dir_path + '/final_weights'
    qml_model.tfq_model.save_weights(model_path)

    ## Plotting results.
    plt.plot(history.history['loss'], label='qml_model')
    plt.title('QML model performance')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    path = dir_path + '/img/train_loss.png'
    plt.savefig(path)
    plt.close()
    plt.plot(history.history['val_loss'], label='qml_model')
    plt.title('QML model performance')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    path = dir_path + '/img/val_loss.png'
    plt.savefig(path)
    plt.close()
