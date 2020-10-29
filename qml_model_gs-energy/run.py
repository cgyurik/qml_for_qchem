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

"""
Hyperparameters
"""
n_uploads = 2 
n_var_qubits= 0
var_depth= 3

""" 
Setting up directory
"""
print("-----Setting up directories-----")
# Creating directory.
dt_string = datetime.now().strftime('%d-%m_%H%M')
#dir_path = './results/' + dt_string
dir_path = './results/test'
if os.path.exists(dir_path):
    print("Directory already exists; Aborting!")
    exit()
# Creating subdirectories.
os.mkdir(dir_path)
os.mkdir(dir_path + '/img')
os.mkdir(dir_path + '/txt')
os.mkdir(dir_path + '/checkpoints')
os.mkdir(dir_path + '/loss')
# Reporting hyperparameters.
with open(dir_path + '/txt/hyperparams.txt', 'w') as f:
    hyperparams_txt = "uploads: " + str(n_uploads)
    hyperparams_txt += ", var_qubits: " + str(n_var_qubits) 
    hyperparams_txt += " and var_depth: " + str(var_depth)
    print(hyperparams_txt, file=f)
print("Success!")


"""
Setting up the model.
"""
print("-----Setting up model-----")
model = tfq_model(n_gs_uploads=2, n_aux_qubits=0, model_circuit_depth=2, var_depth=2,
                    dir_path=dir_path, print_summary=True, 
                    processed_data='H4_processed_parallel_3'
                    )

print("Compiling model.")
model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)

## Setting up callback to save during training.
checkpoint_path = dir_path + "/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True) 

## Loading weights from previous experiment.
#print("Loading weights")
#model.tfq_model.load_weights("./??")

print("Fitting model.")
history = model.tfq_model.fit(x=model.train_input,
                                y=model.train_labels,
                                epochs=1000, 
                                verbose=1,
                                callbacks=[cp_callback],
                                validation_data=(model.test_input, model.test_labels)
                              )

"""
Saving results.
"""
## Pickling loss and val_loss.
print("Saving results trained model.")
train_loss = history.history['loss'] 
val_loss = history.history['val_loss']
pickle_path = dir_path + '/loss/'
with open(pickle_path + 'train_loss.p', 'wb') as f:      
        pickle.dump(train_loss, f)
with open(pickle_path + 'val_loss.p', 'wb') as f:      
        pickle.dump(val_loss, f)
## Saving final weights.   
model_path = dir_path + '/final_weights'
model.tfq_model.save_weights(model_path)

"""
Plotting results.
"""
print("Plotting validation/training accuracy")
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
