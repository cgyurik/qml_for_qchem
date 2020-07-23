# disable terminal warning tf
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.path.abspath('..'))
from tfq_model import *
# general tools
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
# visualization tools
import matplotlib.pyplot as plt

# Setting up the model.
print("-----Setting up model-----")
model = tfq_model(n_var_qubits=2, var_depth=2, n_reuploads=2, intermediate_readouts=True, 
                    processed_data='H4_dataset_processed_10_parallel', print_summary=True, plot_to_file=False)
print("Compiling model.")
model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02), loss=tf.losses.mse)

# Setting up callback to save during training.
checkpoint_path = "models/test/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1)

print("Fitting model.")
history = model.tfq_model.fit(x=[model.train_groundstates, model.train_classical_inputs],
                                    y=model.train_labels,
                                    batch_size=10,
                                    epochs=3,
                                    verbose=1,
                                    validation_data=([model.test_groundstates, model.test_classical_inputs], 
                                                        model.test_labels))
print("Saving trained model.")
model_path = './models/test/test_model'
model.tfq_model.save(model_path)

# Plotting results.
print("Plotting validation accuracy")
plt.plot(history.history['val_loss'], label='qml_model')
plt.title('QML model performance')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
path = './img/val_acc-'
path += 'v-qubits:' + str(model.n_var_qubits)
path += '_v-depth:' + str(model.var_depth) 
path += '_reuploads:' + str(model.n_reuploads)
path += '_intermediate_readouts:' + str(model.intermediate_readouts)
path += '.png'
plt.savefig(path)
plt.close()
