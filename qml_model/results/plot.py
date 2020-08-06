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
## Reading .json files.
dataset = []
print("  - reading .json files.")
for filename in os.listdir(JSON_DIR):
    if filename.endswith('.json'):
        datapoint = load_data(filename)
        dataset.append(datapoint) 
        #print("    * read molecule", len(dataset), ".")

# Setting up the model.
print("-----Setting up model-----")
model = tfq_model(n_var_qubits=2, var_depth=2, n_reuploads=2, 
                    intermediate_readouts=True, 
                    processed_data='H4_dataset_processed_501_parallel',
                    print_summary=True, plot_to_file=False)

print("Obtaining HF energies")
test_hfe = []
for j in range(len(model.test_labels)):
    index = [i for i in range(len(dataset)) if dataset[i]['exact_energy']==model.test_labels[j]][0]
    test_hfe.append(dataset[index]['hf_energy'])

print("Compiling model.")
model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
print("Loading weights.")
model.tfq_model.load_weights('./results/parallel/parallel_experiment_1/cp-10.ckpt')
print("Evaluating model.")
test_predictions1 = model.tfq_model.predict([model.test_groundstates, model.test_classical_inputs], verbose=1)
print("-----Setting up model-----")
model = tfq_model(n_var_qubits=3, var_depth=5, n_reuploads=1, 
                    intermediate_readouts=True, 
                    processed_data='H4_dataset_processed_501_parallel',
                    print_summary=True, plot_to_file=False)

print("Compiling model.")
model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
print("Loading weights.")
model.tfq_model.load_weights('./results/parallel/parallel_experiment_0/cp-15.ckpt')
print("Evaluating model.")
test_predictions0 = model.tfq_model.predict([model.test_groundstates, model.test_classical_inputs], verbose=1)
a = plt.axes(aspect='equal')
plt.scatter(model.test_labels, test_predictions1, label = 'Predicted by QML model w/ 2 reups')
plt.scatter(model.test_labels, test_predictions0, label = 'Predicted by QML model w/ 1 reups')
plt.scatter(model.test_labels, test_hfe, label = 'HF energy', marker='x')
plt.xlabel('True GS energies (FCI) [Ha]')
plt.ylabel('Energy by other method [Ha]')
lims = plt.gca().get_ylim()
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.legend()
plt.show()
"""

"""
indices = ['01', '02', '03', '04', '05', '06', '07']
train_loss1 = []
validation_loss1 = []

# Setting up the model.
print("-----Setting up model-----")
model = tfq_model(n_var_qubits=2, var_depth=1, n_reuploads=3, 
                    intermediate_readouts=True, 
                    processed_data='H4_dataset_processed_501_parallel_only-geometry',
                    print_summary=True, plot_to_file=False)
print("Compiling model.")
model.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
for i in range(len(indices)):
    print("Loading weights.")
    model.tfq_model.load_weights('./results/parallel/parallel_experiment_extra/cp-' + indices[i] +'.ckpt')
    print("Evaluating model.")
    tr = model.tfq_model.evaluate(x=[model.train_groundstates, model.train_classical_inputs], 
                                    y=model.train_labels, verbose=2)
    val = model.tfq_model.evaluate(x=[model.test_groundstates, model.test_classical_inputs], 
                                    y=model.test_labels, verbose=2)
    print("Train loss:", tr)
    print("Val loss:", val)
    train_loss1.append(tr)
    validation_loss1.append(val)

print("Saving results trained model.")
losses = [train_loss1, validation_loss1]
pickle_path = "./results/parallel/parallel_experiment_extra/losses.p"
with open(pickle_path, 'wb') as f:      
    pickle.dump(losses, f)

"""

train_losses = []
validation_losses = []
for i in range(2):
    path = './results/parallel/parallel_experiment_' + str(i) + '/losses.p'
    with open(path, 'rb') as f:
        losses = pickle.load(f)
    train_losses.append(losses[0])
    validation_losses.append(losses[1])

path = './results/parallel/parallel_experiment_extra/losses.p'
with open(path, 'rb') as f:
    losses = pickle.load(f)
train_losses.append(losses[0])
val_loss = losses[1]
val_loss.append(0.4662)
val_loss.append(0.3854)
val_loss.append(0.3217)
validation_losses.append(val_loss)

fig, ax = plt.subplots()
ax.plot(list(range(1, 11)), validation_losses[0][:10], label='1 upload')
ax.plot(list(range(1, 11)), validation_losses[1][:10], label='2 uploads')
ax.plot(list(range(1, 10)), validation_losses[2][:9], label='3 uploads')
ax.set(xlabel='epochs', ylabel='mse validation loss',
       title='QML model performance')
ax.grid()
#plt.yticks(np.arange(0, 1, step=0.05))
#plt.ylim(0, 1)
plt.xlim(1, 10)
plt.legend()
#p = './results/parallel/validation_loss.png'
#plt.savefig(p)
plt.show()
#plt.close()
"""
fig, ax = plt.subplots()
ax.plot(list(range(1, 11)), train_losses[0][:10], label='1 upload')
ax.plot(list(range(1, 11)), train_losses[1][:10], label='2 uploads')
ax.set(xlabel='epochs', ylabel='mse train loss',
       title='QML model training convergence')
ax.grid()
plt.yticks(np.arange(0, 1, step=0.05))
plt.ylim(0, 1)
plt.xlim(1, 10)
plt.legend()
p = './results/parallel/train_loss.png'
plt.savefig(p)
plt.close()
#plt.show()
"""
