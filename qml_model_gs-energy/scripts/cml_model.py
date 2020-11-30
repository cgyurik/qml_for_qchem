import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os 
import sys
sys.path.append(os.path.abspath('..'))
from utils import load_data, JSON_DIR


def flatten_H4_geometry(geometry):
    positions = np.array(list(zip(*geometry))[1]) # discard the atom names
    return [positions[i][j] for i in range(1, 4) for j in range(0, i)]

def extract_input_output(dataset):
    return map(np.array, list(zip(*(
        (flatten_H4_geometry(d['geometry']) + d['orbital_energies'].tolist(),
         d['exact_energy'])
        for d in dataset))))

def split_train_test(dataset, training_fraction=0.8):
    rng = np.random.default_rng()
    shuffled_set = rng.permutation(dataset)
    train_set = shuffled_set[:int(training_fraction*len(dataset))]
    test_set = shuffled_set[int(training_fraction*len(dataset)):]
    
    return train_set, test_set

def build_model(layer_size = 8, depth = 2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer_size, activation='relu',
                              input_shape=[len(train_norm_input[0])])
    ] + [
        tf.keras.layers.Dense(layer_size, activation='relu')
        for _ in range(depth - 1)
    ] + [
        tf.keras.layers.Dense(1)
    ])

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.05,
      decay_steps=len(train_output)*10,
      decay_rate=1,
      staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

def naive_training(model, max_epochs=500):
    history = model.fit(
      train_norm_input, train_output,
      epochs=max_epochs, validation_split = 0.2, verbose=0)
          
files = os.listdir(JSON_DIR)
dataset = [load_data(JSON_DIR + file) for file in files]
train_set, test_set = split_train_test(dataset)
train_input, train_output = extract_input_output(train_set)
test_input, test_output = extract_input_output(test_set)
means = np.mean(train_input, axis=0)
stds = np.std(train_input, axis=0)
def normalize_input(x):
    return (x - means) / stds
train_norm_input = normalize_input(train_input)
test_norm_input = normalize_input(test_input)

model = build_model()
naive_training(model)
model.save_weights("../results/simple_CML_model/final_weights")
