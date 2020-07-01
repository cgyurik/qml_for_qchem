import os
import numpy as np

import json

DATA_DIR = os.path.realpath('../data')
MOLECULES_DIR = DATA_DIR + '/molecules/'
JSON_DIR = DATA_DIR + '/json/'


def load_data(filename):
    '''
    Returns data dictionary relative to the input filename, if the data file 
    exists. If filename is a relative path, it is assumed to be in this 
    repository's directory data/json/
    
    Raises:
        FileNotFoundError if file does not exist
    '''
    if not filename.endswith('.json'):
        filename = filename + '.json'
    if not filename.startswith('/'):
        filename = JSON_DIR + filename
    if not os.path.exists(filename):
        raise FileNotFoundError(filename + ' not found.')
    with open(filename, 'rt') as f:
        data_dict = json.load(f)
    for k, v in data_dict.items():
        if k in ['geometry', 'multiplicity', 'exact_energy', 'hf_energy']:
            continue
        if k == 'ground_states':
            v = np.array(v)
            data_dict.update({k: np.reshape([complex(s) for s in v.flatten()], v.shape)})
            continue
        # all other fields should be arrays
        data_dict[k] = np.array(v)
    return data_dict