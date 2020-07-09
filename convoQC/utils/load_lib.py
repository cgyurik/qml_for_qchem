"""Functions to load molecules."""
import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # manage relative paths
DATA_DIR = os.path.realpath(BASE_DIR + '/data/')

MOLECULES_DIR = DATA_DIR + 'molecules/'
JSON_DIR = DATA_DIR + 'json/'
UCC_DIR = DATA_DIR + 'ucc/'


def load_data(filename):
    """
    Load molecule data dictionary for the given filename.
    Molecule data dictionaries contain the molecular data that are:
        - Relevant for the QML model;
        - Independent on OpenFermion (or any other external package);
        - Independent on the VQE ansatz chosen to prepare approximate ground
            states.

    If filename is a relative path, it is assumed to be in this
    repository's directory data/json/

    Raises:
        FileNotFoundError if file does not exist
    """
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
            data_dict.update(
                {k: np.reshape([complex(s) for s in v.flatten()], v.shape)})
            continue
        # all other fields should be arrays
        data_dict[k] = np.array(v)
    return data_dict


def load_ucc_data(filename):
    """
    Load UCC data dictionary for the given filename.
    UCC data dictionaries contain the data for the given molecule that are
    relative to the UCC ansatz used to prepare approximate ground states.

    If filename is a relative path, it is assumed to be in this
    repository's directory data/json/

    Raises:
        FileNotFoundError if file does not exist
    """
    if not filename.endswith('.json'):
        filename = filename + '.json'
    if not filename.startswith('/'):
        filename = UCC_DIR + filename
    if not os.path.exists(filename):
        raise FileNotFoundError(filename + ' not found.')
    with open(filename, 'rt') as f:
        data_dict = json.load(f)
    for k, v in data_dict.items():
        if k == 'params':
            v = np.array(v)
            data_dict.update({k: np.array(v)})
        if k == 'optimizer_success':
            data_dict.update({k: bool(v)})
        # other keys: 'energy_expval', 'energy_error', 'overlap'
    return data_dict
