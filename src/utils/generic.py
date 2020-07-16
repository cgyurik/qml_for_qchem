'''Generic utilities without requirements'''

import numpy as np


def encode_complex_and_array(obj):
    """
    supplement for json.dump to be able to deal with complex numbers
    and np.arrays.
    """
    if isinstance(obj, np.ndarray):
        return(obj.tolist())

    if isinstance(obj, complex):
        return str(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def chop(array, abs_tol=1E-12):
    """
    Set to zero numerical zeros (values below abs_tol) in a np.array.
    Acts in-place.
    """
    array[abs(array) < abs_tol] = 0
    return array
