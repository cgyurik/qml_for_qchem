""""Initialize init"""

from .generic import (
    encode_complex_and_array
)
from .data_utils.load_lib import (
    MOLECULES_DIR, UCC_DIR, JSON_DIR, DATA_DIR,
    load_data,
    load_ucc_data
)

# If modules required to generate the molecules are not installed, raise the
# exception only if the user tries to generate molecules.
try:
    from .generate_lib import (
        H4_generate_random_molecule,
        FailedGeneration,
        DMIN,
        DMAX
    )
except ModuleNotFoundError as err:
    _error_message = err.msg  # type: ignore

    def H4_generate_random_molecule():  # type: ignore
        '''
        This function is disabled as a required module is not installed

        Raises: ModuleNotFoundError
        '''
        raise ModuleNotFoundError(
            _error_message
            + '. This module is required for generating molecules.')


__all__ = (
    'encode_complex_and_array',
    'H4_generate_random_molecule',
    'FailedGeneration',
    'DMIN',
    'DMAX',
    'MOLECULES_DIR',
    'UCC_DIR',
    'JSON_DIR',
    'DATA_DIR',
    'load_data',
    'load_ucc_data'
)
