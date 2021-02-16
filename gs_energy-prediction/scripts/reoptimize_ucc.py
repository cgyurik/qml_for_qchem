import os
import sys
import json

# pylint: disable=wrong-import-position
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from convoQC.scripts.optimize_ucc import optimize_ucc
from convoQC.utils import load_ucc_data, DATA_DIR, encode_complex_and_array
# pylint: enable=wrong-import-position

UCC_REOPT_DIR = DATA_DIR + 'ucc_reoptimized/'

usage = ('Usage: python {} <target_filename> <init_params_filename>'
         .format(sys.argv[0]))
if len(sys.argv) is not 3:
    raise Exception(usage)
if not (isinstance(sys.argv[1], str) and isinstance(sys.argv[1], str)):
    raise TypeError('The first argument is not a string.\n' + usage)
target_filename = sys.argv[1]
source_filename = sys.argv[2]

existing_ucc_reoptimized_files = os.listdir(UCC_REOPT_DIR)
if (target_filename + '.json' in existing_ucc_reoptimized_files):
    print('The file data/ucc_reoptimized/{}.json exists already. Exiting.')
    exit()

source_ucc_dict = load_ucc_data(source_filename)
init_params = source_ucc_dict['params']

target_ucc_dict = optimize_ucc(target_filename, init_params)

print('saving data to file.')
with open(UCC_REOPT_DIR + target_filename + '.json', 'wt') as f:
    json.dump(target_ucc_dict, f, default=encode_complex_and_array)

print(*((k, v) for k, v in target_ucc_dict.items()), sep='\n')
