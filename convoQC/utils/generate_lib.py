"""Functions to generate molecule library."""
import os
import json
from itertools import combinations
import numpy as np

from scipy.sparse.linalg import eigsh
import scipy.linalg

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_sparse_operator
from openfermionpsi4 import run_psi4

from .load_lib import MOLECULES_DIR, JSON_DIR, load_data
from .generic import encode_complex_and_array, chop


DMIN = 0.4
DMAX = 1.5
ROUNDING = 4
DEFAULT_RNG = np.random.default_rng()


class FailedGeneration(Exception):
    pass


#  pylint: disable = undefined-variable
class MoleculeDataGenerator:
    """
    Class to generate the MolecularData object with the right multiplicity,
    extract the relevant data for our QML model, and save them in the right
    directory.

    For now, the code only works for neutral molecules with an even number of
    electron and singlet/triplet ground state (the molecule family we chose for
    this test is H4).

    Raises:
        FailedGeneration: If an exception is raised (e.g. by openfemion) during
        molecule or data generation. Eventual generated files are removed
        first.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self.filename = self._generate_filename()
        m_file = MOLECULES_DIR + self.filename + '.hdf5'
        j_file = JSON_DIR + self.filename + '.json'
        if os.path.exists(m_file):
            self.molecule = MolecularData(filename=m_file)
            self.molecule.load()
            self._solve_ground_states()
        else:
            try:
                self._generate_molecule_unknown_multiplicity()
            except Exception as exc:
                print('Exception during molecule generation for: \n\t'
                      + self.filename + '\nCleaning up eventual files.')
                self._clean_up_files()
                raise FailedGeneration(exc)

        if os.path.exists(j_file):
            self.data_dict = load_data(j_file)
        else:
            try:
                self._generate_data()
            except Exception as exc:
                print('Exception during data dictionary generation for: \n\t'
                      + self.filename + '\nCleaning up eventual files.')
                self._clean_up_files()
                raise FailedGeneration(exc)

    def _generate_filename(self):
        """
        Univocally generates a filename from the geometry.

        The filename has structure (variables values indicated as <var>):
        <atom0>,<x0>,<y0>,<z0>;<atom1>,<x0>,<y0>,<z0>;<...>
        where <atom> is the atomic symbol (one or two letters) and numerical
        values <xi>,<yi>,<zi> are represented without any trailing zero.
        """
        return ((str(self.geometry) + '.EXT')
                .replace(' ', '')
                .replace(']', ')')
                .replace('[', '')
                .replace(')),', ';')
                .replace(')', '')
                .replace('(', '')
                .replace("'", '')
                .replace('.0,', ',')
                .replace('.0;', ';')
                .replace('.0.EXT', '.EXT')
                .replace('.EXT', ''))

    def _clean_up_files(self):
        if os.path.exists(MOLECULES_DIR + self.filename + '.hdf5'):
            os.remove(MOLECULES_DIR + self.filename + '.hdf5')
        if os.path.exists(JSON_DIR + self.filename + '.json'):
            os.path.exists(JSON_DIR + self.filename + '.json')

    def _solve_ground_states(self):
        exact_energies, self.ground_states = eigsh(
            get_sparse_operator(self.molecule.get_molecular_hamiltonian()),
            k=self.molecule.multiplicity, which='SA'
        )
        self.exact_energy = exact_energies[0]
        chop(self.ground_states)

    def _generate_molecule_unknown_multiplicity(self):
        """
        Generate the right GS-multiplicity molecule and its ground states.

        Raises:
            Exception if neither singlet not triplet work.
        """
        # generate singlet and diagonalize sparse hamiltonian
        self._generate_molecule(multiplicity=1)
        self._solve_ground_states()

        # if singlet FCI energy matches exact diagonalization, return
        if np.isclose(self.molecule.fci_energy, self.exact_energy):
            return

        # else, try the same with triplet
        self._generate_molecule(multiplicity=3)
        self._solve_ground_states()

        if np.isclose(self.molecule.fci_energy, self.exact_energy):
            return

        # if neither works, raise an exception
        raise Exception('Neither singlet nor tripet FCI energy '
                        'match exact ground state energy')

    def _generate_molecule(self, multiplicity):
        m_file = MOLECULES_DIR + self.filename
        self.molecule = MolecularData(geometry=self.geometry,
                                      basis='STO-3G',
                                      multiplicity=multiplicity,
                                      charge=0,
                                      filename=m_file)
        self.molecule = run_psi4(self.molecule,
                                 run_scf=True,
                                 run_fci=True,
                                 delete_input=True,
                                 delete_output=True)

    def _generate_data(self):
        # generate unitary that transforms molecular orbitals to Orthogonal
        # Atomic Orbitals (OAO)
        M = self.molecule.canonical_orbitals
        P = scipy.linalg.inv(M)
        canonical_to_oao = (P @ scipy.linalg.sqrtm(M @ M.T))

        self.data_dict = dict(
            geometry=self.molecule.geometry,
            multiplicity=self.molecule.multiplicity,
            canonical_orbitals=self.molecule.canonical_orbitals,
            canonical_to_oao=canonical_to_oao,
            orbital_energies=self.molecule.orbital_energies,
            exact_energy=self.exact_energy,
            ground_states=self.ground_states,
            hf_energy=self.molecule.hf_energy[()]
        )

        with open(JSON_DIR + self.filename + '.json', 'wt') as f:
            json.dump(self.data_dict, f, default=encode_complex_and_array)


# *** H4 family ***


def check_geometry(geometry):
    """
    Check that the geometry respects the rules:
        - minimum distance of atom pairs (see DMIN for value)
        - maximum distance of adjacent atoms (see DMAX for value)
    """
    _, positions = list(zip(*geometry))
    for (i, pA), (j, pB) in combinations(enumerate(positions), 2):
        dist = np.sqrt(np.sum((np.array(pA) - np.array(pB))**2))
        if dist < DMIN:  # no pair closer than dmin
            return False
        if j == i + 1 and dist > DMAX:  # no adjacent pair farther than dmax
            return False
    return True


def H4_generate_random_geometry(rng=DEFAULT_RNG):
    '''
    Generate random geometry for H4 of form
    (
        ('H', (0 , 0 , 0 )),
        ('H', (x1, 0 , 0 )),
        ('H', (x2, y2, 0 )),
        ('H', (x3, y3, z3))
    )
    '''
    geometry = [('H', (0., 0., 0.))]
    x1 = round(rng.uniform(DMIN, DMAX), ROUNDING)
    geometry.append(('H', (x1, 0., 0.)))
    r2 = rng.uniform(DMIN, DMAX)
    theta2 = rng.uniform(0, 2 * np.pi)
    x2 = round(np.cos(theta2) * r2, ROUNDING)
    y2 = round(np.sin(theta2) * r2, ROUNDING)
    geometry.append(('H', (x2, y2, 0.)))
    phi3 = np.arcsin(rng.uniform(-1, 1))
    theta3 = rng.uniform(0, 2 * np.pi)
    r3 = rng.uniform(DMIN, DMAX)
    x3 = round(r3 * np.cos(phi3) * np.cos(theta3), ROUNDING)
    y3 = round(r3 * np.cos(phi3) * np.sin(theta3), ROUNDING)
    z3 = round(r3 * np.sin(phi3), ROUNDING)
    geometry.append(('H', (x3, y3, z3)))
    return geometry


def H4_generate_valid_geometry(rng=DEFAULT_RNG):
    """
    Generate valid random geomertry for H4.
    For detailed help see:
        `H4_generate_random_geometry`
        `check_geometry`
    """
    while True:
        geometry = H4_generate_random_geometry(rng)
        if check_geometry(geometry):
            return geometry
        else:
            # print('Invalid geometry. Retrying...')
            pass


def H4_generate_random_molecule(rng=DEFAULT_RNG):
    """
    Generate and save molecule and data for a valid random geomertry of H4.
    For detailed help see:
        `H4_generate_valid_geometry`
        `check_geometry`
        `MoleculeDataGenerator`

    Args:
        rng: a numpy.random generator

    Returns:
        MoleculeDataGenerator

    Raises:
        FailedGeneration
    """
    return MoleculeDataGenerator(H4_generate_valid_geometry(rng))
