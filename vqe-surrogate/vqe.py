## os/sys tools
import os, sys, itertools, pickle
# disable terminal warning tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
## general tools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cirq
from scipy.optimize import minimize
## vqe/qml tools.
import openfermion
import tensorflow_quantum as tfq
from vqe_utils.uccsd import UCCSDAnsatz
from qml_model.tfq_model import tfq_model
## visualization tools
import matplotlib.pyplot as plt

class vqe():
    """
    Attributes: 
    - ...
    """
    def __init__(self, filename, n_uploads=1, var_depth=1, verbose=False):
        print("-----Initializing VQE with surrogate-----")
        """ 
        Loading molecule
        """
        self.molecule = openfermion.MolecularData(filename=filename)
        self.molecule.load()
     
        ## Constructing qubit Hamiltonian        
        self.hamiltonian = openfermion.jordan_wigner(
                                openfermion.get_fermion_operator(self.molecule.get_molecular_hamiltonian())
                                                    )
        self.true_gs_energy = openfermion.eigenspectrum(self.hamiltonian)[0]

        """
        Setting up VQE
        """
        ## Constructing UCCSD ansatz
        self.qubits = cirq.GridQubit.rect(1, self.molecule.n_qubits)
        self.ansatz = UCCSDAnsatz(self.molecule, self.qubits)
    
        ## Setting up initial parameters.
        self.params = np.zeros(len(self.ansatz.symbols))

        ## If verbose, report all settings of the vqe including surrogate.
        if verbose:
            print("Molecule name:", self.molecule.name)
            #print("Corresponding Hamiltonian:")
            #print(self.hamiltonian)    
            print('Number of parameters: {}'.format(len(self.ansatz.symbols)))
            #print("Ansatz circuit:")
            #print(self.ansatz.circuit.to_text_diagram(transpose=True))
            print_circuit, print_summary, plot = True, True, True
        else:
            print_circuit, print_summary, plot = False, False, False
        
        
        """
        Constructing diagonal observable
        """
        # Constructing readout of diagonal matrix.
        diagonal_matrix = np.diag(openfermion.eigenspectrum(self.hamiltonian))
        
        # Decomposing diagonal matrix in Pauli basis 'by hand'.
        test = np.zeros((2**8, 2**8))
        decomposition = []
        for pauli_product in itertools.product(cirq.PAULI_BASIS.keys(), repeat=8):
            # diagonal matrix so can skip X's and Y's.
            if 'X' in pauli_product or 'Y' in pauli_product:
                continue
            
            # Computing the pauli matrix
            pauli_matrix = cirq.PAULI_BASIS[pauli_product[0]]
            for i in range(1, len(pauli_product)):
                pauli_matrix = np.kron(pauli_matrix, cirq.PAULI_BASIS[pauli_product[i]])
            # Computing coefficient of pauli_string
            coef = (1/ (2**8)) * np.trace(diagonal_matrix @ pauli_matrix)
            test += pauli_matrix*coef
            # Constructing pauli string object in cirq
            pauli_map = {}
            for i in range(len(list(pauli_product))):
                pauli_map[self.qubits[i]] = pauli_product[i]
            pauli_string_obj = cirq.PauliString(pauli_map, coefficient=coef)
            decomposition.append(pauli_string_obj)
        
        # Constructing readout PauliSum operator.
        diag_readouts = cirq.PauliSum.from_pauli_strings(decomposition)
        
        """
        Setting up surrogate
        """
        # Constructing surrogate model.
        self.surrogate = tfq_model(qubits=self.qubits, readouts=diag_readouts, 
                                    n_uploads=n_uploads, var_depth=var_depth,
                                    print_circuit=print_circuit, print_summary=print_circuit, plot=plot)
        self.surrogate.tfq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mse)
    
        ## Setting up 'history' trainingset
        self.eval_history = []


    """
    Cost of the VQE (i.e., expected energy)
    """
    def vqe_cost(self, params=None):
        # Setting up parameters        
        if params is None:
            params = self.params
        
        # Setting up parameter mapping for symbols.
        param_mapping = [(self.ansatz.symbols[i], params[i]) for i in range(len(self.ansatz.symbols))]
        resolve_dict = dict(param_mapping)

        # Resolving the symbols in the PQC.
        resolver = cirq.ParamResolver(resolve_dict)
        resolved_ansatz = cirq.resolve_parameters(self.ansatz.circuit, resolver)

        # Run the circuit and estimate probabilites.
        final_state = cirq.final_wavefunction(resolved_ansatz)
        
        """
        # Testing tensorable circuit.
        test_circuit = self.ansatz.tensorable_ucc_circuit(params, self.qubits)
        test_final_state = cirq.final_wavefunction(test_circuit)
        print("Fidelity of output both circuits:", abs(np.vdot(final_state, test_final_state))**2)
        """

        # Computing energy of output PQC.
        energy = openfermion.expectation(openfermion.get_sparse_operator(self.hamiltonian), final_state).real

        # Saving evaluation for surrogate training.
        self.eval_history.append({"params": params, "energy":energy})

        return energy

    """
    Use tfq_model predictions as surrogate cost function
    """
    def surrogate_cost(self, params):
        input_circuit = self.ansatz.tensorable_ucc_circuit(params, self.qubits)
        input_tensor = tfq.convert_to_tensor([input_circuit])
        return self.surrogate.tfq_model.predict(input_tensor, verbose=0)[0][0]

    """ 
    Train using vanilla VQE approach
    """
    def train_standard(self, n_experiments=1):
        print("----- Training VQE using standard cost function-----")
        vqe_trajectories = []
        for i in range(n_experiments):
            print("  - experiment:", i+1, "/", n_experiments)
            vqe_i_trajectory = []
            def vqe_store(x):
                vqe_i_trajectory.append(self.vqe_cost(x))
                print("    - Finished iteration", len(vqe_i_trajectory), ".")
            # Minimizing vqe cost.
            self.params = minimize(self.vqe_cost, x0=self.params, method="Nelder-Mead", callback=vqe_store).x
            vqe_trajectories.append(vqe_i_trajectory)
            # Reinitializing VQE for next experiment.
            self.params = np.zeros(len(self.ansatz.symbols))
            self.eval_history = []
            

        # Averaging results
        len_trajectories = [len(vqe_trajectories[i]) for i in range(n_experiments)]
        min_len_trajectory = min(n_evals)
        avg_energies = []        
        sd_energies = []
        for i in range(min_len_trajectory):
            eval_i_energies = []
            for j in range(n_experiments):
                iteration_i_energies.append(vqe_trajectory[j][i])
            avg_energies.append(np.mean(iteration_i_energies))
            sd_energies.append(np.std(iteration_i_energies))

        # Plotting results.
        fig, ax = plt.subplots()
        ax.plot(range(min_len_trajectory), avg_energies)
        lower = np.array(avg_energies) - np.array(sd_energies)
        upper = np.array(avg_energies) + np.array(sd_energies)
        ax.fill_between(range(min_len_trajectory), upper, lower, alpha = 0.5)
        ax.set(xlabel='function evals', ylabel='energy', title='Vanilla VQE convergence')
        ax.axhline(y=self.true_gs_energy, label='true groundstate energy')
        plt.savefig('./vanilla_vqe.png')
        plt.close()

        
## Test functions.
if __name__ == "__main__":  
    # Setting up a VQE instance.   
    filename = "./molecules/molecule1"
    vqe_test = vqe(filename, verbose=True)
    
    
    
"""
    ## Train VQE using surrogate.
    def train_with_surrogate(self, vqe_maxiter=25, vqe_maxfev=250, surr_maxiter=10, epochs=25, n_iterations=20):
        print("----- Training VQE using surrogate-----")
        vqe_trajectories = []
        surrogate_trajectories = []
        for i in range(n_iterations):
            ## Training VQE using the standard VQE cost function.
            print("Training with standard VQE cost function")
            vqe_i_trajectory = []
            def vqe_store(x):
                vqe_i_trajectory.append(self.vqe_cost(x))
                print("    - Finished iteration", len(vqe_i_trajectory))
            self.params = minimize(self.vqe_cost, x0=self.params, method="Nelder-Mead", callback=vqe_store,
                                                    options={'maxiter':vqe_maxiter, 'maxfev':vqe_maxfev}).x
            print("  - new parameters:", self.params)
            print("  - completed", len(vqe_i_trajectory), "iterations using standard VQE cost function.")
            vqe_trajectories.append(vqe_i_trajectory)

            ## Training the surrogate.
            print("Switching to surrogate")
            print("  - loading the data")
            self.surrogate.load_data(self.eval_history, self.ansatz, split=1)
            history = self.surrogate.tfq_model.fit(x=self.surrogate.train_states, 
                                            y=self.surrogate.train_labels,
                                            batch_size=32,
                                            epochs=epochs,
                                            verbose=1)
            ## Training VQE using the surrogate as cost function.            
            print("   - using the surrogate as cost function")
            surrogate_i_trajectory = []
            def surrogate_store(x_0):
                surrogate_i_trajectory.append(self.vqe_cost(x_0))
            self.params = minimize(self.surrogate_cost, x0=self.params, method="Nelder-Mead",
                                    callback=surrogate_store, options={'maxiter':surr_maxiter}).x
            print("  - new parameters:", self.params)
            print("  - completed", len(surrogate_i_trajectory), "iterations using surrogate cost function.")
            surrogate_trajectories.append(surrogate_i_trajectory)


        ## Plotting results.
        fig, ax = plt.subplots()           
        n_previous_evals = 0         
        for i in range(n_iterations):
            # Plotting iterations that used standard VQE evals (green)
            xvals = range(n_previous_evals, n_previous_evals + len(vqe_trajectories[i]))
            n_previous_evals += len(vqe_trajectories[i])
            ax.plot(xvals, vqe_trajectories[i], color='green')
            # Plotting iterations that used surrogate evals (blue)
            xvals = range(n_previous_evals, n_previous_evals + len(surrogate_trajectories[i]))
            n_previous_evals += len(surrogate_trajectories[i])
            ax.plot(xvals, surrogate_trajectories[i], color='blue')
        ax.set(xlabel='function evals', ylabel='energy', title='VQE w/ surrogate convergence')
        ax.axhline(y=self.true_gs_energy, label='true groundstate energy')
        plt.savefig('./vqe_with_surrogate.png')
        plt.close()
"""
