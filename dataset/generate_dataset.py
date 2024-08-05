import itertools as it
from matplotlib import gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import qutip
import scipy as sp
from tqdm.auto import tqdm


import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(path)
sys.path.append(parent)

from argparse import ArgumentParser
from configs.utils import parse_config
from modelling import *
import warnings
warnings.filterwarnings("ignore")



def get_args():
    parser = ArgumentParser(description="Generate 2D Heisenberg dataset")
    parser.add_argument('--config', type=str, help='config')
    args = parser.parse_args()

    return args

def sample_coupling_matrix(rows, cols, rng):
    qubits = rows * cols
    
    # Create a 2D Lattice
    edges = [
        (si, sj) for (si, sj) in it.combinations(range(qubits), 2)
        if ((sj % cols > 0) and sj - si == 1) or sj - si == cols
    ]
    
    # sample edge weights uniformly at random from [0, 2]
    edge_weights = rng.uniform(0, 2, size=len(edges))
    
    coupling_matrix = np.zeros((qubits, qubits))
    for (i, j), w in zip(edges, edge_weights):
        coupling_matrix[i, j] = coupling_matrix[j, i] = w
        
    return coupling_matrix

def build_hamiltonian(coupling_matrix):
    coeffs, ops = [], []
    ns = coupling_matrix.shape[0]

    for i, j in it.combinations(range(ns), r=2):
        coeff = coupling_matrix[i, j]
        if coeff:
            for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(coeff)
                ops.append(op(i) @ op(j))

    return qml.Hamiltonian(coeffs, ops)

#create guided state
def create_guided_State(ground_state, ortho_state, epsilon=0.05):
    return np.sqrt(1-epsilon) * ground_state + np.sqrt(epsilon)*ortho_state

def sample_shadow(statevector, wires, shots, device_name='default.qubit'):
    """ sample classical shadows for the state described by the statevector
    The resulting shadows are encoded as integer according to the logic
    0,1 -> +,- in the X Basis
    2,3 -> r,l in the Y Basis
    4,5 -> 0,1 in the Z Bassi
    """
    @qml.qnode(device=qml.device(device_name, wires=wires, shots=shots), diff_method=None, interface=None)
    def shadow_measurement():
        qml.QubitStateVector(statevector, wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    bits, recipes = shadow_measurement()

    # encode measurements and bases as integers
    data = 2 * recipes + bits
    data = np.array(data, dtype=int)

    return data

def main(config):
    rng = np.random.default_rng()

    # Generate train data
    for i in range(config.NUM_TRAIN):
        if i%10 == 0:
            print("[INFO] Processing {}/{}".format(i, config.NUM_TRAIN))

        #create sample directory
        sample_dir = os.path.join(config.TRAIN_DIR, "sample_{}".format(i+1))
        os.makedirs(sample_dir)

        coupling_matrix = sample_coupling_matrix(config.LATTICES[0], config.LATTICES[1], rng) 
        # build hamiltonian
        H = build_hamiltonian(coupling_matrix)
        # Hmat= H.sparse_matrix()
        H_sparse = H.sparse_matrix()

        # diagonalize the Hamiltonian to get the ground state and guided state
        eigvals, eigvecs = sp.sparse.linalg.eigs(H_sparse, which='SR', k=2)
        eigvals = eigvals.real
        ground_state = eigvecs[:, np.argmin(eigvals)]
        ground_state = ground_state/np.linalg.norm(ground_state)
        ortho_state = eigvecs[:, np.argmax(eigvals)]
        ortho_state = ortho_state/np.linalg.norm(ortho_state)
        guided_state = create_guided_State(ground_state, ortho_state, config.EPSILON)
        guided_state = guided_state/np.linalg.norm(guided_state)

        # generate classical shadow of the ground state
        wires = config.LATTICES[0]* config.LATTICES[1]
        #classical_shadow_data = sample_shadow(ground_state, wires=wires, shots=config.SHOTS)

        #np.save(os.path.join(sample_dir, "classical_shadow_data.npy"), classical_shadow_data)
        np.save(os.path.join(sample_dir, "coupling_matrix.npy"), coupling_matrix)
        np.save(os.path.join(sample_dir, "ground_state.npy"), ground_state)
        np.save(os.path.join(sample_dir, "guided_state.npy"), guided_state)
    
    print("[INFO] Done Train Data")
    # Generate test data
    for i in range(config.NUM_TEST):
        if i%10 == 0:
            print("[INFO] Processing {}/{}".format(i, config.NUM_TEST))

        #create sample directory
        sample_dir = os.path.join(config.TEST_DIR, "sample_{}".format(i+1))
        os.makedirs(sample_dir)

        coupling_matrix = sample_coupling_matrix(config.LATTICES[0], config.LATTICES[1], rng) 
        # build hamiltonian
        H = build_hamiltonian(coupling_matrix)
        # Hmat= H.sparse_matrix()
        H_sparse = H.sparse_matrix()

        # diagonalize the Hamiltonian to get the ground state and guided state
        eigvals, eigvecs = sp.sparse.linalg.eigs(H_sparse, which='SR', k=2)
        eigvals = eigvals.real
        ground_state = eigvecs[:, np.argmin(eigvals)]
        ground_state = ground_state/np.linalg.norm(ground_state)
        ortho_state = eigvecs[:, np.argmax(eigvals)]
        ortho_state = ortho_state/np.linalg.norm(ortho_state)
        guided_state = create_guided_State(ground_state, ortho_state, config.EPSILON)
        guided_state = guided_state/np.linalg.norm(guided_state)

        # generate classical shadow of the ground state
        wires = config.LATTICES[0]* config.LATTICES[1]
        #classical_shadow_data = sample_shadow(ground_state, wires=wires, shots=config.SHOTS)

        #np.save(os.path.join(sample_dir, "classical_shadow_data.npy"), classical_shadow_data)
        np.save(os.path.join(sample_dir, "coupling_matrix.npy"), coupling_matrix)
        np.save(os.path.join(sample_dir, "ground_state.npy"), ground_state)
        np.save(os.path.join(sample_dir, "guided_state.npy"), guided_state)
    print("[INFO] Done Test Data")



if __name__ == "__main__":
    args = get_args()
    config = parse_config(args.config)
    main(config)