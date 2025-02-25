import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
import yaml 
import pprint
import argparse
from pathlib import Path
from periodictable import elements
from scipy.spatial.transform import Rotation as R
from scm.plams import Molecule
from CAT.recipes import replace_surface
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def compute_rmsd_matrix(structures1, structures2=None):
    """
    Compute the RMSD matrix for pairs of structures efficiently.

    Parameters:
        structures1 (list or np.ndarray): Array of shape (N1, num_atoms, 3)
                                          representing the first set of structures.
        structures2 (list or np.ndarray): Array of shape (N2, num_atoms, 3)
                                          representing the second set of structures.
                                          If None, computes the RMSD matrix within structures1.

    Returns:
        np.ndarray: RMSD matrix of shape (N1, N2).
    """
    # Ensure we have NumPy arrays
    structures1 = np.array(structures1)
    if structures2 is None:
        structures2 = structures1
        symmetric = True
    else:
        structures2 = np.array(structures2)
        symmetric = False

    N1, A, _ = structures1.shape
    N2 = structures2.shape[0]

    if symmetric:
        # Compute only upper triangle and mirror it to avoid redundant computation
        rmsd_matrix = np.zeros((N1, N2), dtype=float)
        for i in range(N1):
            # Only compute from j = i to the end
            diff = structures2[i:] - structures1[i]  # shape: (N2 - i, A, 3)
            # sum of squared differences per structure
            sq_diff_sum = np.sum(diff**2, axis=(1, 2)) / A
            rmsd_vals = np.sqrt(sq_diff_sum)
            rmsd_matrix[i, i:] = rmsd_vals
            rmsd_matrix[i:, i] = rmsd_vals  # mirror to maintain symmetry
    else:
        # When structures2 is different, compute fully
        # Vectorized approach
        # Broadcast to create a differences array of shape (N1, N2, A, 3)
        diff = structures1[:, None, :, :] - structures2[None, :, :, :]
        # sum over atoms and coordinates
        sq_diff_sum = np.sum(diff**2, axis=(2, 3)) / A
        rmsd_matrix = np.sqrt(sq_diff_sum)

    return rmsd_matrix
