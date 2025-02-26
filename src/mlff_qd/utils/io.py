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

def save_xyz(filename, positions, atom_types, energies=None, comment="Frame"):
    """
    Save atomic positions to an XYZ file in 'data/processed'.

    Parameters
    ----------
    filename : str
        The output filename (e.g. 'aligned_positions.xyz').
    frames : (num_frames, num_atoms, 3) array-like
        Atomic positions or forces for each frame.
    atom_types : list of str
        The atomic symbols corresponding to each atom (e.g., ["Cs", "Br", ...]).
    energies : list[float] or None, optional
        If provided, each frame's energy is appended to the comment line.
        Must match the number of frames if given.
    comment : str, optional
        A custom label for the comment line. Defaults to "Frame".

    Notes
    -----
    - This function always writes to 'data/processed/filename'.
    - If 'energies' is provided, each frame's comment line includes that frame's energy.
    - You can use 'comment' to clarify if the frames are "Aligned positions", "Aligned forces", etc.
    """

    # Determine processed output directory
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Build the full path
    output_path = processed_dir / filename

    # Convert frames to a NumPy array if needed
    frames = np.asarray(positions)
    num_frames = len(frames)
    num_atoms = len(atom_types)
    has_energies = (energies is not None) and (len(energies) == num_frames)

    print(f"Saving XYZ data to: {output_path}")
    with open(output_path, "w") as f:
        for i, frame in enumerate(frames):
            f.write(f"{num_atoms}\n")

            # Construct the comment line
            comment_line = f"{comment} {i+1}"
            if has_energies and energies[i] is not None:
                comment_line += f", Energy = {energies[i]:.6f} eV"

            f.write(comment_line + "\n")

            # Write each atom line
            for atom, (x, y, z) in zip(atom_types, frame):
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Done. Wrote {num_frames} frames to '{output_path}'.")

def save_frequencies(filename, frequencies):
    """
    Save vibrational frequencies to a file.

    Parameters:
        filename (str): Path to the output file.
        frequencies (np.ndarray): Vibrational frequencies (cm^-1).
    """
    print(f"Saving vibrational frequencies to {filename}...")
    with open(filename, "w") as f:
        f.write("Vibrational Frequencies (cm^-1):\n")
        for i, freq in enumerate(frequencies, start=1):
            f.write(f"Mode {i}: {freq:.6f} cm^-1\n")
    
    print(f"Frequencies saved to {filename}.")

def save_binary(filename, frequencies, positions, eigenvectors, atom_types):
    """Save frequencies, positions, eigenvectors, and atom types to a binary file."""
    data = {
        "frequencies": frequencies,
        "positions": positions,
        "eigenvectors": eigenvectors,
        "atom_types": atom_types
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved data to binary file: {filename}")

def load_binary(filename):
    """Load frequencies, positions, eigenvectors, and atom types from a binary file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded data from binary file: {filename}")
    
    return data["frequencies"], data["eigenvectors"]

def reorder_xyz_trajectory(input_file, output_file, num_atoms):
    """Reorder atoms in the XYZ trajectory."""
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / output_file

    print(f"Reordering atoms in trajectory file: {input_file}")

    with open(input_file, "r") as infile, open(output_path, "w") as outfile:
        lines = infile.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            header = lines[i:i + 2]
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            sorted_atoms = sorted(atom_lines, key=lambda x: x.split()[0])
            outfile.writelines(header)
            outfile.writelines(sorted_atoms)
    
    print(f"Reordered trajectory saved to: {output_path}")
