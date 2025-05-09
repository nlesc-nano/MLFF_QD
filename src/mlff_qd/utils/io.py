import numpy as np
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

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
    - This function always writes to 'processed_data/filename'.
    - If 'energies' is provided, each frame's comment line includes that frame's energy.
    - You can use 'comment' to clarify if the frames are "Aligned positions", "Aligned forces", etc.
    """

    # Determine processed output directory
    processed_dir = Path.cwd() / "processed_data"

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

def parse_positions_xyz(filename, num_atoms):
    """
    Parse positions from an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.
        num_atoms (int): Number of atoms in each frame.

    Returns:
        np.ndarray: Atomic positions (num_frames, num_atoms, 3).
        list: Atomic types.
        list: Total energies for each frame (if available; otherwise, empty list).
    """
    logger.info(f"Parsing positions XYZ file: {filename}")
    positions = []
    atom_types = []
    total_energies = []

    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2  # 2 lines for header and comment

        for i in range(0, len(lines), num_lines_per_frame):
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            comment_line = lines[i + 1]

            # Try parsing the energy; otherwise, skip
            try:
                total_energy = float(comment_line.split("=")[-1].strip())
                total_energies.append(total_energy)
            except ValueError:
                total_energies.append(None)  # Placeholder for missing energy

            frame_positions = []
            for line in atom_lines:
                parts = line.split()
                atom_types.append(parts[0])
                frame_positions.append([float(x) for x in parts[1:4]])

            positions.append(frame_positions)

    return np.array(positions), atom_types[:num_atoms], total_energies

def parse_forces_xyz(filename, num_atoms):
    """Parse forces from an XYZ file."""
    logger.info(f"Parsing forces XYZ file: {filename}")
    forces = []
    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            frame_forces = []
            for j in range(2, 2 + num_atoms):
                parts = lines[i + j].split()
                frame_forces.append(list(map(float, parts[1:4])))
            forces.append(frame_forces)
    
    return np.array(forces)

def get_num_atoms(filename):
    """
    Retrieve the number of atoms from the first line of an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.

    Returns:
        int: Number of atoms in the structure.
    """
    with open(filename, "r") as f:
        num_atoms = int(f.readline().strip())
    
    logger.info(f"Number of atoms: {num_atoms}")
    
    return num_atoms
