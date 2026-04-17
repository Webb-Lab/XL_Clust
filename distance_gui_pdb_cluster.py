"""
distance_gui_pdb_cluster.py
============================

Graphical user interface (GUI) for filtering a molecular dynamics trajectory based
on Cα–Cα distances between selected residue pairs and clustering the resulting
frames to identify a representative structure.  The program uses `mdtraj` to
load coordinate/topology data, compute distances and RMSDs, and write out
filtered trajectories.  The user can load a PDB topology file and an optional
XTC trajectory file, specify any number of residue pairs with distance
thresholds (in Å), and then save:

* A multi‑model PDB containing only the frames where every pair satisfies the
  distance criteria, and
* A single‑frame PDB containing a representative structure selected by
  clustering the filtered frames based on backbone RMSD.

The centroid is chosen using the algorithm described in the MDTraj examples
for finding cluster centroids【817754265776463†L71-L105】.  Briefly, all pairwise
RMSDs between the filtered frames are computed (using backbone atoms), these
distances are converted to similarity scores, and the frame with the maximum
sum of similarities is selected as the representative.  This approach avoids
explicitly choosing the number of clusters, while still selecting a structure
that lies near the centre of the conformational ensemble.

Usage:

    python distance_gui_pdb_cluster.py

Requires the `mdtraj` and `numpy` libraries.  If you intend to run the GUI
from a different environment, ensure that Tkinter is available.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import mdtraj as md


class DistanceGUI:
    """Main GUI application for distance filtering and clustering."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Distance Filter and Clustering")

        # Trajectory variables
        self.traj = None  # type: md.Trajectory | None
        self.pdb_path = tk.StringVar()
        self.xtc_path = tk.StringVar()
        self.output_pdb_path = tk.StringVar()
        self.representative_pdb_path = tk.StringVar()

        # Widgets for file selection
        file_frame = ttk.LabelFrame(self.root, text="Input Files")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # PDB selection
        ttk.Label(file_frame, text="PDB file:").grid(row=0, column=0, sticky=tk.W)
        self.pdb_entry = ttk.Entry(file_frame, textvariable=self.pdb_path, width=50)
        self.pdb_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(file_frame, text="Browse", command=self.browse_pdb).grid(row=0, column=2, padx=5)

        # XTC selection (optional)
        ttk.Label(file_frame, text="XTC file (optional):").grid(row=1, column=0, sticky=tk.W)
        self.xtc_entry = ttk.Entry(file_frame, textvariable=self.xtc_path, width=50)
        self.xtc_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Button(file_frame, text="Browse", command=self.browse_xtc).grid(row=1, column=2, padx=5)

        # Output file selection
        output_frame = ttk.LabelFrame(self.root, text="Output Files")
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(output_frame, text="Filtered PDB file:").grid(row=0, column=0, sticky=tk.W)
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_pdb_path, width=50)
        self.output_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_pdb).grid(row=0, column=2, padx=5)

        ttk.Label(output_frame, text="Representative PDB file:").grid(row=1, column=0, sticky=tk.W)
        self.rep_entry = ttk.Entry(output_frame, textvariable=self.representative_pdb_path, width=50)
        self.rep_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Button(output_frame, text="Browse", command=self.browse_rep_pdb).grid(row=1, column=2, padx=5)

        # Pair specification area
        pair_frame = ttk.LabelFrame(self.root, text="Residue Pairs and Thresholds")
        pair_frame.pack(fill=tk.X, padx=10, pady=5)

        header = ["Residue 1", "Residue 2", "Max Distance (Å)", ""]
        for j, text in enumerate(header):
            ttk.Label(pair_frame, text=text).grid(row=0, column=j, padx=5, pady=2)

        # Container for dynamically added rows
        self.pairs_container = ttk.Frame(pair_frame)
        self.pairs_container.grid(row=1, column=0, columnspan=4, sticky=tk.W)

        # List to keep track of row widgets
        self.pair_rows: list[dict[str, any]] = []

        ttk.Button(pair_frame, text="Add Pair", command=self.add_pair_row).grid(row=2, column=0, padx=5, pady=5)

        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Load a PDB file to begin.")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Start by adding a single pair row (user can add more)
        self.add_pair_row()

    # File browsing functions
    def browse_pdb(self) -> None:
        filename = filedialog.askopenfilename(title="Select PDB file", filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")])
        if filename:
            self.pdb_path.set(filename)
            self.load_trajectory()

    def browse_xtc(self) -> None:
        filename = filedialog.askopenfilename(title="Select XTC file", filetypes=[("Trajectory files", "*.xtc"), ("All files", "*.*")])
        if filename:
            self.xtc_path.set(filename)
            self.load_trajectory()

    def browse_output_pdb(self) -> None:
        filename = filedialog.asksaveasfilename(title="Select output PDB file", defaultextension=".pdb", filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")])
        if filename:
            self.output_pdb_path.set(filename)

    def browse_rep_pdb(self) -> None:
        filename = filedialog.asksaveasfilename(title="Select representative PDB file", defaultextension=".pdb", filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")])
        if filename:
            self.representative_pdb_path.set(filename)

    # Loading trajectory
    def load_trajectory(self) -> None:
        """Load the topology and optional trajectory. Populate the comboboxes."""
        pdb = self.pdb_path.get()
        if not pdb:
            return
        try:
            # If xtc is specified, load xtc with pdb as topology
            if self.xtc_path.get():
                self.traj = md.load(self.xtc_path.get(), top=pdb)
            else:
                # md.load can load PDB directly
                self.traj = md.load(pdb)
        except Exception as exc:
            messagebox.showerror("Error loading trajectory", str(exc))
            self.traj = None
            return
        # Populate residue options
        self.update_residue_options()
        self.status_var.set(f"Loaded {self.traj.n_frames} frame(s). Select residue pairs and thresholds.")

    def update_residue_options(self) -> None:
        """Update combobox values with residues in the loaded topology."""
        if not self.traj:
            return
        # Build list of residue display strings: e.g. "LYS 7"
        residues = []
        for res in self.traj.topology.residues:
            # 1‑based residue numbering
            display = f"{res.name} {res.index + 1}"
            residues.append(display)
        # Update each pair row combobox values
        for row in self.pair_rows:
            row['res1'].config(values=residues)
            row['res2'].config(values=residues)
        # Optionally set default values if they are empty
        if residues and len(self.pair_rows) > 0:
            for row in self.pair_rows:
                if not row['res1'].get():
                    row['res1'].set(residues[0])
                if not row['res2'].get():
                    row['res2'].set(residues[0])

    def add_pair_row(self) -> None:
        """Add a new row for specifying a residue pair and distance threshold."""
        row_frame = ttk.Frame(self.pairs_container)
        row_frame.pack(fill=tk.X, pady=2)
        res1_cb = ttk.Combobox(row_frame, width=20)
        res2_cb = ttk.Combobox(row_frame, width=20)
        dist_entry = ttk.Entry(row_frame, width=15)
        remove_btn = ttk.Button(row_frame, text="Remove", command=lambda rf=row_frame: self.remove_pair_row(rf))
        res1_cb.grid(row=0, column=0, padx=5)
        res2_cb.grid(row=0, column=1, padx=5)
        dist_entry.grid(row=0, column=2, padx=5)
        remove_btn.grid(row=0, column=3, padx=5)
        # Store row information
        self.pair_rows.append({'frame': row_frame, 'res1': res1_cb, 'res2': res2_cb, 'dist': dist_entry})
        # If trajectory is loaded, populate values
        self.update_residue_options()

    def remove_pair_row(self, row_frame: ttk.Frame) -> None:
        """Remove a specified row from the pairs container."""
        # Find the row dict and remove it
        for i, row in enumerate(self.pair_rows):
            if row['frame'] == row_frame:
                self.pair_rows.pop(i)
                break
        row_frame.destroy()

    def validate_inputs(self) -> bool:
        """Validate that all required inputs are provided and logically valid."""
        if self.traj is None:
            messagebox.showwarning("Missing data", "Please load a PDB (and optional XTC) first.")
            return False
        if not self.output_pdb_path.get():
            messagebox.showwarning("Missing output", "Please specify an output PDB file for filtered frames.")
            return False
        if not self.representative_pdb_path.get():
            messagebox.showwarning("Missing representative output", "Please specify an output PDB file for the representative structure.")
            return False
        # At least one pair must be specified
        if len(self.pair_rows) == 0:
            messagebox.showwarning("No residue pairs", "Please add at least one residue pair and distance threshold.")
            return False
        return True

    def parse_pair_entries(self) -> tuple[list[tuple[int, int]], list[float]]:
        """Parse user input to produce atom pairs and thresholds in nanometers.

        Returns
        -------
        atom_pairs : list of (int, int)
            Indices of Cα atoms for each residue pair.
        thresholds_nm : list of float
            Distance thresholds in nanometers corresponding to each pair.
        """
        assert self.traj is not None
        atom_pairs: list[tuple[int, int]] = []
        thresholds_nm: list[float] = []
        for row in self.pair_rows:
            res1_text = row['res1'].get()
            res2_text = row['res2'].get()
            dist_text = row['dist'].get()
            if not res1_text or not res2_text or not dist_text:
                raise ValueError("All pair rows must have residue selections and distance thresholds.")
            # Parse residue name and number (format "RESNAME N")
            try:
                name1, num1_str = res1_text.split()
                name2, num2_str = res2_text.split()
                idx1 = int(num1_str) - 1  # convert from 1‑based to 0‑based residue index
                idx2 = int(num2_str) - 1
            except Exception:
                raise ValueError(f"Invalid residue format: '{res1_text}' or '{res2_text}'")
            # Find Cα atom indices for the residues
            res1 = list(self.traj.topology.residues)[idx1]
            res2 = list(self.traj.topology.residues)[idx2]
            ca1 = next((atom.index for atom in res1.atoms if atom.name == 'CA'), None)
            ca2 = next((atom.index for atom in res2.atoms if atom.name == 'CA'), None)
            if ca1 is None or ca2 is None:
                raise ValueError(f"Selected residue(s) lack a Cα atom: {res1} {res2}")
            atom_pairs.append((ca1, ca2))
            try:
                threshold_ang = float(dist_text)
            except ValueError:
                raise ValueError(f"Invalid distance threshold: '{dist_text}'. Provide a numeric value in Å.")
            # Convert Ångströms to nanometers (1 Å = 0.1 nm)
            thresholds_nm.append(threshold_ang * 0.1)
        return atom_pairs, thresholds_nm

    def filter_and_cluster(self) -> None:
        """Filter frames based on Cα distances and cluster filtered frames to pick a representative."""
        if not self.validate_inputs():
            return
        assert self.traj is not None
        try:
            atom_pairs, thresholds = self.parse_pair_entries()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return
        # Compute distances for all frames and all atom pairs
        try:
            distances = md.compute_distances(self.traj, atom_pairs)  # shape (n_frames, n_pairs)
        except Exception as exc:
            messagebox.showerror("Distance Calculation Error", str(exc))
            return
        # Build mask for frames satisfying all thresholds
        mask = np.all(distances <= thresholds, axis=1)
        valid_indices = np.where(mask)[0]
        # Slice filtered trajectory
        filtered_traj = self.traj.slice(valid_indices, inplace=False) if len(valid_indices) > 0 else None
        # Save filtered frames to PDB
        out_path = self.output_pdb_path.get()
        rep_path = self.representative_pdb_path.get()
        if filtered_traj is not None and filtered_traj.n_frames > 0:
            try:
                filtered_traj.save_pdb(out_path)
                msg = f"Filtered PDB saved with {filtered_traj.n_frames} frame(s)."
            except Exception as exc:
                messagebox.showerror("Save Error", f"Failed to save filtered PDB: {exc}")
                return
            # Compute representative structure
            try:
                rep_traj = self.compute_representative(filtered_traj)
                rep_traj.save_pdb(rep_path)
                msg += " Representative structure saved."
            except Exception as exc:
                messagebox.showerror("Clustering Error", str(exc))
                return
        else:
            # Create empty file or notify user if no frames meet criteria
            open(out_path, 'w').close()
            msg = "No frames meet distance criteria; empty PDB created."
        # Update status
        self.status_var.set(msg)

    def compute_representative(self, traj: md.Trajectory) -> md.Trajectory:
        """Compute a representative structure using backbone RMSD clustering.

        This method computes all pairwise RMSDs between frames in the trajectory
        using backbone atoms and then uses the centroid heuristic from the MDTraj
        examples【817754265776463†L71-L105】 to select the frame with the highest
        overall similarity.  If only a single frame is present, it simply returns
        that frame.

        Parameters
        ----------
        traj : md.Trajectory
            Filtered trajectory containing only the frames that satisfy the
            distance criteria.

        Returns
        -------
        md.Trajectory
            A single-frame trajectory containing the representative structure.
        """
        if traj.n_frames == 0:
            raise ValueError("Cannot compute representative of empty trajectory.")
        if traj.n_frames == 1:
            return traj[0]
        # Select backbone atoms (N, CA, C)
        backbone_indices = traj.topology.select('backbone')
        n = traj.n_frames
        # Compute pairwise RMSDs
        distances = np.empty((n, n))
        for i in range(n):
            # md.rmsd returns RMSDs of all frames relative to frame i
            distances[i] = md.rmsd(traj, traj, i, atom_indices=backbone_indices)
        # Convert distances to similarity scores; avoid divide-by-zero
        std = distances.std()
        if std == 0:
            # All frames are identical; choose the first
            index = 0
        else:
            # Beta = 1 as in the MDTraj example
            similarity = np.exp(-1.0 * distances / std)
            # Sum similarities for each frame; pick the one with maximum sum
            index = similarity.sum(axis=1).argmax()
        return traj[index]


def main() -> None:
    root = tk.Tk()
    app = DistanceGUI(root)
    # Add a button to start filtering and clustering
    action_frame = ttk.Frame(root)
    action_frame.pack(fill=tk.X, padx=10, pady=10)
    ttk.Button(action_frame, text="Filter and Cluster", command=app.filter_and_cluster).pack()
    root.mainloop()


if __name__ == '__main__':
    main()
