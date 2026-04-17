"""
Streamlit application for filtering molecular dynamics trajectories based on
Cα–Cα distances and selecting a representative structure.

This app reproduces the functionality of the earlier Tkinter GUI, but
is adapted for a web interface using Streamlit.  Users can upload a PDB file
(topology) and an optional XTC file, specify any number of residue pairs along
with maximum Cα–Cα distances (in Å), and then filter the trajectory to retain
only those frames where all pairs satisfy their thresholds.  The filtered
frames are returned as a multi‑model PDB file, and a single representative
structure is selected using a backbone RMSD clustering approach analogous to
the centroid algorithm in the MDTraj documentation【817754265776463†L71-L105】.

Key features:

* Upload PDB and optional XTC directly from the browser.  Uploaded files are
  written to temporary files on the server and loaded via `mdtraj.load`.
* Define multiple residue pairs and distance thresholds.  Pairs are stored in
  `st.session_state` so that they persist across interactions.
* Filter frames using `mdtraj.compute_distances`, converting Å to nm as
  required by MDTraj【18013314271708†L50-L77】.  If no frames meet the criteria,
  the user is notified.
* Compute a representative structure by calculating all pairwise backbone
  RMSDs among the filtered frames and selecting the frame with the greatest
  summed similarity (exp(–RMSD/σ))【817754265776463†L71-L105】.
* Provide download buttons for the filtered trajectory and the representative
  structure as PDB files.

To run this app locally:

    streamlit run app.py

Dependencies are listed in `requirements.txt`.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import List, Tuple

import numpy as np
import streamlit as st
import mdtraj as md


def load_trajectory(pdb_data: bytes, xtc_data: bytes | None) -> md.Trajectory:
    """Load a trajectory from uploaded file contents.

    Parameters
    ----------
    pdb_data : bytes
        Contents of the PDB file (topology).
    xtc_data : bytes or None
        Contents of the XTC file.  If None, only the PDB is loaded.

    Returns
    -------
    md.Trajectory
        Loaded trajectory.
    """
    # Write the PDB to a temporary file
    pdb_fd, pdb_path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(pdb_fd, "wb") as fh:
        fh.write(pdb_data)
    traj: md.Trajectory
    try:
        if xtc_data:
            xtc_fd, xtc_path = tempfile.mkstemp(suffix=".xtc")
            with os.fdopen(xtc_fd, "wb") as fh_xtc:
                fh_xtc.write(xtc_data)
            try:
                traj = md.load(xtc_path, top=pdb_path)
            finally:
                os.remove(xtc_path)
        else:
            traj = md.load(pdb_path)
    finally:
        # remove the temporary pdb after loading
        os.remove(pdb_path)
    return traj


def parse_residue_id(res_string: str) -> Tuple[str, int]:
    """Parse a residue selection string into name and index.

    Examples
    --------
    >>> parse_residue_id('LYS 7')
    ('LYS', 6)  # zero‑based index
    """
    name, number_str = res_string.split()
    return name, int(number_str) - 1


def compute_atom_pairs(traj: md.Trajectory, pairs: List[dict]) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Given a list of user‑selected residue pairs and thresholds, return atom
    pairs (Cα indices) and thresholds in nanometers.
    """
    atom_pairs: List[Tuple[int, int]] = []
    thresholds_nm: List[float] = []
    residues = list(traj.topology.residues)
    for pair in pairs:
        res1_name, res1_idx = parse_residue_id(pair['res1'])
        res2_name, res2_idx = parse_residue_id(pair['res2'])
        # Validate indices
        if res1_idx < 0 or res1_idx >= len(residues) or res2_idx < 0 or res2_idx >= len(residues):
            raise ValueError(f"Invalid residue index in pair {pair}.")
        # Retrieve residues
        res1 = residues[res1_idx]
        res2 = residues[res2_idx]
        # Get Cα atoms
        ca1 = next((atom.index for atom in res1.atoms if atom.name == 'CA'), None)
        ca2 = next((atom.index for atom in res2.atoms if atom.name == 'CA'), None)
        if ca1 is None or ca2 is None:
            raise ValueError(f"Selected residue(s) lack a Cα atom: {pair}")
        atom_pairs.append((ca1, ca2))
        thresholds_nm.append(pair['dist'] * 0.1)  # Å to nm
    return atom_pairs, thresholds_nm


def filter_frames(traj: md.Trajectory, atom_pairs: List[Tuple[int, int]], thresholds: List[float]) -> Tuple[md.Trajectory | None, List[int]]:
    """Filter a trajectory to frames where all specified distances satisfy their thresholds.

    Parameters
    ----------
    traj : md.Trajectory
        Input trajectory.
    atom_pairs : list of (int, int)
        Atom index pairs for distance computation.
    thresholds : list of float
        Distance thresholds (nanometers) for each pair.

    Returns
    -------
    filtered_traj : md.Trajectory or None
        Trajectory containing only the filtered frames, or None if no frames
        satisfy the criteria.
    indices : list of int
        Indices of frames that satisfy the criteria.
    """
    if not atom_pairs:
        return None, []
    distances = md.compute_distances(traj, atom_pairs)  # shape (n_frames, n_pairs)【18013314271708†L50-L77】
    mask = np.all(distances <= thresholds, axis=1)
    valid_indices = np.where(mask)[0].tolist()
    if not valid_indices:
        return None, []
    filtered = traj.slice(valid_indices)
    return filtered, valid_indices


def select_representative(traj: md.Trajectory) -> md.Trajectory:
    """Select a representative frame from a trajectory using backbone RMSD clustering.

    The algorithm computes the pairwise RMSD matrix on backbone atoms and
    converts distances into similarity scores; the frame with the highest
    summed similarity is chosen as the representative【817754265776463†L71-L105】.

    Parameters
    ----------
    traj : md.Trajectory
        Input trajectory with at least one frame.

    Returns
    -------
    md.Trajectory
        A single-frame trajectory containing the representative structure.
    """
    if traj.n_frames == 0:
        raise ValueError("Cannot compute representative of empty trajectory.")
    if traj.n_frames == 1:
        return traj[0]
    # backbone atom indices: N, CA, C
    backbone_indices = traj.topology.select('backbone')
    n = traj.n_frames
    distances = np.empty((n, n))
    for i in range(n):
        distances[i] = md.rmsd(traj, traj, i, atom_indices=backbone_indices)
    std = distances.std()
    if std == 0:
        idx = 0
    else:
        similarity = np.exp(-1.0 * distances / std)
        idx = similarity.sum(axis=1).argmax()
    return traj[idx]


def trajectory_to_pdb_bytes(traj: md.Trajectory) -> bytes:
    """Write a trajectory to a temporary PDB file and return its bytes."""
    fd, path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        traj.save_pdb(path)
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        os.remove(path)
    return data


def main() -> None:
    st.set_page_config(page_title="Trajectory Distance Filter and Clustering", layout="wide")
    st.title("Trajectory Distance Filter and Clustering")
    st.write(
        "Upload a PDB file (and optional XTC), choose residue pairs and distance thresholds, "
        "then filter frames and retrieve a representative structure based on backbone RMSD."
    )

    # Initialize session state for pairs
    if 'pairs' not in st.session_state:
        st.session_state.pairs = []  # list of dicts {'res1', 'res2', 'dist'}

    # Step 1: File upload
    uploaded_pdb = st.file_uploader("PDB file", type=["pdb"], key="pdb")
    uploaded_xtc = st.file_uploader("XTC file (optional)", type=["xtc"], key="xtc")

    traj = None
    residue_options: List[str] = []
    if uploaded_pdb is not None:
        pdb_bytes = uploaded_pdb.getvalue()
        xtc_bytes = uploaded_xtc.getvalue() if uploaded_xtc is not None else None
        try:
            traj = load_trajectory(pdb_bytes, xtc_bytes)
            residue_options = [f"{res.name} {res.index + 1}" for res in traj.topology.residues]
            st.success(f"Loaded trajectory with {traj.n_frames} frame(s).")
        except Exception as exc:
            st.error(f"Error loading files: {exc}")

    # Step 2: Add residue pairs and thresholds
    if traj is not None:
        with st.expander("Add residue pair", expanded=True):
            col1, col2, col3 = st.columns([3, 3, 2])
            with col1:
                res1_select = st.selectbox("Residue 1", residue_options, key="res1_select")
            with col2:
                res2_select = st.selectbox("Residue 2", residue_options, key="res2_select")
            with col3:
                dist_input = st.number_input(
                    "Max distance (Å)", min_value=0.0, value=5.0, key="dist_input"
                )
            if st.button("Add pair"):
                # Append to session state
                st.session_state.pairs.append({
                    'res1': res1_select,
                    'res2': res2_select,
                    'dist': float(dist_input),
                })
                st.success(f"Added pair: {res1_select} – {res2_select} with max {dist_input:.2f} Å")

        # Display existing pairs with removal buttons
        if st.session_state.pairs:
            st.subheader("Defined pairs")
            for i, pair in enumerate(st.session_state.pairs):
                cols = st.columns([3, 3, 2, 1])
                cols[0].write(pair['res1'])
                cols[1].write(pair['res2'])
                cols[2].write(f"{pair['dist']:.2f} Å")
                # Provide a remove button with a unique key
                if cols[3].button("Remove", key=f"remove_{i}"):
                    st.session_state.pairs.pop(i)
                    st.experimental_rerun()

        # Step 3: Filtering and clustering
        if st.session_state.pairs:
            if st.button("Filter and cluster"):
                try:
                    atom_pairs, thresholds_nm = compute_atom_pairs(traj, st.session_state.pairs)
                    filtered_traj, valid_indices = filter_frames(traj, atom_pairs, thresholds_nm)
                    if filtered_traj is None or filtered_traj.n_frames == 0:
                        st.warning("No frames meet the distance criteria.")
                    else:
                        # Prepare download for filtered trajectory
                        filtered_data = trajectory_to_pdb_bytes(filtered_traj)
                        st.success(f"Filtered trajectory contains {filtered_traj.n_frames} frame(s).")
                        st.download_button(
                            label="Download filtered PDB",
                            data=filtered_data,
                            file_name="filtered_trajectory.pdb",
                            mime="chemical/x-pdb",
                        )
                        # Compute representative
                        representative = select_representative(filtered_traj)
                        rep_data = trajectory_to_pdb_bytes(representative)
                        st.download_button(
                            label="Download representative PDB",
                            data=rep_data,
                            file_name="representative_structure.pdb",
                            mime="chemical/x-pdb",
                        )
                except Exception as exc:
                    st.error(f"Error during filtering/clustering: {exc}")


if __name__ == "__main__":
    main()        ca1 = find_ca_index(traj, res1)
        ca2 = find_ca_index(traj, res2)

        try:
            threshold_ang = float(dist_text)
        except ValueError as exc:
            raise ValueError(f"Invalid distance threshold: {dist_text!r}. Use a numeric value in Å.") from exc

        atom_pairs.append((ca1, ca2))
        thresholds_nm.append(threshold_ang * 0.1)

    return atom_pairs, thresholds_nm


def compute_representative(traj: md.Trajectory) -> md.Trajectory:
    if traj.n_frames == 0:
        raise ValueError("Cannot compute a representative structure from an empty trajectory.")
    if traj.n_frames == 1:
        return traj[0]

    backbone_indices = traj.topology.select("backbone")
    n = traj.n_frames
    distances = np.empty((n, n), dtype=float)

    for i in range(n):
        distances[i] = md.rmsd(traj, traj, i, atom_indices=backbone_indices)

    std = distances.std()
    if std == 0:
        return traj[0]

    similarity = np.exp(-1.0 * distances / std)
    index = int(similarity.sum(axis=1).argmax())
    return traj[index]


def ensure_state() -> None:
    if "pair_rows" not in st.session_state:
        st.session_state.pair_rows = [str(uuid.uuid4())[:8]]
    if "results" not in st.session_state:
        st.session_state.results = None


def add_pair_row() -> None:
    st.session_state.pair_rows.append(str(uuid.uuid4())[:8])


def remove_pair_row(row_id: str) -> None:
    st.session_state.pair_rows = [rid for rid in st.session_state.pair_rows if rid != row_id]
    for suffix in ("res1_", "res2_", "distance_"):
        st.session_state.pop(f"{suffix}{row_id}", None)


def main() -> None:
    ensure_state()

    st.title("Distance Filter and Clustering")
    st.write(
        "Upload a PDB topology file and, optionally, an XTC trajectory. "
        "Then define residue pairs and maximum Cα–Cα distances in Å. "
        "The app will save a filtered multi-model PDB and a representative structure."
    )

    with st.sidebar:
        st.header("Input files")
        pdb_file = st.file_uploader("PDB file", type=["pdb"])
        xtc_file = st.file_uploader("XTC file (optional)", type=["xtc"])
        st.caption("The XTC file is optional. If omitted, the PDB is loaded as a single-frame trajectory.")

    traj: Optional[md.Trajectory] = None
    residues: List[str] = []

    if pdb_file is not None:
        try:
            traj = load_trajectory(pdb_file, xtc_file)
            residues = residue_labels(traj)
            st.success(f"Loaded trajectory with {traj.n_frames} frame(s) and {len(residues)} residue(s).")
        except Exception as exc:
            st.error(f"Failed to load trajectory: {exc}")
            st.stop()
    else:
        st.info("Upload a PDB file to begin.")

    st.subheader("Residue pairs and thresholds")
    left, right = st.columns([1, 1])
    with left:
        if st.button("Add pair"):
            add_pair_row()
            st.rerun()
    with right:
        st.caption("Thresholds are entered in Å and converted internally to nm.")

    if traj is not None and len(st.session_state.pair_rows) == 0:
        st.session_state.pair_rows = [str(uuid.uuid4())[:8]]

    pairs_state: List[Dict[str, str]] = []
    for row_id in list(st.session_state.pair_rows):
        cols = st.columns([3, 3, 2, 1])
        with cols[0]:
            res1 = st.selectbox(
                "Residue 1",
                options=residues if residues else [""],
                key=f"res1_{row_id}",
                label_visibility="collapsed",
                disabled=not residues,
            )
        with cols[1]:
            res2 = st.selectbox(
                "Residue 2",
                options=residues if residues else [""],
                key=f"res2_{row_id}",
                label_visibility="collapsed",
                disabled=not residues,
            )
        with cols[2]:
            dist_default = st.session_state.get(f"distance_{row_id}", "")
            distance = st.text_input(
                "Max distance (Å)",
                value=dist_default,
                key=f"distance_{row_id}",
                label_visibility="collapsed",
                placeholder="e.g. 8.0",
            )
        with cols[3]:
            st.write("")
            st.write("")
            if st.button("Remove", key=f"remove_{row_id}"):
                remove_pair_row(row_id)
                st.rerun()

        pairs_state.append({"res1": res1, "res2": res2, "distance": distance})

    st.divider()

    out_col1, out_col2 = st.columns(2)
    with out_col1:
        filtered_name = st.text_input("Filtered PDB filename", value="filtered_frames.pdb")
    with out_col2:
        representative_name = st.text_input("Representative PDB filename", value="representative_structure.pdb")

    run = st.button("Filter and Cluster", type="primary", disabled=traj is None)

    if run:
        if traj is None:
            st.error("Upload a PDB file first.")
            st.stop()
        if not pairs_state:
            st.error("Add at least one residue pair.")
            st.stop()

        try:
            atom_pairs, thresholds = parse_pairs(traj, pairs_state)
            distances = md.compute_distances(traj, atom_pairs)
            mask = np.all(distances <= thresholds, axis=1)
            valid_indices = np.where(mask)[0]

            filtered_bytes = b""
            representative_bytes = b""
            summary = {
                "filtered_frames": int(len(valid_indices)),
                "total_frames": int(traj.n_frames),
                "criteria_met": bool(len(valid_indices) > 0),
            }

            if len(valid_indices) > 0:
                filtered_traj = traj.slice(valid_indices, inplace=False)
                filtered_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
                filtered_tmp.close()
                try:
                    filtered_traj.save_pdb(filtered_tmp.name)
                    filtered_bytes = Path(filtered_tmp.name).read_bytes()
                finally:
                    if os.path.exists(filtered_tmp.name):
                        os.remove(filtered_tmp.name)

                rep_traj = compute_representative(filtered_traj)
                rep_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
                rep_tmp.close()
                try:
                    rep_traj.save_pdb(rep_tmp.name)
                    representative_bytes = Path(rep_tmp.name).read_bytes()
                finally:
                    if os.path.exists(rep_tmp.name):
                        os.remove(rep_tmp.name)

                st.success(
                    f"Found {len(valid_indices)} filtered frame(s) out of {traj.n_frames}. Representative structure generated."
                )
            else:
                st.warning("No frames matched the criteria. The filtered output will be an empty PDB.")

            st.session_state.results = {
                "filtered_bytes": filtered_bytes,
                "representative_bytes": representative_bytes,
                "filtered_name": filtered_name,
                "representative_name": representative_name,
                "summary": summary,
            }
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.session_state.results = None

    if st.session_state.results:
        results = st.session_state.results
        summary = results["summary"]
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total frames", summary["total_frames"])
        c2.metric("Filtered frames", summary["filtered_frames"])
        c3.metric("Match found", "Yes" if summary["criteria_met"] else "No")

        st.download_button(
            label="Download filtered PDB",
            data=results["filtered_bytes"],
            file_name=results["filtered_name"],
            mime="chemical/x-pdb",
            disabled=len(results["filtered_bytes"]) == 0,
        )
        st.download_button(
            label="Download representative PDB",
            data=results["representative_bytes"],
            file_name=results["representative_name"],
            mime="chemical/x-pdb",
            disabled=len(results["representative_bytes"]) == 0,
        )

        if summary["filtered_frames"] == 0:
            st.caption("The filtered PDB download is disabled because no frames matched the threshold(s).")


if __name__ == "__main__":
    main()
