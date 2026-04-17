import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import streamlit as st


st.set_page_config(page_title="Distance Filter and Clustering", layout="wide")


def _save_upload_to_temp(uploaded_file, suffix: str) -> str:
    """Persist a Streamlit UploadedFile to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def load_trajectory(pdb_file, xtc_file=None) -> md.Trajectory:
    """Load a topology-only PDB or a PDB+XTC trajectory from uploaded files."""
    pdb_path = _save_upload_to_temp(pdb_file, ".pdb")
    try:
        if xtc_file is not None:
            xtc_path = _save_upload_to_temp(xtc_file, ".xtc")
            try:
                traj = md.load(xtc_path, top=pdb_path)
            finally:
                if os.path.exists(xtc_path):
                    os.remove(xtc_path)
        else:
            traj = md.load(pdb_path)
    finally:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
    return traj


def residue_labels(traj: md.Trajectory) -> List[str]:
    return [f"{res.name} {res.index + 1}" for res in traj.topology.residues]


def find_ca_index(traj: md.Trajectory, residue_display: str) -> int:
    try:
        _, residue_number = residue_display.split()
        idx = int(residue_number) - 1
    except Exception as exc:
        raise ValueError(f"Invalid residue selection: {residue_display!r}") from exc

    residues = list(traj.topology.residues)
    if idx < 0 or idx >= len(residues):
        raise ValueError(f"Residue index out of range: {residue_display!r}")

    residue = residues[idx]
    ca = next((atom.index for atom in residue.atoms if atom.name == "CA"), None)
    if ca is None:
        raise ValueError(f"Selected residue lacks a Cα atom: {residue_display}")
    return ca


def parse_pairs(traj: md.Trajectory, pairs_state: List[Dict[str, str]]) -> Tuple[List[Tuple[int, int]], List[float]]:
    atom_pairs: List[Tuple[int, int]] = []
    thresholds_nm: List[float] = []

    for row in pairs_state:
        res1 = row.get("res1", "")
        res2 = row.get("res2", "")
        dist_text = row.get("distance", "")

        if not res1 or not res2 or not dist_text:
            raise ValueError("Each pair row must include two residues and a distance threshold.")

        ca1 = find_ca_index(traj, res1)
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
