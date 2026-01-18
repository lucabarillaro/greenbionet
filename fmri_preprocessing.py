# --- fMRIPrep -> per-subject labeled graphs (.pt), parallel ------------------
import os, re, glob, json, math
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------------ CONFIG ---------------------------------------------------
FMRIPREP_DIR     = "/ds********-fmriprep"  # <-- Path to fmriprep derivatives
OUT_DIR          = "./data/fmriprep_graphs_pt"
ATLAS            = "schaefer"       # "schaefer" | "harvardoxford" | "aal"
N_ROIS           = 200
STANDARDIZE_TS   = True
TOPK_PER_NODE    = 10               # top-k neighbors by |corr| per node
FEATURES         = "ts_stats"       # "ts_stats" | "ts_pca"
PCA_DIM          = 16               # used if FEATURES == "ts_pca"
RANDOM_SEED      = 42
N_JOBS           = max(os.cpu_count() - 1, 1)

# Labeling rule you specified:
NUM_CASES        = 51               # first 51 subjects = case (1), rest = control (0)

os.makedirs(OUT_DIR, exist_ok=True)

# Reduce oversubscription in workers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------ DISCOVER & GROUP BY SUBJECT ------------------------------
# Grab all BOLDs in MNI space produced by fMRIPrep
# check for actual file name
bold_paths = sorted(glob.glob(
    os.path.join(FMRIPREP_DIR, "sub-*", "func",
                 "sub-*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
))
assert len(bold_paths) > 0, "No BOLD files found. Check FMRIPREP_DIR."

# get subject id like "sub-0001" or "sub-01"
def subj_id(path):
    m = re.search(r"(sub-(\d+))", path)
    if m:
        return m.group(1), int(m.group(2))
    # fallback: take first sub-XXX token as id, and try to parse number
    m2 = re.search(r"(sub-[^_/]+)", path)
    sid = m2.group(1) if m2 else "sub-unknown"
    nmatch = re.search(r"(\d+)", sid)
    nint = int(nmatch.group(1)) if nmatch else 10**9
    return sid, nint

# group all runs/tasks per subject
from collections import defaultdict
by_subj = defaultdict(list)
for p in bold_paths:
    sid, n = subj_id(p)
    by_subj[(sid, n)].append(p)

# stable numeric order
subjects_sorted = sorted(by_subj.keys(), key=lambda x: x[1])
num_subjects = len(subjects_sorted)
print(f"Found {num_subjects} subjects.")

# build label map: first NUM_CASES = 1 (case), remaining = 0 (control)
labels = {}
for idx, (sid, n) in enumerate(subjects_sorted):
    y = 1 if idx < NUM_CASES else 0
    labels[sid] = y

# ------------------ UTILS ----------------------------------------------------
def make_masker():
    from nilearn import datasets, input_data
    if ATLAS == "schaefer":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROIS)
        labels_img = atlas.maps
    elif ATLAS == "harvardoxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        labels_img = atlas.maps
    elif ATLAS == "aal":
        atlas = datasets.fetch_atlas_aal()
        labels_img = atlas.maps
    else:
        raise ValueError("Unsupported ATLAS")
    return input_data.NiftiLabelsMasker(labels_img=labels_img,
                                        standardize=STANDARDIZE_TS,
                                        detrend=True)

def confounds_for(bold_path):
    """Optional basic confounds (extend as needed)."""
    import pandas as pd
    tsv = bold_path.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
    if not os.path.exists(tsv):
        return None
    try:
        df = pd.read_csv(tsv, sep="\t")
        cols = [c for c in df.columns if c.startswith(("trans_", "rot_", "framewise_displacement"))]
        if not cols:
            return None
        return df[cols].values
    except Exception:
        return None

def build_graph_from_timeseries(ts, topk=10):
    from torch_geometric.utils import coalesce
    R = ts.shape[1]
    corr = np.corrcoef(ts.T)        # (R,R)
    np.fill_diagonal(corr, 0.0)
    edges = []
    for i in range(R):
        idx = np.argsort(-np.abs(corr[i]))[:topk]
        for j in idx:
            edges.append([i, j])
            edges.append([j, i])     # undirected
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = coalesce(edge_index, None, R, R)
    return edge_index

def make_node_features(ts, mode="ts_stats", pca_dim=16, seed=RANDOM_SEED):
    if mode == "ts_stats":
        feats = np.stack([
            ts.mean(axis=0),
            ts.std(axis=0),
            np.median(ts, axis=0),
            np.ptp(ts, axis=0)
        ], axis=1)  # (R,4)
    elif mode == "ts_pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=seed)
        feats = pca.fit_transform(ts.T)  # (R, pca_dim)
    else:
        raise ValueError("Unsupported FEATURES mode")
    return torch.tensor(feats, dtype=torch.float32)

# ------------------ WORKER: one subject --------------------------------------
def process_subject(sid, bold_list, label):
    try:
        out_path = os.path.join(OUT_DIR, f"{sid}_graph.pt")
        if os.path.exists(out_path):
            return ("skip", sid, None)

        masker = make_masker()

        # concatenate time series across all runs/tasks for this subject
        ts_cat = None
        for bold in sorted(bold_list):
            conf = confounds_for(bold)
            ts = masker.fit_transform(bold, confounds=conf)  # (T,R)
            if ts is None or ts.size == 0:
                continue
            ts_cat = ts if ts_cat is None else np.vstack([ts_cat, ts])

        if ts_cat is None or ts_cat.size == 0:
            return ("empty", sid, None)

        edge_index = build_graph_from_timeseries(ts_cat, topk=TOPK_PER_NODE)
        x = make_node_features(ts_cat, mode=FEATURES, pca_dim=PCA_DIM)

        from torch_geometric.data import Data
        g = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        g.num_nodes = x.size(0)
        g.atlas = ATLAS
        g.n_rois = x.size(0)
        g.topk = TOPK_PER_NODE
        g.features = FEATURES

        torch.save(g, out_path)
        return ("ok", sid, (g.num_nodes, g.edge_index.size(1), int(g.y.item())))
    except Exception as e:
        return ("err", sid, str(e))

# ------------------ RUN PARALLEL ---------------------------------------------
tasks = [(sid, by_subj[(sid, n)], labels[sid]) for (sid, n) in subjects_sorted]
results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(
    delayed(process_subject)(sid, bolds, y) for sid, bolds, y in tqdm(tasks)
)

# ------------------ SUMMARY ---------------------------------------------------
ok = sum(1 for s,_,_ in results if s == "ok")
skip = sum(1 for s,_,_ in results if s == "skip")
empty = sum(1 for s,_,_ in results if s == "empty")
err = [info for s,_,info in results if s == "err"]

manifest = {
    "fmriprep_dir": FMRIPREP_DIR,
    "atlas": ATLAS,
    "n_rois": N_ROIS,
    "standardize_ts": STANDARDIZE_TS,
    "topk_per_node": TOPK_PER_NODE,
    "features": FEATURES,
    "pca_dim": PCA_DIM,
    "num_subjects": num_subjects,
    "num_graphs_saved": ok,
    "num_skipped_existing": skip,
    "num_empty": empty,
    "num_errors": len(err),
    "labeling_rule": f"first {NUM_CASES} subjects (numeric sort) => case=1, rest control=0",
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("Summary:", manifest)
if err:
    print("Errors:")
    for e in err[:10]:
        print(" -", e)






