#!/usr/bin/env python3

#IMPORTS
import argparse
import os
import time
from pathlib import Path
import gc,torch
import sys, json, platform, re, shutil, math, random
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_add_pool
from torch_geometric.datasets import Planetoid, TUDataset, ZINC as ZINC_DS
import subprocess, threading
from contextlib import contextmanager
from torch_geometric.data import Batch
from ogb.nodeproppred import PygNodePropPredDataset
import collections
from torch_geometric.datasets import TUDataset as TUDS
from torch_geometric.utils import degree
import torch_sparse # or pyg_lib 
from torch_geometric.nn import GINConv
import csv
from torch_geometric.datasets import Reddit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from types import SimpleNamespace
from statistics import median
from torch_geometric.datasets import ZINC as ZINC_DS
import traceback
from copy import deepcopy

# all other imports (torch, pyg, codecarbon, etc.)

#ROOT = Path(__file__).parent.resolve()
#DATA_DIR = ROOT / "data"
#RESULTS_DIR = ROOT / "results"
#RESULTS_DIR.mkdir(exist_ok=True)

#LOGIC
def _cleanup_cuda():
	"""Free all cached CUDA memory and Python refs."""
	if torch.cuda.is_available():
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
	gc.collect()

# Optional OGB (not used here by default; Jetson wheels vary)
try:
	from ogb.nodeproppred import PygNodePropPredDataset
	HAVE_OGB = True
except Exception:
	HAVE_OGB = False

# Repro
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Device:", DEVICE, "| Torch:", torch.__version__, "| CUDA:", torch.version.cuda)

# Folders
Path("outputs").mkdir(exist_ok=True)
Path("traces").mkdir(exist_ok=True)
Path("eigs_cache").mkdir(exist_ok=True)

# ==== Config ====
CONFIG = {
	"epochs": 50,               # node-level
	"epochs_graph": 80,         # graph-level
	"batch": 64,
	"hidden": 128,
	"layers": 3,
	"heads": 4,
	"dropout": 0.5,
	"seed": SEED,

	# Precision & timing
	"precision": "fp32",
	"reps": 200,
	"warmup": 20,
	"min_infer_time_s": 3.0,
	"tiny_graph_threshold": 8000,

	# ds002748 (folder of precomputed graph .pt files)
	"ds002748_graph_dir": "./data/200/fmriprep_graphs_pt",
}

def _model_device(model=None):
	if model is not None:
		try:
			return next(model.parameters()).device
		except StopIteration:
			pass
	# Fallback to current device
	return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

def reset_gpu_peak(model=None):
	if torch.cuda.is_available():
		dev = _model_device(model)
		#print(dev) # returns cuda:0
		torch.cuda.reset_peak_memory_stats(device=dev)

def gpu_peak_mb(model=None):
	if torch.cuda.is_available():
		dev = _model_device(model)
		#print(dev) # returns cuda:0
		torch.cuda.synchronize(device=dev)
		return torch.cuda.max_memory_allocated(device=dev) / 1e6  # MB
	return float('nan')

def gpu_reserved_peak_mb(model=None):
	if torch.cuda.is_available():
		dev = _model_device(model)
		torch.cuda.synchronize(device=dev)
		return torch.cuda.max_memory_reserved(device=dev) / 1e6  # MB
	return float('nan')

# ==== Robust TegraStatsSampler (Jetson AGX Orin) ====

class TegraStatsSampler:
	"""
	Background reader for `tegrastats`.

	interval_ms: sampling interval (default 200 ms)
	rail:
	  - "auto" (default): prefer VDD_GPU_SOC + VDD_CPU_CV sum, else POM_5V_IN, else POM_5V_GPU, else GPU mW.
	  - "orin_sum": VDD_GPU_SOC + VDD_CPU_CV
	  - "VDD_GPU_SOC", "VDD_CPU_CV", "POM_5V_IN", "POM_5V_GPU", "GPU" for explicit rails
	save_trace_path: optional JSONL storing raw parsed samples.
	"""
	def __init__(self, interval_ms=200, rail="auto", save_trace_path=None):
		self.interval_ms = int(interval_ms)
		self.rail = str(rail)
		self.save_trace_path = save_trace_path
		self.proc = None
		self._thr = None
		self._stop = threading.Event()
		self._samples = []
		self._lock = threading.Lock()

		self.re_vdd_gpu_soc = re.compile(r"\bVDD_GPU_SOC\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_vdd_cpu_cv  = re.compile(r"\bVDD_CPU_CV\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_5v_in       = re.compile(r"\bPOM_5V_IN\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_5v_gpu      = re.compile(r"\bPOM_5V_GPU\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_gpu_fallback= re.compile(r"\bGPU\b[^\n]*?(\d+(?:\.\d+)?)(m?W)\b", re.I)

	@staticmethod
	def _to_watts(val_str, unit):
		v = float(val_str); unit = (unit or "").lower()
		return v/1000.0 if unit == "mw" else v

	def __enter__(self):
		tegra = shutil.which("tegrastats") or "/usr/bin/tegrastats"
		if not Path(tegra).exists():
			print("[tegrastats] not found at", tegra)
			return self
		cmd = [tegra, "--interval", str(self.interval_ms)]
		self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
									 text=True, bufsize=1, universal_newlines=True)
		self._thr = threading.Thread(target=self._reader, daemon=True); self._thr.start()
		time.sleep(max(0.001, self.interval_ms/1000.0))  # allow first line
		return self

	def _parse_line(self, line: str):
		out = {"VDD_GPU_SOC": None, "VDD_CPU_CV": None, "POM_5V_IN": None, "POM_5V_GPU": None, "GPU": None}
		m = self.re_vdd_gpu_soc.search(line);  out["VDD_GPU_SOC"] = self._to_watts(*m.groups()) if m else None
		m = self.re_vdd_cpu_cv.search(line);   out["VDD_CPU_CV"]  = self._to_watts(*m.groups()) if m else None
		m = self.re_5v_in.search(line);        out["POM_5V_IN"]   = self._to_watts(*m.groups()) if m else None
		m = self.re_5v_gpu.search(line);       out["POM_5V_GPU"]  = self._to_watts(*m.groups()) if m else None
		m = self.re_gpu_fallback.search(line); out["GPU"]         = self._to_watts(*m.groups()) if m else None
		return out

	def _pick_power(self, rails: dict):
		mode = self.rail.lower()
		if mode in ("orin_sum","auto"):
			g, c = rails.get("VDD_GPU_SOC"), rails.get("VDD_CPU_CV")
			if (g is not None) or (c is not None):
				return (g or 0.0) + (c or 0.0)
			if mode != "auto": return None
		if mode in ("pom_5v_in","auto"):
			v = rails.get("POM_5V_IN")
			if v is not None: return v
			if mode != "auto": return None
		if mode in ("pom_5v_gpu","auto"):
			v = rails.get("POM_5V_GPU")
			if v is not None: return v
			if mode != "auto": return None
		if mode == "vdd_gpu_soc":
			return rails.get("VDD_GPU_SOC")
		if mode == "vdd_cpu_cv":
			return rails.get("VDD_CPU_CV")
		if mode in ("gpu","auto"):
			v = rails.get("GPU")
			if v is not None: return v
		return None

	def _reader(self):
		try:
			while not self._stop.is_set():
				if self.proc is None: break
				line = self.proc.stdout.readline()
				if not line:
					if self.proc.poll() is not None: break
					time.sleep(0.01); continue
				rails = self._parse_line(line)
				p = self._pick_power(rails)
				if p is not None:
					with self._lock: self._samples.append((time.time(), p))
					if self.save_trace_path:
						with open(self.save_trace_path, "a") as f:
							f.write(json.dumps({"ts": time.time(), "power_w": p, "rails": rails, "raw": line.strip()})+"\n")
		except Exception as e:
			print("[tegrastats] reader error:", e)

	def __exit__(self, *exc):
		time.sleep(max(0.001, self.interval_ms/1000.0))
		self._stop.set()
		try:
			if self.proc:
				self.proc.terminate()
				try: self.proc.wait(timeout=1.0)
				except subprocess.TimeoutExpired: self.proc.kill()
		except Exception: pass
		if self._thr and self._thr.is_alive():
			try: self._thr.join(timeout=1.0)
			except Exception: pass

		if not self._samples:
			# single-shot fallback to avoid NaN
			try:
				tegra = shutil.which("tegrastats") or "/usr/bin/tegrastats"
				out = subprocess.check_output([tegra, "--interval", "1000"], timeout=1.2, text=True)
				for line in out.splitlines():
					rails = self._parse_line(line)
					p = self._pick_power(rails)
					if p is not None:
						self._samples.append((time.time(), p))
						break
			except Exception: pass

	def mean_power(self):
		with self._lock:
			if not self._samples: return float("nan")
			return float(sum(p for _,p in self._samples)/len(self._samples))
# ==== TRAIN METRICS UTILS ====


def _percentiles(ms_list, p):
	if not ms_list: return None
	arr = np.array(ms_list, dtype=np.float64)
	return float(np.percentile(arr, p))

@contextmanager
def train_energy_timer(power_sampler):
	t0 = time.time()
	try:
		yield
	finally:
		globals()["_train_wall_secs"] = time.time() - t0

def _finalize_train_metrics(seen_examples, step_times_ms, power_sampler):
	wall = globals().get("_train_wall_secs", None)
	mean_pwr = power_sampler.mean_power() if power_sampler is not None else None
	energy_total_j = (mean_pwr * wall) if (mean_pwr is not None and wall is not None) else None
	thr = (seen_examples / wall) if (wall and seen_examples>0) else None
	avg_step = (np.mean(step_times_ms) if step_times_ms else None)
	p50 = _percentiles(step_times_ms, 50) if step_times_ms else None
	p95 = _percentiles(step_times_ms, 95) if step_times_ms else None
	per_example = (energy_total_j/seen_examples) if (energy_total_j and seen_examples>0) else None
	return dict(
		train_total_time_s=wall,
		train_mean_power_w=mean_pwr,
		train_energy_total_j=energy_total_j,
		train_energy_per_example_j=per_example,
		train_throughput_examples_s=thr,
		train_avg_step_ms=avg_step,
		train_step_p50_ms=p50,
		train_step_p95_ms=p95,
		train_seen_examples=int(seen_examples),
	)
# --- helper: duplicate a full node-level graph N times into a PyG Batch ---


def nodelevel_infer_batch(data, n: int):
	"""
	Duplicate a single full-graph node dataset (e.g., Cora, PubMed) n times
	and stack into a batched graph (Batch).
	Preserves eigvec/eigval if present (for SAN/GraphGPS).
	"""
	copies = []
	for _ in range(n):
		d = data.clone()
		# clear caches that depend on num_nodes
		for attr in ["_cached_edge_index", "_cached_adj_t", "_num_nodes"]:
			if hasattr(d, attr): setattr(d, attr, None)
		# keep spectral features if present
		if hasattr(data, "eigvec"): d.eigvec = getattr(data, "eigvec", None)
		if hasattr(data, "eigval"): d.eigval = getattr(data, "eigval", None)
		copies.append(d)

	batched = Batch.from_data_list(copies)
	# PyG will concatenate node-wise attributes (like eigvec) along dim 0.
	# Ensure eigval is kept as-is (same eigenvalues for each copy):
	if hasattr(data, "eigval") and data.eigval is not None:
		batched.eigval = data.eigval  # not repeated (shared spectrum)
	return batched		

def check_labels(graphs, out_dim=None):
	vals=[]
	for g in graphs:
		y = g.y.view(-1)[0].item()
		vals.append(float(y))
	uniq = sorted(set(vals))
	cnt  = collections.Counter(vals)
	print("Unique labels:", uniq)
	print("Counts:", dict(cnt))
	if out_dim is not None:
		bad = [v for v in uniq if (math.isnan(v) or v<0 or v>=out_dim)]
		if bad:
			print("!! Out-of-range for CE given n_classes =", out_dim, " ->", bad)
# ==== EIGEN CACHE (Jetson) ====
EIGS_ROOT = Path("eigs_cache")
EIG_K = 16  # must match your cached k

def _norm_name(name: str) -> str:
	mapping = {
		"cora":"Cora", "pubmed":"PubMed", "reddit":"Reddit",
		"mutag":"MUTAG", "zinc":"ZINC", "collab":"COLLAB",
		"ds002748":"ds002748"
	}
	key = re.sub(r"[^A-Za-z0-9_]+", "", str(name)).lower()
	return mapping.get(key, str(name))

def eig_cache_dir(dataset_name: str, k: int = EIG_K) -> Path:
	return EIGS_ROOT / f"{_norm_name(dataset_name)}_k{k}"

def have_node_eigs(dataset_name: str, k: int = EIG_K) -> bool:
	return (eig_cache_dir(dataset_name,k) / "nodegraph.pt").exists()

def attach_node_eigs_from_cache(data, dataset_name: str, k: int = EIG_K, strict: bool = True):
	f = eig_cache_dir(dataset_name,k) / "nodegraph.pt"
	if not f.exists():
		msg = f"[EIG] Missing node eigens: {f}"
		if strict: raise FileNotFoundError(msg)
		print(msg); return data
	payload = torch.load(f, map_location="cpu")
	ev = payload["eigvec"]; ew = payload["eigval"]
	tgt_dtype = getattr(data.x, "dtype", torch.float32); tgt_dev = getattr(data.x, "device", torch.device("cpu"))
	data.eigvec = ev.to(dtype=tgt_dtype, device=tgt_dev)
	data.eigval = ew.to(dtype=tgt_dtype, device=tgt_dev)
	return data

def attach_graph_eigs_list_from_cache(dataset, dataset_name: str, k: int = EIG_K, strict: bool = True):
	d = eig_cache_dir(dataset_name,k)
	out = []; missing = []
	for i in range(len(dataset)):
		f = d / f"graph_{i}.pt"
		g = dataset[i]
		if not f.exists():
			missing.append(i)
			if strict:
				raise FileNotFoundError(f"[EIG] Missing graph eigens for {dataset_name} idx={i}: {f}")
			out.append(g); continue
		payload = torch.load(f, map_location="cpu")
		tgt_dtype = getattr(g.x, "dtype", torch.float32)
		g.eigvec = payload["eigvec"].to(dtype=tgt_dtype)
		g.eigval = payload["eigval"].to(dtype=tgt_dtype)
		out.append(g)
	if missing:
		print(f"[EIG] WARNING: {len(missing)} graphs missing eigens for {dataset_name} (first few: {missing[:5]}).")
	return out

def eig_status_report(k: int = EIG_K):
	rep = {}
	for name in ["Cora","PubMed","Reddit","MUTAG","ZINC","COLLAB","ds002748"]:
		try:
			if name in ["Cora","PubMed","Reddit"]:
				ok = have_node_eigs(name, k)
				rep[name] = {"type":"node","cached": ok, "dir": str(eig_cache_dir(name,k))}
			else:
				d = eig_cache_dir(name,k)
				rep[name] = {"type":"graph","cached_files": len(list(d.glob('graph_*.pt'))), "dir": str(d)}
		except Exception as e:
			rep[name] = {"error": str(e)}
	print(json.dumps(rep, indent=2)); return rep

# ==== Attach_graph_eigs_list_from_cache with padding to EIG_K ====
def attach_graph_eigs_list_from_cache(dataset, dataset_name: str, k: int = EIG_K, strict: bool = True):
	"""
	Loads per-graph eigvec/eigval from cache and ensures shapes:
	  eigvec: [num_nodes, k], eigval: [k]
	by padding (zeros) or truncating per graph to match k.
	"""
	d = eig_cache_dir(dataset_name,k)
	out = []; missing = []
	for i in range(len(dataset)):
		f = d / f"graph_{i}.pt"
		g = dataset[i]
		if not f.exists():
			missing.append(i)
			if strict:
				raise FileNotFoundError(f"[EIG] Missing graph eigens for {dataset_name} idx={i}: {f}")
			out.append(g); continue

		payload = torch.load(f, map_location="cpu")
		eigvec = payload["eigvec"]  # [n_i, k_i]
		eigval = payload["eigval"]  # [k_i]
		n_i    = int(g.num_nodes)
		k_i    = int(eigvec.shape[1]) if eigvec.ndim == 2 else 0

		# Pad/truncate to k
		if k_i < k:
			pad_cols = k - k_i
			if eigvec.ndim == 2:
				pad = torch.zeros((n_i, pad_cols), dtype=eigvec.dtype)
				eigvec = torch.cat([eigvec, pad], dim=1)
			else:
				eigvec = torch.zeros((n_i, k), dtype=torch.float32)
			eigval = torch.cat([eigval, torch.zeros((pad_cols,), dtype=eigval.dtype)], dim=0)
		elif k_i > k:
			eigvec = eigvec[:, :k]
			eigval = eigval[:k]

		# Cast to match x dtype (keeps collate happy)
		tgt_dtype = getattr(g.x, "dtype", torch.float32)
		g.eigvec = eigvec.to(dtype=tgt_dtype)
		g.eigval = eigval.to(dtype=tgt_dtype)
		out.append(g)

	if missing:
		print(f"[EIG] WARNING: {len(missing)} graphs missing eigens for {dataset_name} (first few: {missing[:5]}).")
	# Sanity: ensure all eigvec feature dims are k
	try:
		assert all(getattr(g, "eigvec", torch.empty(0, k)).shape[-1] == k for g in out)
	except AssertionError:
		raise RuntimeError(f"[EIG] Inconsistent eigvec width after padding for {dataset_name}.")
	return out

# ==== Dataset Registry ====
DATA_ROOT = Path("data"); DATA_ROOT.mkdir(exist_ok=True)

def get_cora():
	ds = Planetoid(root=str(DATA_ROOT/"Planetoid"), name="Cora")
	return ds, ds[0]

def get_pubmed():
	ds = Planetoid(root=str(DATA_ROOT/"Planetoid"), name="PubMed")
	return ds, ds[0]

def get_reddit():
	# If you actually want Reddit, swap to Reddit dataset class (requires more RAM/VRAM).
	ds = Planetoid(root=str(DATA_ROOT/"Planetoid"), name="CiteSeer")
	return ds, ds[0]

def get_mutag():
	ds = TUDataset(root=str(DATA_ROOT/"TU"), name="MUTAG")
	return ds

def get_zinc():
	ds = ZINC_DS(root=str(DATA_ROOT/"ZINC"), subset=True)  # subset=True for Jetson friendliness
	return ds

def get_collab():
	ds = TUDataset(root=str(DATA_ROOT/"TU"), name="COLLAB")
	return ds

def get_ds002748():
	gdir = Path(CONFIG.get("ds002748_graph_dir","data/ds002748_graphs"))
	if not gdir.exists():
		raise FileNotFoundError(f"ds002748 graphs not found at {gdir}")
	files = sorted(gdir.glob("*.pt"))
	graphs = [torch.load(f) for f in files]
	return graphs

DATASET_REGISTRY = {
	"Cora": get_cora,
	"PubMed": get_pubmed,
	"Reddit": get_reddit,
	"MUTAG": get_mutag,
	"ZINC": get_zinc,
	"COLLAB": get_collab,
	"ds002748": get_ds002748,
}
#print("Datasets available:", list(DATASET_REGISTRY.keys()))

# ---- COLLAB: create degree features if x is missing ----


def _collab_make_feats(g: Data):
	if getattr(g, "x", None) is None or (isinstance(g.x, torch.Tensor) and g.x.numel()==0):
		n = int(g.num_nodes) if g.num_nodes is not None else int(g.edge_index.max().item()+1)
		deg = degree(g.edge_index[0], num_nodes=n, dtype=torch.float32).view(-1,1)  # [N,1]
		g.x = deg  # minimal, reproducible, and fast
	if not torch.is_floating_point(g.x): g.x = g.x.float()
	# Labels: ensure present & long (graph-level)
	if getattr(g, "y", None) is None: g.y = torch.zeros(1, dtype=torch.long)
	elif (not torch.is_floating_point(g.y)) and g.y.dtype != torch.long: g.y = g.y.long()
	return g

def get_collab():
	ds = TUDS(root=str(DATA_ROOT/"TU"), name="COLLAB")
	graphs = []
	for i in range(len(ds)):
		graphs.append(_collab_make_feats(ds[i]))
	return graphs  # list[Data], graph-level

DATASET_REGISTRY["COLLAB"] = get_collab
#print("Datasets:", list(DATASET_REGISTRY.keys()))

# ==== OGBN-ARXIV support (Jetson) ====
# Some Jetson wheels may not have OGB; this cell keeps things optional.
try:
	from ogb.nodeproppred import PygNodePropPredDataset
	HAVE_OGB = True
except Exception:
	HAVE_OGB = False

def _have_neighbor_backend():
	# NeighborLoader requires either pyg-lib or torch-sparse
	try:
		import pyg_lib  # noqa
		return True, "pyg-lib"
	except Exception:
		try:
			import torch_sparse  # noqa
			return True, "torch-sparse"
		except Exception:
			return False, None

def get_ogbn_arxiv():
	if not HAVE_OGB:
		raise RuntimeError("OGB not installed on this environment.")
	d = PygNodePropPredDataset("ogbn-arxiv", root=str(DATA_ROOT/"OGB"))
	data = d[0]
	split_idx = d.get_idx_split()
	return d, data, split_idx

# Register conditionally (so your registry print still works on Jetson without OGB)
if HAVE_OGB:
	DATASET_REGISTRY["ogbn-arxiv"] = get_ogbn_arxiv

print("Datasets available:", list(DATASET_REGISTRY.keys()))

# ==== Models ====
class GCN(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.5, **kw):
		super().__init__()
		self.convs = nn.ModuleList()
		dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
		for i in range(len(dims)-1): self.convs.append(GCNConv(dims[i], dims[i+1]))
		self.dropout = nn.Dropout(dropout); self.act = nn.ReLU()
	def forward(self, x, edge_index, batch=None, **kwargs):
		for conv in self.convs[:-1]: x = self.dropout(self.act(conv(x, edge_index)))
		return self.convs[-1](x, edge_index)

class GraphSAGE(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.5, **kw):
		super().__init__()
		self.convs = nn.ModuleList()
		dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
		for i in range(len(dims)-1): self.convs.append(SAGEConv(dims[i], dims[i+1]))
		self.dropout = nn.Dropout(dropout); self.act = nn.ReLU()
	def forward(self, x, edge_index, batch=None, **kwargs):
		for conv in self.convs[:-1]: x = self.dropout(self.act(conv(x, edge_index)))
		return self.convs[-1](x, edge_index)

class GAT(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=2, heads=4, dropout=0.5, **kw):
		super().__init__()
		self.layers = nn.ModuleList(); self.heads = heads
		dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
		for i in range(len(dims)-1):
			h = heads if i < layers-1 else 1
			self.layers.append(GATConv(dims[i], dims[i+1]//h, heads=h, dropout=dropout))
		self.dropout = nn.Dropout(dropout); self.act = nn.ReLU()
	def forward(self, x, edge_index, batch=None, **kwargs):
		for layer in self.layers[:-1]: x = self.dropout(self.act(layer(x, edge_index)))
		return self.layers[-1](x, edge_index)

class GIN(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=3, dropout=0.5, **kw):
		super().__init__()
		
		self.layers = nn.ModuleList()
		def mlp(in_c, out_c): return nn.Sequential(nn.Linear(in_c, hidden), nn.ReLU(), nn.Linear(hidden, out_c))
		dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
		for i in range(len(dims)-1): self.layers.append(GINConv(mlp(dims[i], dims[i+1])))
		self.dropout = nn.Dropout(dropout); self.act = nn.ReLU()
	def forward(self, x, edge_index, batch=None, **kwargs):
		for layer in self.layers[:-1]: x = self.dropout(self.act(layer(x, edge_index)))
		return self.layers[-1](x, edge_index)

# --- Minimal SAN_Full (needs eigvec/eigval) outputs hidden embeddings ---
class SAN_Full(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=3, heads=4, k=16, dropout=0.5, **kw):
		super().__init__()
		self.k = k
		self.proj_in = nn.Linear(in_dim, hidden)
		self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)) for _ in range(layers)])
		self.proj_out = nn.Linear(hidden, hidden)
		self.act = nn.ReLU(); self.dropout = nn.Dropout(dropout)
	def forward(self, x, edge_index, batch=None, eigvec=None, eigval=None):
		x = self.act(self.proj_in(x)); x = self.dropout(x)
		if eigvec is None or eigval is None:
			raise RuntimeError("SAN_Full needs eigvec/eigval.")
		if eigvec.dim()==2:
			spec = eigvec[:, :self.k]
			gate = torch.sigmoid(torch.matmul(spec, spec.transpose(0,1)).diagonal().unsqueeze(-1))
			x = x * gate.clamp(0.5, 1.5)
		for blk in self.blocks: x = blk(x)
		return self.proj_out(x)

# --- Minimal GraphGPS placeholder (no LapPE, so no eigens) ---
class GraphGPS_Full(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, layers=3, dropout=0.5, **kw):
		super().__init__()
		self.proj_in = nn.Linear(in_dim, hidden)
		self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)) for _ in range(layers)])
		self.proj_out = nn.Linear(hidden, hidden)
		self.act = nn.ReLU(); self.dropout = nn.Dropout(dropout)
	def forward(self, x, edge_index, batch=None, **kwargs):
		x = self.act(self.proj_in(x)); x = self.dropout(x)
		for blk in self.blocks: x = blk(x)
		return self.proj_out(x)

MODEL_REGISTRY = {
	"gcn": GCN,
	"gat": GAT,
	"graphsage": GraphSAGE,
	"gin": GIN,
	"san_full": SAN_Full,
	"graphgps_full": GraphGPS_Full,
}
#print("Models:", list(MODEL_REGISTRY.keys()))

# ==== Helpers: AMP, forward shims, training, timing ====
def amp_context(precision: str):
	use_fp16 = (precision or "").lower() == "fp16" and DEVICE.type=="cuda"
	return torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()

def _call_model(model, x, edge_index, batch, obj, precision=None):
	kwargs = {}
	ev = getattr(obj, "eigvec", None); ew = getattr(obj, "eigval", None)
	if ev is not None: kwargs["eigvec"] = ev.to(dtype=x.dtype, device=x.device)
	if ew is not None: kwargs["eigval"] = ew.to(dtype=x.dtype, device=x.device)
	with amp_context(precision or "fp32"):
		return model(x, edge_index, batch, **kwargs)

def _model_dtype(model):
	for p in model.parameters():
		return p.dtype
	return torch.float32

def node_forward_fnOLD(model, data, precision=None):
	target_dtype = _model_dtype(model)
	def _fn():
		with torch.no_grad():
			with amp_context(precision):
				x = data.x.to(target_dtype) if target_dtype==torch.float16 else data.x
				_ = model(x, data.edge_index, None,
						  eigvec=getattr(data,'eigvec',None),
						  eigval=getattr(data,'eigval',None))
	return _fn

is_node=False
# --- patched node_forward_fn: adds infer_batch and AMP support ---
def node_forward_fn(model, data, *, precision="fp32", infer_batch: int = 1):
	"""
	Returns a closure `fn()` that runs a forward pass for timing.
	If infer_batch > 1, we make a batched copy of the full graph (n copies).
	Energy per inference should then be divided by infer_batch upstream.
	"""
	use_amp = (precision.lower() == "fp16" and (hasattr(torch, "cuda") and torch.cuda.is_available()))
	target_dtype = torch.float16 if use_amp else None

	# prepare data (single or batched)
	if infer_batch > 1:
		batch_graph = nodelevel_infer_batch(data, infer_batch).to(DEVICE)
	else:
		batch_graph = data.to(DEVICE)

	# if FP16, cast features only (keep weights in FP32 unless you explicitly half() the model)
	def _cast_x(x):
		return x.to(torch.float16) if target_dtype == torch.float16 else x

	def _fn():
		with torch.no_grad():
			ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
			with ctx:
				x = _cast_x(batch_graph.x)
				# Many of your node models accept (x, edge_index) only; batch is ignored for full-graph.
				# If your model signature needs 'batch', it will be present for infer_batch>1.
				try:
					model(x, batch_graph.edge_index, getattr(batch_graph, "batch", None),
						  eigvec=getattr(batch_graph, "eigvec", None),
						  eigval=getattr(batch_graph, "eigval", None))
				except TypeError:
					# Fallback for models that only take (x, edge_index)
					model(x, batch_graph.edge_index)
	return _fn





def graph_forward_fn(model, batch, precision=None):
	target_dtype = _model_dtype(model)
	def _fn():
		with torch.no_grad():
			with amp_context(precision):
				x = batch.x.to(target_dtype) if target_dtype==torch.float16 else batch.x
				_ = model(x, batch.edge_index, batch.batch,
						  eigvec=getattr(batch,'eigvec',None),
						  eigval=getattr(batch,'eigval',None))
	return _fn

def run_inference_timed(fn, reps=50, warmup=10, min_time_s=2.0):
	for _ in range(int(warmup)):
		fn()
		if DEVICE.type=="cuda": torch.cuda.synchronize()
	ts=[]; start=time.time(); it=0
	while it<int(reps) or (time.time()-start)<float(min_time_s):
		t0=time.time(); fn()
		if DEVICE.type=="cuda": torch.cuda.synchronize()
		ts.append((time.time()-t0)*1000.0); it+=1
	ts.sort(); p50=ts[len(ts)//2]; p95=ts[int(0.95*(len(ts)-1))]
	thr=1000.0/(sum(ts)/len(ts))
	return {"latency_p50_ms":p50,"latency_p95_ms":p95,"throughput_req_s":thr}

def make_optimizer(model, lr=1e-3, wd=5e-4):
	return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def evaluate_node_acc(model, data, precision="fp32"):
	model.eval()
	with torch.no_grad():
		logits = _call_model(model, data.x, data.edge_index, None, data, precision=precision)
		pred = logits.argmax(-1)
		if hasattr(data,"test_mask"):
			return (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
		return float("nan")

def train_node_levelOLD(model, data, epochs=50, precision="fp32"):
	model = model.to(DEVICE); data = data.to(DEVICE)
	opt = make_optimizer(model)
	scaler = torch.amp.GradScaler("cuda", enabled=(precision.lower()=="fp16" and DEVICE.type=="cuda"))
	ce = nn.CrossEntropyLoss()
	train_idx = data.train_mask if hasattr(data,"train_mask") else torch.arange(data.num_nodes, device=DEVICE)
	t0=time.time(); best_val=-1.0
	for _ in range(int(epochs)):
		model.train(); opt.zero_grad(set_to_none=True)
		with amp_context(precision):
			out = _call_model(model, data.x, data.edge_index, None, data, precision=precision)
			loss = ce(out[train_idx], data.y[train_idx].long())
		if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
		else: loss.backward(); opt.step()
		# val
		model.eval()
		with torch.no_grad():
			logits = _call_model(model, data.x, data.edge_index, None, data, precision=precision)
			pred = logits.argmax(-1)
			if hasattr(data,"val_mask"):
				val_acc = (pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()
				best_val = max(best_val, val_acc)
	total_t = time.time()-t0
	if best_val < 0: best_val = evaluate_node_acc(model, data, precision=precision)
	return {"train_total_time_s": total_t, "val_metric_final": best_val}

def train_graph_levelOLD(model, dataset, epochs=80, precision="fp32", batch=32):
	loader = PyGDataLoader(dataset, batch_size=int(batch), shuffle=True)
	model = model.to(DEVICE)
	opt = make_optimizer(model)
	scaler = torch.amp.GradScaler("cuda", enabled=(precision.lower()=="fp16" and DEVICE.type=="cuda"))
	ce = nn.CrossEntropyLoss()
	t0=time.time()
	for _ in range(int(epochs)):
		model.train()
		for b in loader:
			b = b.to(DEVICE)
			opt.zero_grad(set_to_none=True)
			with amp_context(precision):
				out = _call_model(model, b.x, b.edge_index, b.batch, b, precision=precision)
				if out.dim()==1: out = out.unsqueeze(1)
				loss = ce(out, b.y.view(-1).long())
			if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
			else: loss.backward(); opt.step()
	return {"train_total_time_s": time.time()-t0}

#NEW
def train_node_level(model, data, epochs=5, precision="fp32"):
	model = model.to(DEVICE); data = data.to(DEVICE) #copiata da vecchia
	model.train()
	ce  = torch.nn.CrossEntropyLoss()
	opt = make_optimizer(model)
	scaler = torch.amp.GradScaler("cuda", enabled=(precision.lower()=="fp16" and DEVICE.type=="cuda"))
	amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16) if (precision.lower()=="fp16" and DEVICE.type=="cuda") else nullcontext())

	# indices
	if getattr(data, "train_mask", None) is not None:
		train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
	else:
		train_idx = torch.arange(data.num_nodes, device=data.x.device)

	step_times_ms, seen_examples = [], 0
	os.makedirs("traces", exist_ok=True)
	train_trace_path = os.path.join("traces", "train_node__pwr.jsonl")
	reset_gpu_peak()
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=train_trace_path) as mon, \
		 train_energy_timer(mon):
		for _ in range(int(epochs)):
			t0 = time.time()
			opt.zero_grad(set_to_none=True)
			with amp_ctx:
				out = model(data.x, data.edge_index)
				loss = ce(out[train_idx], data.y[train_idx].long())
			if scaler.is_enabled():
				scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
			else:
				loss.backward(); opt.step()
			step_times_ms.append((time.time()-t0)*1000.0)
			seen_examples += int(train_idx.numel())
	res = _finalize_train_metrics(seen_examples, step_times_ms, mon)
	#if torch.cuda.is_available():
		#dev = _model_device(model)
		#print("[mem dbg] current:", torch.cuda.current_device(), "dev:", dev)
		#print("[mem dbg] allocated:", torch.cuda.memory_allocated(dev)/1e6, "MB",
		#  "reserved:", torch.cuda.memory_reserved(dev)/1e6, "MB")
		#print(torch.cuda.memory_summary(device=dev, abbreviated=True))

	res["train_gpu_mem_peak_mb"] = gpu_peak_mb()
	res["train_gpu_mem_reserved_peak_mb"] = gpu_reserved_peak_mb()
	return res
	#return _finalize_train_metrics(seen_examples, step_times_ms, mon)

#NEW
def train_graph_level(model, dataset, epochs=5, precision="fp32", batch=64):
	loader = PyGDataLoader(dataset, batch_size=int(batch), shuffle=True, drop_last=False)
	model = model.to(DEVICE) #copiato da vecchia
	ce  = torch.nn.CrossEntropyLoss()
	opt = make_optimizer(model)
	scaler = torch.amp.GradScaler("cuda", enabled=(precision.lower()=="fp16" and DEVICE.type=="cuda"))
	amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16) if (precision.lower()=="fp16" and DEVICE.type=="cuda") else nullcontext())

	step_times_ms, seen_examples = [], 0
	os.makedirs("traces", exist_ok=True)
	train_trace_path = os.path.join("traces", "train_graph__pwr.jsonl")
	reset_gpu_peak()
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=train_trace_path) as mon, \
		 train_energy_timer(mon):
		for _ in range(int(epochs)):
			for b in loader:
				b = b.to(DEVICE)
				t0 = time.time()
				opt.zero_grad(set_to_none=True)
				with amp_ctx:
					out = _call_model(model, b.x, b.edge_index, b.batch, b, precision=precision)
					# ZINC regression vs classification:
					if out.dim()==1: out = out.unsqueeze(1)
					if b.y.dim()>1 and out.size(-1)==1:
						loss = torch.nn.functional.smooth_l1_loss(out.view(-1), b.y.view(-1).float())
					else:
						loss = ce(out, b.y.view(-1).long())
				if scaler.is_enabled():
					scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
				else:
					loss.backward(); opt.step()
				step_times_ms.append((time.time()-t0)*1000.0)
				seen_examples += int(b.y.view(-1).size(0))
	res = _finalize_train_metrics(seen_examples, step_times_ms, mon)
	res["train_gpu_mem_peak_mb"] = gpu_peak_mb()
	res["train_gpu_mem_reserved_peak_mb"] = gpu_reserved_peak_mb()
	return res            
	#return _finalize_train_metrics(seen_examples, step_times_ms, mon)	

# ==== ZINC eigens: compute/cache per-graph Laplacian eigenpairs (robust) ====
 

EIG_ROOT = Path("eigs_cache")

def eig_cache_dir(name: str, k: int) -> Path:
	d = EIG_ROOT / f"{name}_k{k}"
	d.mkdir(parents=True, exist_ok=True)
	return d

def _lap_eigs_dense(edge_index: torch.Tensor, num_nodes: int, k: int):
	"""
	Build symmetric normalized Laplacian and compute first k eigenpairs on CPU (robust for tiny graphs).
	Returns (eigvec [N, k’], eigval [k’]) where k’<=k (caller pads).
	"""
	device = torch.device("cpu")
	edge_index = edge_index.to(device)
	row, col = edge_index
	deg = torch.bincount(row, minlength=num_nodes).float()
	deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
	A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
	A[row, col] = 1.0
	A[col, row] = 1.0
	Dm12 = torch.diag(deg_inv_sqrt)
	L = torch.eye(num_nodes, dtype=torch.float32, device=device) - Dm12 @ A @ Dm12
	# eigh is symmetric & stable
	eigval, eigvec = torch.linalg.eigh(L)  # ascending
	k_use = min(k, eigval.numel())
	return eigvec[:, :k_use].contiguous(), eigval[:k_use].contiguous()

def _pad_cols(mat: torch.Tensor, target_k: int, pad_value: float = 0.0):
	if mat.size(1) == target_k:
		return mat
	if mat.size(1) < target_k:
		pad = torch.full((mat.size(0), target_k - mat.size(1)), pad_value, dtype=mat.dtype)
		return torch.cat([mat, pad], dim=1)
	return mat[:, :target_k].contiguous()

def _pad_len(vec: torch.Tensor, target_k: int, pad_value: float = 1e9):
	if vec.numel() == target_k:
		return vec
	if vec.numel() < target_k:
		pad = torch.full((target_k - vec.numel(),), pad_value, dtype=vec.dtype)
		return torch.cat([vec, pad], dim=0)
	return vec[:target_k].contiguous()

def compute_and_attach_eigs_list(graphs, dataset_name: str, k: int = 16, strict: bool = False):
	"""
	Ensures EVERY graph in `graphs` has:
	  - g.eigvec: [num_nodes, k] float32
	  - g.eigval: [k] float32
	Uses on-disk caching; never leaves a graph without eigs (unless strict=True and a hard failure occurs).
	"""
	cache = eig_cache_dir(dataset_name, k)
	out = []
	for i, g in enumerate(graphs):
		num_nodes = int(g.num_nodes) if g.num_nodes is not None else (
			int(g.edge_index.max().item()+1) if g.edge_index.numel() > 0 else 1
		)
		fvec = cache / f"graph_{i}_vec.pt"
		fval = cache / f"graph_{i}_val.pt"

		eigvec = None
		eigval = None
		if fvec.exists() and fval.exists():
			try:
				eigvec = torch.load(fvec, map_location="cpu")
				eigval = torch.load(fval, map_location="cpu")
			except Exception:
				eigvec = None; eigval = None  # fall through to recompute

		if eigvec is None or eigval is None:
			try:
				vec, val = _lap_eigs_dense(g.edge_index, num_nodes, k)
			except Exception as e:
				if strict:
					raise
				# safe fallback: identity columns + zeros (keeps dims correct; accuracy won’t benefit)
				vec = torch.eye(num_nodes, dtype=torch.float32)[:, :min(k, num_nodes)]
				val = torch.zeros(vec.size(1), dtype=torch.float32)
			# pad/truncate to fixed k columns
			eigvec = _pad_cols(vec, k, pad_value=0.0)
			eigval = _pad_len(val, k, pad_value=1e9)
			try:
				torch.save(eigvec, fvec)
				torch.save(eigval, fval)
			except Exception:
				pass  # cache write errors are non-fatal

		# Final guards: ensure row count equals num_nodes and k columns
		if eigvec.size(0) != num_nodes:
			# reconstruct minimal identity fallback of correct height
			vec = torch.eye(num_nodes, dtype=torch.float32)[:, :min(k, num_nodes)]
			eigvec = _pad_cols(vec, k, pad_value=0.0)
			eigval = _pad_len(torch.zeros(vec.size(1), dtype=torch.float32), k, pad_value=1e9)

		# attach (CPU float; moved/cast in forward)
		g.eigvec = eigvec.float()
		g.eigval = eigval.float()
		out.append(g)

	return out

def sanity_check_eigs(graphs, k: int):
	bad = 0
	for i, g in enumerate(graphs):
		evc = getattr(g, "eigvec", None)
		evl = getattr(g, "eigval", None)
		if evc is None or evl is None:
			bad += 1; continue
		if evc.dim() != 2 or evc.size(1) != k or evl.dim() != 1 or evl.numel() != k:
			bad += 1; continue
		n = int(g.num_nodes) if g.num_nodes is not None else evc.size(0)
		if evc.size(0) != n:
			bad += 1
	if bad > 0:
		print(f"[EIG-SANITY] {bad} graphs have invalid/missing eigs.")
	else:
		print("[EIG-SANITY] all graphs have valid eigvec/eigval.")

# ==== Helpers: infer dims from list[Data] (for TU datasets like COLLAB) ====
def infer_in_out_dims_from_graphs(graphs):
	"""
	graphs: list[torch_geometric.data.Data]
	returns: in_dim (int), out_dim (int)
	"""
	# in_dim: from .x
	g0 = graphs[0]
	if getattr(g0, "x", None) is None:
		raise RuntimeError("Graph has no .x after sanitation; cannot infer in_dim.")
	in_dim = int(g0.x.size(-1))

	# out_dim: from .y across all graphs (classification: max label + 1)
	ys = []
	for g in graphs:
		y = getattr(g, "y", None)
		if y is None:
			continue
		if y.numel() == 0:
			continue
		y = y.view(-1).long()
		ys.append(y)
	if len(ys) == 0:
		# fallback: binary
		out_dim = 2
	else:
		y_all = torch.cat(ys, dim=0)
		out_dim = int(y_all.max().item()) + 1
	return in_dim, out_dim
	
# ==== PATCH: always feed float tensors to the model (handles ZINC Long x) ====

def _to_model_dtype(t: torch.Tensor, model: torch.nn.Module):
	"""Cast feature tensor to model's parameter dtype (float32/float16)."""
	# Default to first param dtype, else float32
	target = next((p.dtype for p in model.parameters() if p is not None), torch.float32)
	if not torch.is_floating_point(t) or t.dtype != target:
		return t.to(dtype=target)
	return t

def _ensure_float_x(obj):
	"""In-place: ensure obj.x is floating (float32 if no model context)."""
	if hasattr(obj, "x") and not torch.is_floating_point(obj.x):
		obj.x = obj.x.float()
	return obj

# Hook into existing call sites:

def _call_model(model, x, edge_index, batch, obj, precision=None):
	# Ensure eigen tensors are aligned and floating
	kwargs = {}
	ev = getattr(obj, "eigvec", None); ew = getattr(obj, "eigval", None)
	if ev is not None:
		if not torch.is_floating_point(ev): ev = ev.float()
		kwargs["eigvec"] = ev.to(device=x.device)
	if ew is not None:
		if not torch.is_floating_point(ew): ew = ew.float()
		kwargs["eigval"] = ew.to(device=x.device)
	# Cast features to model dtype (float32/float16)
	x = _to_model_dtype(x, model)

	with amp_context(precision or "fp32"):
		return model(x, edge_index, batch, **kwargs)

def node_forward_fnOLD2(model, data, precision=None):
	_ensure_float_x(data)  # make data.x floating upfront
	def _fn():
		with torch.no_grad():
			with amp_context(precision):
				x = _to_model_dtype(data.x, model)
				_ = model(x, data.edge_index, None,
						  eigvec=getattr(data,'eigvec',None),
						  eigval=getattr(data,'eigval',None))
	return _fn

def graph_forward_fn(model, batch, precision=None):
	_ensure_float_x(batch)  # ensure batch.x is floating
	def _fn():
		with torch.no_grad():
			with amp_context(precision):
				x = _to_model_dtype(batch.x, model)
				_ = model(x, batch.edge_index, batch.batch,
						  eigvec=getattr(batch,'eigvec',None),
						  eigval=getattr(batch,'eigval',None))
	return _fn

# Also guard datasets at load time:

def _coerce_dataset_x_float(dataset):
	"""Iterate and cast .x to float for all graphs in a dataset/list."""
	try:
		for i in range(len(dataset)):
			g = dataset[i]
			if hasattr(g, "x") and not torch.is_floating_point(g.x):
				g.x = g.x.float()
	except TypeError:
		# dataset may be an iterator; best effort pass
		pass
	return dataset

# ==== SAN safety wrapper & GraphHead encoder ====
class WithEigs(nn.Module):
	"""Wrap base model to always inject eigvec/eigval for node-level SAN."""
	def __init__(self, base, eigvec, eigval):
		super().__init__()
		self.base = base
		self._eigvec = eigvec
		self._eigval = eigval
	def forward(self, x, edge_index, batch=None, **kwargs):
		kwargs.setdefault("eigvec", self._eigvec)
		kwargs.setdefault("eigval", self._eigval)
		return self.base(x, edge_index, batch, **kwargs)

class GraphHead(nn.Module):
	"""Graph-level: base returns per-node embeddings -> pool -> classify."""
	def __init__(self, base, hidden, out_dim):
		super().__init__()
		self.base = base
		self.readout = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out_dim))
	def forward(self, x, edge_index, batch, **kwargs):
		try: z = self.base(x, edge_index, batch, **kwargs)
		except TypeError: z = self.base(x, edge_index, batch)
		pooled = global_add_pool(z, batch); return self.readout(pooled)

def build_model(name:str, in_dim:int, out_dim:int, cfg:dict):
	name = name.lower(); ctor = MODEL_REGISTRY[name]
	if name == "gat":
		return ctor(in_dim, cfg["hidden"], out_dim, layers=cfg["layers"], heads=cfg.get("heads",4), dropout=cfg["dropout"])
	elif name in ["san_full","graphgps_full"]:
		return ctor(in_dim, cfg["hidden"], out_dim, layers=cfg["layers"], heads=cfg.get("heads",4), dropout=cfg["dropout"])
	else:
		return ctor(in_dim, cfg["hidden"], out_dim, layers=cfg["layers"], dropout=cfg["dropout"])

def build_graph_encoder(name: str, in_dim: int, hidden: int, cfg: dict):
	name = name.lower()
	if name == "gat":
		return MODEL_REGISTRY[name](in_dim, hidden, hidden, layers=cfg["layers"], heads=cfg.get("heads",4), dropout=cfg["dropout"])
	elif name == "san_full":
		return MODEL_REGISTRY[name](in_dim, hidden, hidden, layers=cfg["layers"], heads=cfg.get("heads",4), dropout=cfg["dropout"])
	elif name == "graphgps_full":
		return MODEL_REGISTRY[name](in_dim, hidden, hidden, layers=cfg["layers"], dropout=cfg["dropout"])
	else:
		return MODEL_REGISTRY[name](in_dim, hidden, hidden, layers=cfg["layers"], dropout=cfg["dropout"])

# Show datasets dict one more time
#print("Datasets:", list(DATASET_REGISTRY.keys()))

# Helper: graph-level validation accuracy
def graph_val_acc(dataset, model, batch, precision):
	
	def amp_context(prec):
		if prec == "fp16" and DEVICE.type == "cuda":
			return torch.amp.autocast("cuda", dtype=torch.float16)
		return nullcontext()

	model.eval()
	correct = 0
	total = 0
	loader = PyGDataLoader(dataset, batch_size=batch, shuffle=False)
	with torch.no_grad():
		for b in loader:
			b = b.to(DEVICE)
			with amp_context(precision):
				out = _call_model(model, b.x, b.edge_index, b.batch, b, precision=precision)
			if out.dim() == 1:  # safety
				out = out.unsqueeze(1)
			pred = out.argmax(dim=-1)
			y = b.y.view(-1).long()
			correct += (pred == y).sum().item()
			total += y.numel()
	return (correct / total) if total > 0 else None


# ==== Helper: node-level validation accuracy ====

def node_val_acc(data, model, precision: str = 'fp32'):
	"""
	Compute accuracy on data.val_mask if present; else on data.test_mask; else None.
	Works for single-graph PyG node classification datasets (Cora/PubMed/Reddit/ogbn-arxiv).
	"""
	model.eval()
	with torch.no_grad():
		x = data.x.to(DEVICE)
		edge_index = data.edge_index.to(DEVICE)
		batch = getattr(data, 'batch', None)
		if batch is None:
			batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
		out = _call_model(model, x, edge_index, batch, data, precision=precision)
		if out.dim() == 1:
			out = out.unsqueeze(-1)
		pred = out.argmax(dim=-1)
		y = data.y.view(-1).long().to(DEVICE)
		mask = getattr(data, 'val_mask', None)
		if mask is None or int(mask.sum().item()) == 0:
			mask = getattr(data, 'test_mask', None)
		if mask is None or int(mask.sum().item()) == 0:
			return None
		mask = mask.to(DEVICE)
		correct = int((pred[mask] == y[mask]).sum().item())
		total = int(mask.sum().item())
		return (correct / total) if total > 0 else None

# ==== run_experiment (node & graph, Jetson/tegrastats) ==== GENERIC
def run_experiment(dataset_name: str, model_name: str, cfg: dict, run_name_suffix=""):
	precision = cfg["precision"].lower()
	batch = int(cfg["batch"])
	reps, warmup = int(cfg["reps"]), int(cfg["warmup"])
	run_name = f"jetson_{dataset_name}_{model_name}_{precision}_b{batch}{run_name_suffix}"
	summary_path = os.path.join("traces", run_name + "__summary.json")
	power_trace_path = os.path.join("traces", run_name + "__pwr.jsonl")

	obj = DATASET_REGISTRY[dataset_name]()
	node_level = dataset_name in ["Cora","PubMed","Reddit"]

	# ---- Node-level ----
	if node_level:
		ds, data = obj
		in_dim = data.x.size(-1)
		out_dim = int(data.y.max().item()+1) if data.y.dim()==1 else data.y.size(-1)

		if model_name.lower() == "san_full":
			data = attach_node_eigs_from_cache(data, dataset_name, k=EIG_K, strict=True)
			

		model = build_model(model_name, in_dim, out_dim, cfg)
		if model_name.lower() == "san_full":
			model = WithEigs(model, data.eigvec, data.eigval)
		#AGGIUNTA
		_ensure_float_x(data)
		#reset_gpu_peak()

		# Train
		res_train = train_node_level(model, data, epochs=cfg["epochs"], precision=precision)

		# Inference + power
		if precision=="fp16" and DEVICE.type=="cuda": model = model.half()
		model = model.to(DEVICE).eval()
		val_acc = node_val_acc(data, model, precision)
		
		#VECCHIA INFERENZA
		#fwd = node_forward_fn(model, data.to(DEVICE), precision=precision)

		#local_reps, local_warmup, local_min = reps, warmup, float(CONFIG.get('min_infer_time_s', 3.0))
		#try: n_nodes = int(data.num_nodes)
		#except Exception: n_nodes = None
		#if n_nodes is not None and n_nodes < int(CONFIG.get('tiny_graph_threshold', 8000)):
		#    local_reps = max(local_reps, 200); local_warmup = max(local_warmup, 20); local_min = max(local_min, 3.0)

		#with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=power_trace_path) as mon2:
		#    inf_metrics = run_inference_timed(fwd, reps=local_reps, warmup=local_warmup, min_time_s=local_min)
		#mean_power_w = mon2.mean_power()
		#energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0))
		#FINE VECCHIA INFERENZA

		#NUOVA INFERENZA
		# pull desired inference batch from config; default 1 keeps previous behavior
		infer_b = int(cfg.get("infer_batch", 1))
		print(infer_b)
		# build timing closure with infer-batching + AMP awareness
		fwd = node_forward_fn(model, data, precision=precision, infer_batch=infer_b)
		reset_gpu_peak
		# run timed inference + power
		with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=power_trace_path) as mon2:
			inf_metrics = run_inference_timed(
				fwd,
				reps=int(cfg.get("reps", 60)),
				warmup=int(cfg.get("warmup", 10)),
				min_time_s=float(cfg.get("min_infer_time_s", 2.0))
			)
		infer_gpu_mem_peak_mb = gpu_peak_mb()
		infer_gpu_mem_reserved_peak_mb = gpu_reserved_peak_mb()
		mean_power_w = mon2.mean_power()
		
		# NORMALIZE energy per *item* when infer_b > 1
		lat_s = inf_metrics["latency_p50_ms"] / 1000.0
		energy_per_inf_j = (mean_power_w * lat_s) / max(1, infer_b)
		#FINE NUOVA INFERENZA
		
		
		
		#summary = {
		#    "timestamp": datetime.utcnow().isoformat(),
		#    "platform":"jetson", "device_name": platform.platform(),
		#    "dataset":dataset_name, "model":model_name, "precision":precision, "batch":batch, "seed": cfg["seed"],
		#    "phase":"train+infer_node", **res_train, **inf_metrics,
		#    "mean_power_w": mean_power_w, "energy_per_inf_j": energy_per_inf_j,
		#}
		summary = {
		"timestamp": datetime.utcnow().isoformat(),
		"platform": "jetson",
		"device_name": platform.platform(),
		"dataset": dataset_name, "model": model_name, "precision": precision,
		"batch": batch if isinstance(batch, (int,str)) else str(batch),
		"seed": cfg.get("seed", 42),
		"phase": "train+infer_node" if is_node else "train+infer_graph",
		# === NEW: inject TRAIN metrics ===
		**res_train,
		# === validation / test (whatever the runner computed) ===
		"val_metric_final": float(val_acc) if 'val_acc' in locals() and val_acc is not None else None,
		"test_acc": float(test_acc) if 'test_acc' in locals() and test_acc is not None else "",
		# === inference (as before) ===
		**inf_metrics,                  # has latency_p50_ms, latency_p95_ms, throughput_req_s
		"mean_power_w": mean_power_w,
		"energy_per_inf_j": energy_per_inf_j,
	}
		summary.update({
		"infer_gpu_mem_peak_mb": infer_gpu_mem_peak_mb,
		"infer_gpu_mem_reserved_peak_mb": infer_gpu_mem_reserved_peak_mb,
		})


	# ---- Graph-level ----
	else:
		ds = obj[0] if isinstance(obj, tuple) else obj
		if dataset_name=="ds002748":
			graphs = ds
			check_labels(graphs)
			# After loading ds002748 graphs
			#MODIFICA
			# 1) Compute class set & enforce zero-based ints (you already have 0/1)
			uniq = sorted({int(float(g.y.view(-1)[0].item())) for g in graphs})
			assert uniq == [0,1] or len(uniq) >= 2, f"unexpected labels: {uniq}"
			
			# 2) Force y to long ints and shape [1] per-graph (graph-level CE wants Long indices)
			for g in graphs:
				g.y = g.y.view(-1)[:1].long()
			
			# 3) IMPORTANT: binary classification => out_dim = 2 (not 1)
			#out_dim = 2

			loader_tmp = PyGDataLoader(graphs, batch_size=batch, shuffle=True)
			sample = next(iter(loader_tmp)).to('cpu')
			in_dim = sample.x.size(-1)
			#out_dim = int(sample.y.max().item()+1) if sample.y.dim()==1 else sample.y.size(-1)
			out_dim=2
			print(out_dim)
			
		else:
			sample = ds[0]; in_dim = sample.num_features; out_dim = int(ds.num_classes)
			   
		if model_name.lower()=="san_full":
			if dataset_name != "ds002748":
				graphs = attach_graph_eigs_list_from_cache(ds, dataset_name, k=EIG_K, strict=True)
			else:
				class _Seq:  # to reuse same function
					def __init__(self,L): self.L=L
					def __len__(self): return len(self.L)
					def __iter__(self): return iter(self.L)
					def __getitem__(self,i): return self.L[i]
				graphs = attach_graph_eigs_list_from_cache(_Seq(graphs), "ds002748", k=EIG_K, strict=True)
		else:
			graphs = ds if dataset_name!="ds002748" else graphs
		
		
		if dataset_name == "COLLAB":
			
			graphs = DATASET_REGISTRY["COLLAB"]()
			
			in_dim, out_dim = infer_in_out_dims_from_graphs(graphs)
			
			print(in_dim)
			print(out_dim)
			# features added only for COLLAB
			# If SAN is selected, attach eigs (your padded attach is fine):
			if model_name.lower() == "san_full":
				graphs = attach_graph_eigs_list_from_cache(graphs, "COLLAB", k=EIG_K, strict=False)
				#in_dim, out_dim = infer_in_out_dims_from_graphs(graphs)
				
		
		#NUOVA RIGA
		graphs = _coerce_dataset_x_float(graphs)
		encoder = build_graph_encoder(model_name, in_dim, CONFIG["hidden"], CONFIG)
		model = GraphHead(encoder, CONFIG["hidden"], out_dim).to(DEVICE)

		# Train
		res_train = train_graph_level(model, graphs, epochs=cfg["epochs"], precision=precision, batch=batch)
		val_acc = graph_val_acc(ds if dataset_name != "ds002748" else graphs, model, batch, precision)
		# Inference + power
		loader_eval = PyGDataLoader(graphs, batch_size=batch, shuffle=False)
		batch0 = next(iter(loader_eval)).to(DEVICE)
		if precision=="fp16" and DEVICE.type=="cuda": model = model.half()
		model.eval(); fwd = graph_forward_fn(model, batch0, precision=precision)

		local_reps, local_warmup, local_min = reps, warmup, float(CONFIG.get('min_infer_time_s', 3.0))
		try: n_nodes = int(batch0.num_nodes)
		except Exception: n_nodes = None
		if n_nodes is not None and n_nodes < int(CONFIG.get('tiny_graph_threshold', 8000)):
			local_reps = max(local_reps, 200); local_warmup = max(local_warmup, 20); local_min = max(local_min, 3.0)
		reset_gpu_peak()
		with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=power_trace_path) as mon2:
			inf_metrics = run_inference_timed(fwd, reps=local_reps, warmup=local_warmup, min_time_s=local_min)
		infer_gpu_mem_peak_mb = gpu_peak_mb()
		infer_gpu_mem_reserved_peak_mb = gpu_reserved_peak_mb()
		mean_power_w = mon2.mean_power()
		#energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0))
		per_batch = batch if isinstance(batch, int) and batch > 0 else 1
		energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)) / per_batch

		summary = {
		"timestamp": datetime.utcnow().isoformat(),
		"platform": "jetson",
		"device_name": platform.platform(),
		"dataset": dataset_name, "model": model_name, "precision": precision,
		"batch": batch if isinstance(batch, (int,str)) else str(batch),
		"seed": cfg.get("seed", 42),
		"phase": "train+infer_node" if is_node else "train+infer_graph",
		# === NEW: inject TRAIN metrics ===
		**res_train,
		# === validation / test (whatever the runner computed) ===
		"val_metric_final": float(val_acc) if 'val_acc' in locals() and val_acc is not None else None,
		"test_acc": float(test_acc) if 'test_acc' in locals() and test_acc is not None else "",
		# === inference (as before) ===
		**inf_metrics,                  # has latency_p50_ms, latency_p95_ms, throughput_req_s
		"mean_power_w": mean_power_w,
		"energy_per_inf_j": energy_per_inf_j,
	}
		summary.update({
		"infer_gpu_mem_peak_mb": infer_gpu_mem_peak_mb,
		"infer_gpu_mem_reserved_peak_mb": infer_gpu_mem_reserved_peak_mb,
		})
	#print(json.dumps(summary, indent=2))


	with open(summary_path, "w") as f: json.dump(summary, f, indent=2)
	print(json.dumps(summary, indent=2))
	# Append compact CSV
	header = ["timestamp","platform","device_name","dataset","model","precision","batch","seed","phase","train_total_time_s","train_mean_power_w",
"train_energy_total_j","train_energy_per_example_j","train_throughput_examples_s","train_avg_step_ms","train_step_p50_ms","train_step_p95_ms","train_seen_examples",
	"val_metric_final","latency_p50_ms","latency_p95_ms","throughput_req_s","mean_power_w","energy_per_inf_j","test_acc"]
	for k in header: summary.setdefault(k, "")
	# write CSV
	
	csv_path = "outputs/runs_jetson.csv"
	exists = Path(csv_path).exists()
	with open(csv_path, "a", newline="") as f:
		w = csv.DictWriter(f, fieldnames=header)
		if not exists: w.writeheader()
		w.writerow({k: summary.get(k,"") for k in header})
	return summary

# === Full-graph Laplacian eigenpairs precompute for large node datasets ===
# Works once per dataset; caches to eigs_cache/<name>_full_k<k>.pt
def precompute_fullgraph_eigs(dataset_name: str, *, k: int = 16, tol: float = 1e-3, maxiter: int | None = None):
	"""
	dataset_name ∈ {"Reddit","ogbn-arxiv"} for your notebook.
	- Builds symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2} (in CSR).
	- Uses scipy.sparse.linalg.eigsh to compute k smallest eigenpairs.
	- Saves tensors to eigs_cache/<dataset>_full_k<k>.pt (CPU float32).
	"""
	
	os.makedirs("eigs_cache", exist_ok=True)

	# 1) Load the big graph as a single PyG Data with edge_index
	if dataset_name.lower() == "reddit":
		
		data = Reddit(root="data/Reddit")[0]
	elif dataset_name.lower() in {"ogbn-arxiv", "arxiv"}:
		
		ds = PygNodePropPredDataset(name="ogbn-arxiv", root="data/OGB")
		data = ds[0]
	else:
		raise ValueError(f"Unsupported dataset for full-graph eigs: {dataset_name}")

	n = int(data.num_nodes)
	ei = data.edge_index.cpu().numpy()
	u, v = ei[0], ei[1]

	# 2) Build undirected adjacency (coalesce)
	

	# Symmetrize: add transpose & remove duplicates
	uu = np.concatenate([u, v])
	vv = np.concatenate([v, u])
	vals = np.ones_like(uu, dtype=np.float32)
	A = csr_matrix((vals, (uu, vv)), shape=(n, n))
	A.sum_duplicates()

	# 3) Normalized Laplacian in CSR
	deg = np.asarray(A.sum(axis=1)).reshape(-1)
	with np.errstate(divide='ignore'):
		dinvs = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
	Dinv = csr_matrix((dinvs, (np.arange(n), np.arange(n))), shape=(n, n))
	# L = I - D^{-1/2} A D^{-1/2}
	N = csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.arange(n))), shape=(n, n))
	L = N - (Dinv @ A @ Dinv)

	# 4) Sparse eigens (k smallest)
	
	k_eff = max(1, min(k, n-1))
	e, V = eigsh(L, k=k_eff, which="SM", tol=tol, maxiter=maxiter)

	# 5) Sort ascending, cast, save
	idx = np.argsort(e)
	e = e[idx].astype(np.float32)
	V = V[:, idx].astype(np.float32)

	out = {
		"eigval": torch.from_numpy(e),          # [k]
		"eigvec": torch.from_numpy(V),          # [n, k] (rows align with node IDs)
		"num_nodes": n,
		"k": k_eff,
	}
	save_path = os.path.join("eigs_cache", f"{dataset_name}_full_k{k_eff}.pt")
	torch.save(out, save_path)
	print(f"[OK] saved full-graph eigs → {save_path} (n={n}, k={k_eff})")
	return save_path

# Example:
#precompute_fullgraph_eigs("Reddit", k=16, tol=1e-2)
#precompute_fullgraph_eigs("ogbn-arxiv", k=16, tol=1e-3)

def load_fullgraph_eigs(dataset_name: str, k: int = 16):
	
	path = os.path.join("eigs_cache", f"{dataset_name}_full_k{k}.pt")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Missing full-graph eigs at {path}. Run precompute_fullgraph_eigs(...).")
	blob = torch.load(path, map_location="cpu")
	# tolerate k mismatch by slicing
	E = blob["eigval"]; V = blob["eigvec"]
	k_eff = min(k, E.numel(), V.shape[1])
	return E[:k_eff], V[:, :k_eff]

# ==== Diagnostics ====
print("=== Jetson checks ===")
#print("tegrastats present?:", Path(shutil.which("tegrastats") or "/usr/bin/tegrastats").exists())
print("DEVICE:", DEVICE, "| Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("=== EIG STATUS (k=16) ===")
eig_status_report(k=EIG_K)
	
# ==== COLLAB-only loader (safe features + labels) ====


def _collab_make_feats(g: Data):
	"""
	COLLAB ships with x=None. We fabricate lightweight node features:
	  - degree per node (float) => shape [N,1]
	and ensure labels are present & long dtype.
	"""
	# Build node features if missing/empty
	if getattr(g, "x", None) is None or (isinstance(g.x, torch.Tensor) and g.x.numel()==0):
		n = int(g.num_nodes) if g.num_nodes is not None else (
			int(g.edge_index.max().item()+1) if g.edge_index.numel()>0 else 1
		)
		deg = degree(g.edge_index[0], num_nodes=n, dtype=torch.float32).view(-1, 1)  # [N,1]
		g.x = deg
	# Force float features
	if not torch.is_floating_point(g.x):
		g.x = g.x.float()

	# Ensure graph-level label exists and is long
	if getattr(g, "y", None) is None or g.y.numel() == 0:
		# create a dummy label (won't affect timing; accuracy will be meaningless)
		g.y = torch.zeros(1, dtype=torch.long)
	elif g.y.dtype != torch.long:
		g.y = g.y.long()

	return g

def load_collab_graphs(data_root):
	ds = TUDS(root=str(data_root/"TU"), name="COLLAB")
	graphs = [ _collab_make_feats(ds[i]) for i in range(len(ds)) ]
	# quick sanity
	g0 = graphs[0]
	print(f"[COLLAB] loaded {len(graphs)} graphs; x[0].shape={tuple(g0.x.shape)}, y[0].shape={tuple(g0.y.shape)}")
	return graphs

# ==== Infer in/out dims from list[Data] + optional eig attach (with padding) ====

def infer_in_out_dims_from_graphs(graphs):
	# in_dim from x
	g0 = graphs[0]
	if getattr(g0, "x", None) is None:
		raise RuntimeError("COLLAB: graph has no x after sanitation; cannot infer in_dim.")
	in_dim = int(g0.x.size(-1))

	# out_dim from labels (max label + 1), fallback=2
	ys = []
	for g in graphs:
		y = getattr(g, "y", None)
		if y is not None and y.numel() > 0:
			ys.append(y.view(-1).long())
	out_dim = int(torch.cat(ys, dim=0).max().item()) + 1 if len(ys) else 2
	return in_dim, out_dim

# If you already have eig helpers, this will reuse them. Otherwise SAN gets skipped.
def maybe_attach_collab_eigs(graphs, k=16, strict=False):
	"""
	If your notebook already defined:
	  - eig_cache_dir(name,k)
	  - attach_graph_eigs_list_from_cache(dataset_name,k,strict)
	this will pad/truncate to k and attach eigvec/eigval.
	If not available, it will warn and return graphs unchanged.
	"""
	try:
		_ = eig_cache_dir("COLLAB", k)  # probe
		graphs = attach_graph_eigs_list_from_cache(graphs, "COLLAB", k=k, strict=strict)
		print(f"[COLLAB] eigs attached with k={k}")
	except NameError:
		print("[COLLAB] eig helpers not found; skipping eig attachment (SAN will error).")
	return graphs

# COLLAB helper 
def _split_idx_simple(num_graphs, train_ratio=0.8, val_ratio=0.1):
	n = int(num_graphs)
	n_train = int(n * train_ratio)
	n_val   = int(n * val_ratio)
	perm = torch.randperm(n)
	train = perm[:n_train]
	valid = perm[n_train:n_train+n_val]
	test  = perm[n_train+n_val:]
	return {"train": train, "valid": valid, "test": test}

# ==== COLLAB-only runner ====


def run_experiment_collab(model_name: str, cfg: dict, run_name_suffix: str = ""):
	"""
	model_name: one of ["gcn","gat","graphsage","gin","san_full","graphgps_full"]
	cfg needs keys you already use elsewhere:
	  - "precision" in {"fp32","fp16"}
	  - "hidden", "epochs", "batch", "reps", "warmup", "min_infer_time_s", "seed"
	Requires existing utilities in your notebook:
	  - DEVICE, amp_context, build_model, GraphHead
	  - train_graph_level(model, dataset, epochs, precision, batch)
	  - run_inference_timed(fn, reps, warmup, min_time_s)
	  - TegraStatsSampler(interval_ms=..., rail=..., save_trace_path=...)
	"""
	dataset_name="COLLAB"
	torch.manual_seed(int(cfg.get("seed", 42)))
	precision = cfg["precision"].lower()
	batch = int(cfg.get("batch", 64))

	# 1) Load + sanitize COLLAB
	graphs = load_collab_graphs(DATA_ROOT)

	# 2) Eigs if SAN_Full
	if model_name.lower() == "san_full":
		graphs = maybe_attach_collab_eigs(graphs, k=int(cfg.get("eig_k", 16)), strict=False)

	# 3) Dims
	in_dim, out_dim = infer_in_out_dims_from_graphs(graphs)

	# 4) Build model: encoder + graph head (reuse your convention)
	#encoder = build_model(model_name, in_dim, cfg.get("hidden", 128), cfg)
	#model = GraphHead(encoder, cfg.get("hidden", 128), out_dim).to(DEVICE)
	#if precision=="fp16" and DEVICE.type=="cuda":
	#    model = model.half()
	# 4m) Build model: encoder + graph head (reuse your convention)
	encoder = build_model(model_name, in_dim, cfg.get("hidden", 128), cfg)
	model = GraphHead(encoder, cfg.get("hidden", 128), out_dim).to(DEVICE)
	precision = cfg["precision"].lower()
	# IMPORTANT: keep FP32 weights during training (AMP will cast activations/ops)
	# DO NOT call model.half() here for training!
	spl = _split_idx_simple(len(graphs))
	@torch.no_grad()
	def _eval_graph_acc(loader):
		model.eval()
		total, correct = 0, 0
		for b in loader:
			b = b.to(DEVICE)
			with amp_context(cfg["precision"]):
				out = model(b.x, b.edge_index, b.batch,
						eigvec=getattr(b,'eigvec',None),
						eigval=getattr(b,'eigval',None))
			pred = out.argmax(dim=-1)
			y = b.y.view(-1).long()
			total += y.size(0)
			correct += (pred == y).sum().item()
		return correct/total if total>0 else None
	# 5) Train
	#reset_gpu_peak()
	res_train = train_graph_level(model, graphs, epochs=int(cfg.get("epochs", 30)),
								  precision=precision, batch=batch)
	val_loader  = PyGDataLoader([graphs[i] for i in spl["valid"]], batch_size=cfg.get("batch",64), shuffle=False)
	val_acc = _eval_graph_acc(val_loader)
	# 6) Timed inference (one pass over a batch from loader)
	
	
	# After training finishes:
	loader_eval = PyGDataLoader(graphs, batch_size=batch, shuffle=False)
	batch0 = next(iter(loader_eval)).to(DEVICE)
	def _fwd():
		with torch.no_grad():
			with amp_context(precision):
				_ = model(batch0.x, batch0.edge_index, batch0.batch,
						  eigvec=getattr(batch0, "eigvec", None),
						  eigval=getattr(batch0, "eigval", None))

	run_name = f"jetson_COLLAB_{model_name}_{precision}_b{batch}{run_name_suffix}"
	os.makedirs("traces", exist_ok=True)
	power_trace_path = os.path.join("traces", run_name + "__pwr.jsonl")
	# rail="orin_sum" expects your earlier tegrastats rail-mapping fix
	reset_gpu_peak()
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=power_trace_path) as mon2:
		inf_metrics = run_inference_timed(_fwd,
										  reps=int(cfg.get("reps", 60)),
										  warmup=int(cfg.get("warmup", 10)),
										  min_time_s=float(cfg.get("min_infer_time_s", 2.0)))
	infer_gpu_mem_peak_mb = gpu_peak_mb()
	infer_gpu_mem_reserved_peak_mb = gpu_reserved_peak_mb()
	mean_power_w = mon2.mean_power()
	#energy_per_inf_j = mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)
	per_batch = batch if isinstance(batch, int) and batch > 0 else 1
	energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)) / per_batch


	# 7) Emit summary + CSV append (same columns as your other runs)
	summary = {
		"timestamp": datetime.utcnow().isoformat(),
		"platform": "jetson",
		"device_name": platform.platform(),
		"dataset": dataset_name, "model": model_name, "precision": precision,
		"batch": batch if isinstance(batch, (int,str)) else str(batch),
		"seed": cfg.get("seed", 42),
		"phase": "train+infer_node" if is_node else "train+infer_graph",
		# === NEW: inject TRAIN metrics ===
		**res_train,
		# === validation / test (whatever the runner computed) ===
		"val_metric_final": float(val_acc) if 'val_acc' in locals() and val_acc is not None else None,
		"test_acc": float(test_acc) if 'test_acc' in locals() and test_acc is not None else "",
		# === inference (as before) ===
		**inf_metrics,                  # has latency_p50_ms, latency_p95_ms, throughput_req_s
		"mean_power_w": mean_power_w,
		"energy_per_inf_j": energy_per_inf_j,
	}
	summary.update({
	"infer_gpu_mem_peak_mb": infer_gpu_mem_peak_mb,
	"infer_gpu_mem_reserved_peak_mb": infer_gpu_mem_reserved_peak_mb,
	})
	print(json.dumps(summary, indent=2))

	header = ["timestamp","platform","device_name","dataset","model","precision","batch","seed","phase","train_total_time_s","train_mean_power_w",
"train_energy_total_j","train_energy_per_example_j","train_throughput_examples_s","train_avg_step_ms","train_step_p50_ms","train_step_p95_ms","train_seen_examples",
	"val_metric_final","latency_p50_ms","latency_p95_ms","throughput_req_s","mean_power_w","energy_per_inf_j","test_acc","infer_gpu_mem_peak_mb","infer_gpu_mem_reserved_peak_mb"]
	for k in header: summary.setdefault(k, "")
	os.makedirs("outputs", exist_ok=True)
	csv_path = "outputs/runs_jetson.csv"
	exists = os.path.exists(csv_path)
	
	with open(csv_path, "a", newline="") as f:
		w = csv.DictWriter(f, fieldnames=header)
		if not exists: w.writeheader()
		w.writerow({k: summary.get(k,"") for k in header})

	return summary

# Arxiv and Reddit
# ==== CSR adjacency builder ====


def build_csr(edge_index: torch.Tensor, num_nodes: int):
	"""
	edge_index: [2, E] (COO, zero-based, on CPU or GPU)
	Returns: CSR {indptr, indices} as CPU numpy arrays (compact, fast).
	"""
	ei = edge_index.detach().cpu().numpy()
	src, dst = ei[0], ei[1]
	order = np.argsort(src, kind="mergesort")
	src_sorted = src[order]; dst_sorted = dst[order]
	counts = np.bincount(src_sorted, minlength=num_nodes)
	indptr = np.zeros(num_nodes + 1, dtype=np.int64)
	indptr[1:] = np.cumsum(counts, dtype=np.int64)
	indices = dst_sorted.astype(np.int64, copy=False)
	return SimpleNamespace(indptr=indptr, indices=indices, num_nodes=int(num_nodes))

# ====  k-hop sampler and induced subgraph constructor ====


# ==== KHOP UTILITIES ====


def khop_expand(seeds: np.ndarray, csr, num_hops: int, max_nodes: int):
	seeds = np.asarray(seeds, dtype=np.int64)
	visited = set(seeds.tolist())
	frontier = seeds
	for _ in range(num_hops):
		nxt = []
		for u in frontier:
			start, end = csr.indptr[u], csr.indptr[u+1]
			nbrs = csr.indices[start:end]
			for v in nbrs:
				if v not in visited:
					visited.add(v); nxt.append(v)
		if not nxt: break
		frontier = np.asarray(nxt, dtype=np.int64)
		if len(visited) >= max_nodes: break
	nodes = np.fromiter(visited, dtype=np.int64)
	if nodes.size > max_nodes:
		sel = np.random.choice(nodes.size, size=max_nodes, replace=False)
		nodes = nodes[sel]
	return np.sort(nodes)

def induced_subgraph_from_nodesOLD(data, nodes_np: np.ndarray):
	nodes_np = np.asarray(nodes_np, dtype=np.int64)
	Np = int(nodes_np.size)
	if Np == 0:
		nodes_np = np.array([0], dtype=np.int64); Np = 1
	max_id = int(nodes_np.max())
	map_arr = -np.ones(max_id + 1, dtype=np.int64)
	map_arr[nodes_np] = np.arange(Np, dtype=np.int64)

	ei = data.edge_index.detach().cpu().numpy()
	u_all, v_all = ei[0], ei[1]

	mu = (u_all <= max_id)
	mv = (v_all <= max_id)

	valid_u = np.zeros_like(u_all, dtype=bool)
	valid_v = np.zeros_like(v_all, dtype=bool)
	if mu.any(): valid_u[mu] = (map_arr[u_all[mu]] != -1)
	if mv.any(): valid_v[mv] = (map_arr[v_all[mv]] != -1)

	mask = valid_u & valid_v
	if not mask.any():
		edge_index_sub = torch.empty((2,0), dtype=torch.long)
	else:
		u_sub = map_arr[u_all[mask]]
		v_sub = map_arr[v_all[mask]]
		edge_index_sub = torch.from_numpy(np.vstack([u_sub, v_sub]).astype(np.int64))

	x_sub = data.x[nodes_np]
	y_sub = data.y[nodes_np]
	if y_sub.dim() > 1: y_sub = y_sub.view(-1)

	return Data(x=x_sub, edge_index=edge_index_sub, y=y_sub, n_id=torch.from_numpy(nodes_np))

def induced_subgraph_from_nodes(data, nodes_np: np.ndarray):
	nodes_np = np.asarray(nodes_np, dtype=np.int64)
	Np = int(nodes_np.size)
	if Np == 0:
		nodes_np = np.array([0], dtype=np.int64); Np = 1
	max_id = int(nodes_np.max())
	map_arr = -np.ones(max_id + 1, dtype=np.int64)
	map_arr[nodes_np] = np.arange(Np, dtype=np.int64)

	ei = data.edge_index.detach().cpu().numpy()
	u_all, v_all = ei[0], ei[1]

	mu = (u_all <= max_id)
	mv = (v_all <= max_id)

	valid_u = np.zeros_like(u_all, dtype=bool)
	valid_v = np.zeros_like(v_all, dtype=bool)
	if mu.any(): valid_u[mu] = (map_arr[u_all[mu]] != -1)
	if mv.any(): valid_v[mv] = (map_arr[v_all[mv]] != -1)

	mask = valid_u & valid_v
	if not mask.any():
		edge_index_sub = torch.empty((2,0), dtype=torch.long)
	else:
		u_sub = map_arr[u_all[mask]]
		v_sub = map_arr[v_all[mask]]
		edge_index_sub = torch.from_numpy(np.vstack([u_sub, v_sub]).astype(np.int64))

	x_sub = data.x[nodes_np]
	y_sub = data.y[nodes_np]
	if y_sub.dim() > 1: y_sub = y_sub.view(-1)

	# IMPORTANT: keep original node ids for eig slicing
	sub = Data(x=x_sub, edge_index=edge_index_sub, y=y_sub)
	sub.n_id = torch.from_numpy(nodes_np)       # used elsewhere in your code
	sub.orig_n_id = sub.n_id.clone()            # explicit alias for clarity
	return sub

def slice_full_eigs_to_subgraph(sub, full_eigvec, full_eigval):
	"""
	Copies rows of the precomputed full-graph eigenvectors to this subgraph,
	using its original node ids (sub.orig_n_id or sub.n_id).
	"""
	idx = getattr(sub, "orig_n_id", None)
	if idx is None:
		idx = getattr(sub, "n_id", None)
	if idx is None:
		raise RuntimeError("subgraph missing orig_n_id/n_id for eig slicing")

	# Ensure on same device/dtype as features
	idx = idx.to(full_eigvec.device)
	sub.eigvec = full_eigvec.index_select(0, idx)               # [n_sub, k]
	sub.eigval = full_eigval                                    # [k]
	return sub


# ==== KHOP TRAINING LOOP ====
def run_khop_training_loopOLD(*, model, optimizer, scaler, ce, autocast_ctx,
						   epochs, k_batches, sample_batch_for, DEVICE,
						   dataset_name="khop", traces_dir="traces"):
	step_times_ms, seen_examples = [], 0
	os.makedirs(traces_dir, exist_ok=True)
	train_trace_path = os.path.join(traces_dir, f"train_{dataset_name}_khop__pwr.jsonl")

	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=train_trace_path) as mon_train, \
		 train_energy_timer(mon_train):
		for _ in range(int(epochs)):
			model.train()
			for _it in range(int(k_batches)):
				sub = sample_batch_for().to(DEVICE)
				t0 = time.time()
				optimizer.zero_grad(set_to_none=True)
				with autocast_ctx:
					out = model(sub.x, sub.edge_index)
					loss = ce(out, sub.y.long())
				if scaler.is_enabled():
					scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
				else:
					loss.backward(); optimizer.step()
				step_times_ms.append((time.time()-t0)*1000.0)
				seen_examples += int(sub.y.numel())
	return _finalize_train_metrics(seen_examples, step_times_ms, mon_train)

def run_khop_training_loop(*, model, optimizer, scaler, ce, autocast_ctx,
						   epochs, k_batches, sample_batch_for, DEVICE,
						   dataset_name="khop", traces_dir="traces",
						   spec_cb=None):
	"""
	spec_cb(subgraph) -> subgraph
	  Optional callback to attach spectral features (eigvec/eigval) to each sampled subgraph.
	"""
	step_times_ms, seen_examples = [], 0
	os.makedirs(traces_dir, exist_ok=True)
	train_trace_path = os.path.join(traces_dir, f"train_{dataset_name}_khop__pwr.jsonl")
	reset_gpu_peak()
	
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=train_trace_path) as mon_train, \
		 train_energy_timer(mon_train):
		for _ in range(int(epochs)):
			model.train()
			for _it in range(int(k_batches)):
				sub = sample_batch_for().to(DEVICE)
				if spec_cb is not None:
					sub = spec_cb(sub)  # attach eigvec/eigval if SAN/GraphGPS

				t0 = time.time()
				optimizer.zero_grad(set_to_none=True)
				with autocast_ctx:
					out = model(
						sub.x, sub.edge_index,
						eigvec=getattr(sub, "eigvec", None),
						eigval=getattr(sub, "eigval", None)
					)
					loss = ce(out, sub.y.long())
				if scaler.is_enabled():
					scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
				else:
					loss.backward(); optimizer.step()
				step_times_ms.append((time.time()-t0)*1000.0)
				seen_examples += int(sub.y.numel())
	train_gpu_mem_peak_mb = gpu_peak_mb()
	train_gpu_mem_reserved_peak_mb = gpu_reserved_peak_mb()
	res = _finalize_train_metrics(seen_examples, step_times_ms, mon_train)
	res["train_gpu_mem_peak_mb"] = gpu_peak_mb()
	res["train_gpu_mem_reserved_peak_mb"] = gpu_reserved_peak_mb()
	return res
	#return _finalize_train_metrics(seen_examples, step_times_ms, mon_train)

# ==== k-hop mini-batch runner for large node datasets (Reddit, ogbn-arxiv) ====


def run_experiment_khop(dataset_name, model_name, cfg, run_name_suffix=""):
	"""
	K-hop runner for Reddit / ogbn-arxiv on Jetson (no torch-sparse/pyg-lib).
	Requires:
	  - khop_expand / induced_subgraph_from_nodes (Cell D)
	  - run_khop_training_loop (Cell E)
	  - build_summary / append_summary_to_csv (Cell F)
	  - TegraStatsSampler, DEVICE, build_model, make_optimizer
	"""
	

	assert dataset_name in {"Reddit", "ogbn-arxiv"}

	# ---------- cfg ----------
	precision = cfg.get("precision", "fp32")
	epochs    = int(cfg.get("epochs", 20))
	hidden    = int(cfg.get("hidden", 128))
	reps      = int(cfg.get("reps", 60))
	warmup    = int(cfg.get("warmup", 10))
	min_time  = float(cfg.get("min_infer_time_s", 2.0))
	khops     = int(cfg.get("khop_hops", 2))
	k_nodes   = int(cfg.get("khop_nodes", 20000))
	k_batches = int(cfg.get("khop_batches_per_epoch", 4))
	num_eval  = int(cfg.get("num_eval_samples", 8))
	seed      = int(cfg.get("seed", 42))

	torch.manual_seed(seed); np.random.seed(seed)
	needs_spec = model_name.lower() in {"san_full", "graphgps_full"}
	if needs_spec:
		full_eigval, full_eigvec = load_fullgraph_eigs(dataset_name, k=int(cfg.get("eig_k", 16)))
		full_eigval = full_eigval.to(DEVICE)
		full_eigvec = full_eigvec.to(DEVICE)
		spec_cb = lambda sub: slice_full_eigs_to_subgraph(sub, full_eigvec, full_eigval)
	else:
		spec_cb = None
	# ---------- data ----------
	if dataset_name == "Reddit":
		#from torch_geometric.datasets import Reddit
		ds = Reddit(root="data/Reddit")
		data = ds[0].to(DEVICE)
	else:  # ogbn-arxiv
		#from ogb.nodeproppred import PygNodePropPredDataset
		ds = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
		data = ds[0]
		data.y = data.y.view(-1)
		split = ds.get_idx_split()
		data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
		data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
		data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
		data.train_mask[split["train"]] = True
		data.val_mask[split["valid"]]   = True
		data.test_mask[split["test"]]   = True
		data = data.to(DEVICE)

	# ---------- model ----------
	in_dim  = int(data.num_features)
	out_dim = int(data.y.max().item()+1)
	base = build_model(model_name, in_dim, hidden, cfg)
	model = base.to(DEVICE)

	# ---------- optim / amp ----------
	opt = make_optimizer(model)
	ce  = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler("cuda", enabled=(precision.lower()=="fp16" and DEVICE.type=="cuda"))
	autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.float16) if (precision.lower()=="fp16" and DEVICE.type=="cuda") else nullcontext())

	# ---------- CPU CSR for expansion ----------
	data_cpu = data.detach().cpu()
	row, col  = data_cpu.edge_index.numpy()
	N = int(data_cpu.num_nodes)
	csr = csr_matrix((np.ones_like(row, dtype=np.int8), (row, col)), shape=(N, N))

	# ---------- masks ----------
	train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
	val_idx   = data.val_mask.nonzero(as_tuple=False).view(-1) if getattr(data,"val_mask",None) is not None else None

	# ---------- sampler closures ----------
	def sample_batch_for(idx_t):
		# sample seeds safely, then expand
		n_pick = max(1, min(1024, idx_t.numel()))
		pick = torch.randint(0, idx_t.numel(), (n_pick,))
		seeds = idx_t[pick].detach().cpu().numpy().astype(np.int64)
		seeds = seeds[(seeds >= 0) & (seeds < data_cpu.num_nodes)]
		if seeds.size == 0: seeds = np.array([0], dtype=np.int64)
		nodes_np = khop_expand(seeds, csr, num_hops=khops, max_nodes=k_nodes)
		return induced_subgraph_from_nodes(data_cpu, nodes_np)

	# ---------- TRAIN (via Cell E helper) ----------
	#res_trainOLD = run_khop_training_loop(
		#model=model, optimizer=opt, scaler=scaler, ce=ce, autocast_ctx=autocast_ctx,
		#epochs=epochs, k_batches=k_batches,
		#sample_batch_for=lambda: sample_batch_for(train_idx),
		#DEVICE=DEVICE, dataset_name=dataset_name, traces_dir="traces"
	#)
	#spazio per reset valore memoria gpu
	res_train = run_khop_training_loop(
		model=model, optimizer=opt, scaler=scaler, ce=ce, autocast_ctx=autocast_ctx,
		epochs=epochs, k_batches=k_batches,
		sample_batch_for=lambda: sample_batch_for(train_idx),
		DEVICE=DEVICE, dataset_name=dataset_name, traces_dir="traces",
		spec_cb=spec_cb  # <--- add this
	)

	# ---------- EVAL on micro-batches ----------
	#@torch.no_grad()
	#def eval_acc_on(idx, samples=num_eval):
	#    if idx is None or idx.numel()==0: return None
	#    correct = 0; total = 0
	#    for _ in range(samples):
	#        sub = sample_batch_for(idx).to(DEVICE)
	#        pred = model(sub.x, sub.edge_index).argmax(dim=-1)
	#        y = sub.y.long()
	#        correct += int((pred == y).sum().item()); total += int(y.numel())
	#    return (correct/total) if total>0 else None
	
	#val_acc = eval_acc_on(val_idx)

	@torch.no_grad()
	def eval_acc_on(idx, samples=num_eval):
		if idx is None or idx.numel()==0: return None
		correct = 0; total = 0
		for _ in range(samples):
			sub = sample_batch_for(idx).to(DEVICE)
			if spec_cb is not None:
				sub = spec_cb(sub)
			pred = model(
				sub.x, sub.edge_index,
				eigvec=getattr(sub, "eigvec", None),
				eigval=getattr(sub, "eigval", None)
			).argmax(dim=-1)
			y = sub.y.long()
			correct += int((pred == y).sum().item()); total += int(y.numel())
		return (correct/total) if total>0 else None

	val_acc = eval_acc_on(val_idx)
	

	# ---------- INFERENCE timing + power ----------
	def run_inference_timed(fn, reps=10, warmup=3, min_time_s=1.5):
		for _ in range(warmup):
			fn(); 
			if DEVICE.type=="cuda": torch.cuda.synchronize()
		ts = []; t0 = time.time()
		while len(ts) < reps or (time.time()-t0) < min_time_s:
			t1 = time.time(); fn()
			if DEVICE.type=="cuda": torch.cuda.synchronize()
			ts.append((time.time()-t1)*1000.0)
		ts.sort()
		p50 = ts[len(ts)//2]
		p95 = ts[int(len(ts)*0.95)-1 if int(len(ts)*0.95)-1>=0 else 0]
		thr = 1000.0/ p50 if p50>0 else None
		return {"latency_p50_ms": p50, "latency_p95_ms": p95, "throughput_req_s": thr}

	# representative eval subgraph
	#batch0 = sample_batch_for(val_idx if val_idx is not None and val_idx.numel()>0 else train_idx).to(DEVICE)
	#def fwd():
	#    with torch.no_grad():
	#        _ = model(batch0.x, batch0.edge_index)
	batch0 = sample_batch_for(val_idx if val_idx is not None and val_idx.numel()>0 else train_idx).to(DEVICE)
	if spec_cb is not None:
		batch0 = spec_cb(batch0)
	def fwd():
		with torch.no_grad():
			_ = model(
			batch0.x, batch0.edge_index,
			eigvec=getattr(batch0, "eigvec", None),
			eigval=getattr(batch0, "eigval", None)
			)
	reset_gpu_peak()
	with TegraStatsSampler(interval_ms=200, rail="orin_sum") as mon_inf:
		inf_metrics = run_inference_timed(fwd, reps=reps, warmup=warmup, min_time_s=min_time)
	infer_gpu_mem_peak_mb = gpu_peak_mb()
	infer_gpu_mem_reserved_peak_mb = gpu_reserved_peak_mb()
	mean_power_w = mon_inf.mean_power()
	energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)) if mean_power_w is not None else None

	# ---------- SUMMARY + CSV ----------
	summary = build_summary(
		dataset_name=dataset_name, model_name=model_name, precision=precision,
		batch=f"khop_nodes={k_nodes}", seed=seed,
		phase="train+infer_node",
		res_train=res_train,
		inf_metrics=inf_metrics,
		mean_power_w=mean_power_w, energy_per_inf_j=energy_per_inf_j,
		val_acc=val_acc
	)
	summary.update({
	"infer_gpu_mem_peak_mb": infer_gpu_mem_peak_mb,
	"infer_gpu_mem_reserved_peak_mb": infer_gpu_mem_reserved_peak_mb,
	})
	print(json.dumps(summary, indent=2))
	append_summary_to_csv(summary, csv_path="outputs/runs_all.csv")
	return summary

# ==== SUMMARY + CSV HELPERS ====

CSV_HEADER = [
	"timestamp","platform","device_name","dataset","model","precision","batch","seed","phase",
	# training
	"train_total_time_s","train_mean_power_w","train_energy_total_j","train_energy_per_example_j",
	"train_throughput_examples_s","train_avg_step_ms","train_step_p50_ms","train_step_p95_ms","train_seen_examples",
	"train_gpu_mem_peak_mb","train_gpu_mem_reserved_peak_mb",   # ← NEW
	# validation
	"val_metric_final",
	# inference
	"latency_p50_ms","latency_p95_ms","throughput_req_s","mean_power_w","energy_per_inf_j",
	"infer_gpu_mem_peak_mb","infer_gpu_mem_reserved_peak_mb",   # ← NEW
	# optional
	"test_acc",
]


def build_summary(*, dataset_name, model_name, precision, batch, seed, phase,
				  res_train, inf_metrics, mean_power_w, energy_per_inf_j,
				  val_acc=None, test_acc=None):
	base = dict(
		timestamp = datetime.utcnow().isoformat(),
		platform  = "jetson",
		device_name = platform.platform(),
		dataset = dataset_name,
		model   = model_name,
		precision = precision,
		batch   = batch if isinstance(batch, (int,str)) else str(batch),
		seed    = int(seed),
		phase   = phase,
	)
	out = {**base, **(res_train or {})}
	out["val_metric_final"] = float(val_acc) if val_acc is not None else None
	out.update(inf_metrics or {})
	out["mean_power_w"] = mean_power_w
	out["energy_per_inf_j"] = energy_per_inf_j
	out["test_acc"] = float(test_acc) if test_acc is not None else ""
	return out

def append_summary_to_csv(summary, csv_path="outputs/runs_all.csv"):
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	exists = os.path.exists(csv_path)
	with open(csv_path, "a", newline="") as f:
		w = csv.DictWriter(f, fieldnames=CSV_HEADER)
		if not exists: w.writeheader()
		row = {k: summary.get(k, "") for k in CSV_HEADER}
		w.writerow(row); f.flush()


# ==== ZINC-only loader====


def _zinc_fix(g: Data) -> Data:
	# Features must be float
	if getattr(g, "x", None) is None:
		raise RuntimeError("ZINC graph has no x")
	if not torch.is_floating_point(g.x):
		g.x = g.x.float()
	# Targets must be float [B, 1]
	if getattr(g, "y", None) is None:
		raise RuntimeError("ZINC graph has no y")
	if not torch.is_floating_point(g.y):
		g.y = g.y.float()
	if g.y.dim() == 1:
		g.y = g.y.unsqueeze(-1)  # [1] -> [1,1] per graph
	return g

def load_zinc_graphs(data_root):
	ds = ZINC_DS(root=str(data_root/"ZINC"))
	# Turn into python list of sanitized graphs
	graphs = [ _zinc_fix(ds[i]) for i in range(len(ds)) ]
	g0 = graphs[0]
	print(f"[ZINC] loaded {len(graphs)} graphs; x[0].shape={tuple(g0.x.shape)}, y[0].shape={tuple(g0.y.shape)} (regression)")
	return graphs
	

# ==== Task registry & helpers ====
TASK_BY_DATASET = {
	# node-level datasets you use remain classification by default
	# graph-level special case:
	"ZINC": "regression",
	# everything else defaults to "classification"
}

def get_task_for_dataset(name: str) -> str:
	return TASK_BY_DATASET.get(name, "classification")

def make_loss_for_task(task: str):
	if task == "regression":
		# SmoothL1 is stable in FP16; you can switch to nn.L1Loss() for pure MAE
		return torch.nn.SmoothL1Loss()
	return torch.nn.CrossEntropyLoss()

@torch.no_grad()
def compute_metric_for_task(task: str, logits: torch.Tensor, y: torch.Tensor):
	"""
	Returns a scalar metric:
	  - classification: accuracy in [0,1]
	  - regression: MAE (lower is better)
	`logits` is [B, C] for classification or [B, 1] for regression.
	"""
	if task == "regression":
		# Ensure shapes: [B,1] vs [B,1]
		if y.dim()==1:
			y = y.unsqueeze(-1)
		if logits.dim()==1:
			logits = logits.unsqueeze(-1)
		return torch.mean(torch.abs(logits - y)).item()
	else:
		# classification
		pred = logits.argmax(dim=-1)
		if y.dim()>1:
			y = y.view(-1)
		return (pred == y.long()).float().mean().item()
	

# ==== ZINC-only runner (graph-level regression; Jetson) ====


def run_experiment_zincOLD(model_name: str, cfg: dict, run_name_suffix: str = ""):
	torch.manual_seed(int(cfg.get("seed", 42)))
	precision = cfg["precision"].lower()
	batch = int(cfg.get("batch", 64))
	task = "regression"

	# 1) Load ZINC (regression-safe)
	graphs = load_zinc_graphs(DATA_ROOT)
	
	# 1b) If SAN_Full, attach eigs (k from cfg or default 16)
	if model_name.lower() == "san_full":
		graphs = compute_and_attach_eigs_list(graphs, dataset_name="ZINC", k=int(cfg.get("eig_k", 16)), strict=False)
		sanity_check_eigs(graphs, k=int(cfg.get("eig_k", 16)))
	# 2) Dims
	in_dim  = int(graphs[0].x.size(-1))
	out_dim = 1  # regression head

	# 3) Build model & head (keep weights FP32 for training with AMP)
	encoder = build_model(model_name, in_dim, cfg.get("hidden", 128), cfg)
	model = GraphHead(encoder, cfg.get("hidden", 128), out_dim).to(DEVICE)

	# 4) Train loop (regression loss)
	opt = make_optimizer(model)
	loss_fn = make_loss_for_task(task)
	scaler = torch.amp.GradScaler("cuda", enabled=(precision=="fp16" and DEVICE.type=="cuda"))
	epochs = int(cfg.get("epochs", 30))

	loader = PyGDataLoader(graphs, batch_size=batch, shuffle=True)

	def forward_batch(b):
		# outputs [B,1]; targets [B,1] float
		with amp_context(precision):
			out = model(b.x, b.edge_index, b.batch,
						eigvec=getattr(b,"eigvec",None),
						eigval=getattr(b,"eigval",None))
			y = b.y
			if y.dim()==1: y = y.unsqueeze(-1)
			return out, y

	t0 = time.time()
	model.train()
	for _ in range(epochs):
		for b in loader:
			b = b.to(DEVICE)
			opt.zero_grad(set_to_none=True)
			out, y = forward_batch(b)
			loss = loss_fn(out, y)
			if scaler.is_enabled():
				scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
			else:
				loss.backward(); opt.step()
	train_total_time_s = time.time() - t0

	# 5) Validation metric (MAE on a held-out split if you have one; else on full)
	# If you have train/val/test indices, use them. ZINC loader we used is already split 10k/1k/1k;
	# but we merged them for simplicity. We'll compute MAE on the same loader just to log a value.
	model.eval(); maes=[]
	with torch.no_grad():
		for b in PyGDataLoader(graphs, batch_size=batch, shuffle=False):
			b = b.to(DEVICE)
			with amp_context(precision):
				pred = model(b.x, b.edge_index, b.batch,
							 eigvec=getattr(b,"eigvec",None),
							 eigval=getattr(b,"eigval",None))
			y = b.y
			if y.dim()==1: y = y.unsqueeze(-1)
			mae = torch.mean(torch.abs(pred - y)).item()
			maes.append(mae)
	val_metric_final = float(sum(maes)/len(maes)) if maes else None  # MAE

	# 6) Timed inference + power (one batch)
	eval_loader = PyGDataLoader(graphs, batch_size=batch, shuffle=False)
	batch0 = next(iter(eval_loader)).to(DEVICE)

	# Option A: AMP autocast for inference (no .half())
	def _fwd():
		with torch.no_grad():
			with amp_context(precision):
				_ = model(batch0.x, batch0.edge_index, batch0.batch,
						  eigvec=getattr(batch0,"eigvec",None),
						  eigval=getattr(batch0,"eigval",None))

	run_name = f"jetson_ZINC_{model_name}_{precision}_b{batch}{run_name_suffix}"
	os.makedirs("traces", exist_ok=True)
	power_trace_path = os.path.join("traces", run_name + "__pwr.jsonl")
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=power_trace_path) as mon2:
		inf_metrics = run_inference_timed(_fwd,
										  reps=int(cfg.get("reps", 60)),
										  warmup=int(cfg.get("warmup", 10)),
										  min_time_s=float(cfg.get("min_infer_time_s", 2.0)))
	mean_power_w = mon2.mean_power()
	#energy_per_inf_j = mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)
	per_batch = batch if isinstance(batch, int) and batch > 0 else 1
	energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"]/1000.0)) / per_batch


	# 7) Emit summary
	summary = {
		"timestamp": datetime.utcnow().isoformat(),
		"platform": "jetson",
		"device_name": platform.platform(),
		"dataset": dataset_name, "model": model_name, "precision": precision,
		"batch": batch if isinstance(batch, (int,str)) else str(batch),
		"seed": cfg.get("seed", 42),
		"phase": "train+infer_node" if is_node else "train+infer_graph",
		# === NEW: inject TRAIN metrics ===
		**res_train,
		# === validation / test (whatever the runner computed) ===
		"val_metric_final": float(val_acc) if 'val_acc' in locals() and val_acc is not None else None,
		"test_acc": float(test_acc) if 'test_acc' in locals() and test_acc is not None else "",
		# === inference (as before) ===
		**inf_metrics,                  # has latency_p50_ms, latency_p95_ms, throughput_req_s
		"mean_power_w": mean_power_w,
		"energy_per_inf_j": energy_per_inf_j,
	}
	print(json.dumps(summary, indent=2))


	# append to CSV with same columns you use
	#header = ["timestamp","platform","device_name","dataset","model","precision","batch","seed","phase",
	#          "train_total_time_s","val_metric_final","latency_p50_ms","latency_p95_ms","throughput_req_s",
	 #         "mean_power_w","energy_per_inf_j"]
	header = ["timestamp","platform","device_name","dataset","model","precision","batch","seed","phase","train_total_time_s","train_mean_power_w",
	"train_energy_total_j","train_energy_per_example_j","train_throughput_examples_s","train_avg_step_ms","train_step_p50_ms","train_step_p95_ms","train_seen_examples",
	"val_metric_final","latency_p50_ms","latency_p95_ms","throughput_req_s","mean_power_w","energy_per_inf_j","test_acc"]
	os.makedirs("outputs", exist_ok=True)

	csv_path = "outputs/runs_jetson.csv"; exists = os.path.exists(csv_path)
	with open(csv_path, "a", newline="") as f:
		w = csv.DictWriter(f, fieldnames=header)
		if not exists: w.writeheader()
		w.writerow({k: summary.get(k,"") for k in header})
	return summary


def run_experiment_zinc(model_name: str, cfg: dict, run_name_suffix: str = ""):
	"""
	ZINC regression runner, self-contained:
	  - preserves your current training loop,
	  - collects TRAIN metrics (res_train) without calling a generic trainer,
	  - supports SAN_Full eigens (if requested),
	  - times inference + samples power,
	  - emits unified summary and appends to CSV.
	"""
	
	# ---- helpers assumed to exist elsewhere in notebook ----
	# - load_zinc_graphs(DATA_ROOT)
	# - compute_and_attach_eigs_list(graphs, dataset_name, k, strict)
	# - sanity_check_eigs(graphs, k)
	# - build_model, make_optimizer, GraphHead, TegraStatsSampler, run_inference_timed
	# - DEVICE (cuda/cpu torch.device)

	# ---------- cfg / seeds ----------
	torch.manual_seed(int(cfg.get("seed", 42)))
	np.random.seed(int(cfg.get("seed", 42)))

	precision = str(cfg.get("precision", "fp32")).lower()
	batch     = int(cfg.get("batch", 64))
	hidden    = int(cfg.get("hidden", 128))
	epochs    = int(cfg.get("epochs", 30))
	reps      = int(cfg.get("reps", 60))
	warmup    = int(cfg.get("warmup", 10))
	min_time  = float(cfg.get("min_infer_time_s", 2.0))

	# autocast context
	def amp_context(prec):
		if prec == "fp16" and DEVICE.type == "cuda":
			return torch.amp.autocast("cuda", dtype=torch.float16)
		return nullcontext()

	# ---------- 1) Load ZINC (regression) ----------
	graphs = load_zinc_graphs(DATA_ROOT)   # your working loader; returns a list-like of PyG Data

	# ---------- 1b) SAN eigens if needed ----------
	if model_name.lower() == "san_full":
		eig_k = int(cfg.get("eig_k", 16))
		graphs = compute_and_attach_eigs_list(graphs, dataset_name="ZINC", k=eig_k, strict=False)
		# optional: assert k presence
		sanity_check_eigs(graphs, k=eig_k)

	# ---------- 2) Dims ----------
	in_dim  = int(graphs[0].x.size(-1))
	out_dim = 1  # ZINC is regression

	# ---------- 3) Build model & head ----------
	encoder = build_model(model_name, in_dim, hidden, cfg)
	model   = GraphHead(encoder, hidden, out_dim).to(DEVICE)

	# ---------- 4) Train loop (instrumented) ----------
	opt     = make_optimizer(model)
	# regression loss (Huber / SmoothL1)
	loss_fn = torch.nn.SmoothL1Loss()

	scaler  = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and DEVICE.type == "cuda"))
	loader  = PyGDataLoader(graphs, batch_size=batch, shuffle=True)

	def forward_batch(b):
		with amp_context(precision):
			out = model(b.x, b.edge_index, b.batch,
						eigvec=getattr(b, "eigvec", None),
						eigval=getattr(b, "eigval", None))
			y = b.y
			if y.dim() == 1:
				y = y.unsqueeze(-1)
			return out, y

	# Train metrics we will compute:
	# - train_total_time_s
	# - train_step_p50_ms, train_step_p95_ms, train_avg_step_ms
	# - train_throughput_examples_s, train_seen_examples
	# - train_mean_power_w, train_energy_total_j, train_energy_per_example_j

	step_times_ms = []
	seen_examples = 0
	run_name = f"jetson_ZINC_{model_name}_{precision}_b{batch}{run_name_suffix}"
	os.makedirs("traces", exist_ok=True)
	train_power_trace = os.path.join("traces", run_name + "__train_pwr.jsonl")

	t0 = time.time()
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=train_power_trace) as mon_train:
		model.train()
		for _ in range(epochs):
			for b in loader:
				b = b.to(DEVICE)
				opt.zero_grad(set_to_none=True)
				t1 = time.time()
				out, y = forward_batch(b)
				loss = loss_fn(out, y)
				if scaler.is_enabled():
					scaler.scale(loss).backward()
					scaler.step(opt)
					scaler.update()
				else:
					loss.backward()
					opt.step()
				if DEVICE.type == "cuda":
					torch.cuda.synchronize()
				step_times_ms.append((time.time() - t1) * 1000.0)
				# count regression examples = #graphs in this batch
				seen_examples += int(y.size(0))
	train_total_time_s = time.time() - t0

	# summarize training timings
	if len(step_times_ms) > 0:
		step_times_ms.sort()
		train_step_p50_ms = step_times_ms[len(step_times_ms) // 2]
		train_step_p95_ms = step_times_ms[int(max(0, np.floor(0.95 * len(step_times_ms) - 1)))]
		train_avg_step_ms = float(np.mean(step_times_ms))
	else:
		train_step_p50_ms = train_step_p95_ms = train_avg_step_ms = None

	# power / energy
	train_mean_power_w = mon_train.mean_power()
	train_energy_total_j = (train_mean_power_w * train_total_time_s) if train_mean_power_w is not None else None
	train_energy_per_example_j = (train_energy_total_j / seen_examples) if (train_energy_total_j is not None and seen_examples > 0) else None
	train_throughput_examples_s = (seen_examples / train_total_time_s) if train_total_time_s > 0 else None

	# package res_train
	res_train = {
		"train_total_time_s": float(train_total_time_s),
		"train_mean_power_w": float(train_mean_power_w) if train_mean_power_w is not None else None,
		"train_energy_total_j": float(train_energy_total_j) if train_energy_total_j is not None else None,
		"train_energy_per_example_j": float(train_energy_per_example_j) if train_energy_per_example_j is not None else None,
		"train_throughput_examples_s": float(train_throughput_examples_s) if train_throughput_examples_s is not None else None,
		"train_avg_step_ms": float(train_avg_step_ms) if train_avg_step_ms is not None else None,
		"train_step_p50_ms": float(train_step_p50_ms) if train_step_p50_ms is not None else None,
		"train_step_p95_ms": float(train_step_p95_ms) if train_step_p95_ms is not None else None,
		"train_seen_examples": int(seen_examples),
	}

	# ---------- 5) Validation metric (MAE over full set or a held-out split) ----------
	# If your load_zinc_graphs merges splits, this is a global MAE just to log a value.
	model.eval(); maes = []
	with torch.no_grad():
		for b in PyGDataLoader(graphs, batch_size=batch, shuffle=False):
			b = b.to(DEVICE)
			with amp_context(precision):
				pred = model(b.x, b.edge_index, b.batch,
							 eigvec=getattr(b, "eigvec", None),
							 eigval=getattr(b, "eigval", None))
			y = b.y
			if y.dim() == 1:
				y = y.unsqueeze(-1)
			mae = torch.mean(torch.abs(pred - y)).item()
			maes.append(mae)
	val_metric_final = float(sum(maes) / len(maes)) if maes else None  # MAE

	# ---------- 6) Timed inference + power (one batch) ----------
	eval_loader = PyGDataLoader(graphs, batch_size=batch, shuffle=False)
	batch0 = next(iter(eval_loader)).to(DEVICE)

	def _fwd():
		with torch.no_grad():
			with amp_context(precision):
				_ = model(batch0.x, batch0.edge_index, batch0.batch,
						  eigvec=getattr(batch0, "eigvec", None),
						  eigval=getattr(batch0, "eigval", None))

	os.makedirs("traces", exist_ok=True)
	infer_power_trace = os.path.join("traces", run_name + "__infer_pwr.jsonl")
	with TegraStatsSampler(interval_ms=200, rail="orin_sum", save_trace_path=infer_power_trace) as mon_inf:
		inf_metrics = run_inference_timed(_fwd,
										  reps=reps,
										  warmup=warmup,
										  min_time_s=min_time)
	mean_power_w = mon_inf.mean_power()
	energy_per_inf_j = (mean_power_w * (inf_metrics["latency_p50_ms"] / 1000.0)) if mean_power_w is not None else None

	# ---------- 7) Emit summary + CSV ----------
	summary = {
		"timestamp": datetime.utcnow().isoformat(),
		"platform": "jetson",
		"device_name": platform.platform(),
		"dataset": "ZINC",
		"model": model_name,
		"precision": precision,
		"batch": batch,
		"seed": int(cfg.get("seed", 42)),
		"phase": "train+infer_graph",
		# TRAIN metrics:
		**res_train,
		# VALIDATION (MAE goes into val_metric_final to keep a single column):
		"val_metric_final": val_metric_final,
		# INFERENCE metrics:
		**inf_metrics,  # latency_p50_ms, latency_p95_ms, throughput_req_s
		"mean_power_w": mean_power_w,
		"energy_per_inf_j": energy_per_inf_j,
		# Optional test field (unused here, keep for schema compatibility):
		"test_acc": ""
	}

	print(json.dumps(summary, indent=2))

	os.makedirs("outputs", exist_ok=True)
	csv_path = "outputs/runs_jetson.csv"
	header = [
		"timestamp","platform","device_name","dataset","model","precision","batch","seed","phase",
		"train_total_time_s","train_mean_power_w","train_energy_total_j","train_energy_per_example_j",
		"train_throughput_examples_s","train_avg_step_ms","train_step_p50_ms","train_step_p95_ms","train_seen_examples",
		"val_metric_final","latency_p50_ms","latency_p95_ms","throughput_req_s","mean_power_w","energy_per_inf_j","test_acc"
	]
	exists = os.path.exists(csv_path)
	with open(csv_path, "a", newline="") as f:
		w = csv.DictWriter(f, fieldnames=header)
		if not exists:
			w.writeheader()
		w.writerow({k: summary.get(k, "") for k in header})
		f.flush()

	return summary

# ==== G) Final orchestrator ========================================================
def orchestrator(args):
	CONFIG["infer_batch"] = args.infer_batch
	CONFIG["batch"] = args.batch
	CONFIG["seed"]= args.seed
	
	#NODE_GENERIC = {"Cora","PubMed"}
	#GRAPH_GENERIC = {"MUTAG","ds002748"}
	#NODE_KHOP    = {"Reddit","ogbn-arxiv"}
	#GRAPH_SPECIAL = {"COLLAB","ZINC"}
	
	NODE_GENERIC = {}
	GRAPH_GENERIC = {"MUTAG"}
	NODE_KHOP    = {}
	GRAPH_SPECIAL = {"COLLAB"}
	
	RUN_PROFILE = "jetson"  # change on x86 machines 
	
	_DATASET_DEFAULTS = {
		"Cora":       {"epochs": 200, "hidden": 128},
		"PubMed":     {"epochs": 200, "hidden": 64},
		"Reddit": {
			"jetson":    {"epochs": 20, "hidden": 128, "khop_hops": 2, "khop_nodes": 8000, "khop_batches_per_epoch": 1, "num_eval_samples": 8},
			"x86_2060":  {"epochs": 20, "hidden": 128, "khop_hops": 2, "khop_nodes": 8000, "khop_batches_per_epoch": 1, "num_eval_samples": 8},
		},
		"ogbn-arxiv": {
			"jetson":    {"epochs": 30, "hidden": 128, "khop_hops": 2, "khop_nodes": 9000, "khop_batches_per_epoch": 1, "num_eval_samples": 10},
			"x86_2060":  {"epochs": 30, "hidden": 128, "khop_hops": 2, "khop_nodes": 9000,  "khop_batches_per_epoch": 1, "num_eval_samples": 10},
		},
		"MUTAG":      {"epochs": 100, "batch": 16},
		"COLLAB":     {"epochs": 10,  "batch": 16},
		"ZINC":       {"epochs": 100, "batch": 8, "eig_k": 16},
		"ds002748":   {"epochs": 50,  "batch": 1},
	}
	_INFER_DEFAULTS = {"reps": 120, "warmup": 20, "min_infer_time_s": 3.0, "seed": 42}
	
	def dataset_defaults(name: str, profile: str = RUN_PROFILE) -> dict:
		base = _DATASET_DEFAULTS.get(name, {}).copy()
		if isinstance(base.get("jetson"), dict) or isinstance(base.get("x86_2060"), dict):
			prof_block = base.get(profile, {})
			for k in ("jetson","x86_2060","x86_5080"): base.pop(k, None)
			base.update(prof_block)
		base.setdefault("hidden", 128)
		base.update({k: v for k, v in _INFER_DEFAULTS.items() if k not in base})
		return base
	
	def _allowed(dataset, model):
		
		if dataset in {"ogbn-arxiv","Reddit"} and model in {"san_full","graphgps_full"}:
			return True
		return True
	
	def run_one(dataset, model, precision, base_cfg: dict, run_suffix=""):
		cfg = deepcopy(base_cfg); cfg.update(dataset_defaults(dataset)); cfg["precision"] = precision
		cfg.setdefault("seed", 42); cfg.setdefault("reps", 60); cfg.setdefault("warmup", 10); cfg.setdefault("min_infer_time_s", 2.0)
	
		if dataset in NODE_GENERIC or dataset in GRAPH_GENERIC:
			summary = run_experiment(dataset, model, cfg, run_name_suffix=run_suffix)
		elif dataset == "COLLAB":
			summary = run_experiment_collab(model, cfg, run_name_suffix=run_suffix)
		elif dataset == "ZINC":
			summary = run_experiment_zinc(model, cfg, run_name_suffix=run_suffix)
		elif dataset in NODE_KHOP:
			summary = run_experiment_khop(dataset, model, cfg, run_name_suffix=run_suffix)
		else:
			raise ValueError(f"Unknown dataset '{dataset}'")
	
		append_summary_to_csv(summary, csv_path="outputs/runs_all.csv")
		return summary
	
	#DATASETS   = ["Cora","PubMed","ogbn-arxiv","MUTAG","ds002748","COLLAB"]
	DATASETS = [args.dataset_name]
	
	MODELS = (
		[args.model]
		if args.model is not None
		else ["gcn","graphsage","gat","gin","graphgps_full"]
	)

	PRECISIONS = (
		[args.precision]
		if args.precision is not None
		else ["fp32","fp16"]
	)
	def run_grid(datasets=DATASETS, models=MODELS, precisions=PRECISIONS, base_cfg: dict = None, run_suffix=""):
		base_cfg = dict(CONFIG) if base_cfg is None else deepcopy(base_cfg)
		for ds in datasets:
			for m in models:
				if not _allowed(ds, m): 
					print(f"[SKIP] {ds} {m}"); continue
				for p in precisions:
					try:
						print(f"=== RUN {ds} / {m} / {p} ===")
						_ = run_one(ds, m, p, base_cfg, run_suffix)
					except Exception as e:
						print(f"[ERROR] {ds} {m} {p} -> {e}")
						traceback.print_exc()
						# log an error row so CSV remains aligned
						err = dict(timestamp=datetime.utcnow().isoformat(), platform="jetson",
								   device_name=platform.platform(), dataset=ds, model=m,
								   precision=p, batch="", seed=base_cfg.get("seed",42),
								   phase="error")
						append_summary_to_csv(err, csv_path="outputs/runs_allN15_1_s42.csv")
					finally:
						# Always clear memory between runs
						#_cleanup_cuda()
						print(f"=== ENDED RUN {ds} {m} {p}" )
	print("Orchestrator set")	
	run_grid(run_suffix=args.run_suffix)

	

# END LOGIC
#CLI PARSER
def parse_args():
    parser = argparse.ArgumentParser("GNN benchmark")

    parser.add_argument("--dataset_name", default="COLLAB",
                        help="Dataset to run (default: COLLAB)")
    parser.add_argument("--model", default=None,
                        help="Model to run (default: all)")
    parser.add_argument("--precision", default=None,
                        choices=["fp32", "fp16"],
                        help="Precision to run (default: both)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--infer-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-suffix", default="__final")

    return parser.parse_args()
    
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    args = parse_args()
	
	#orchestrator(
    #    dataset_name=args.dataset,
    #    model_name=args.model,
    #    batch=args.batch,
    #    infer_batch=args.infer-batch
    #    precision=args.precision,
    #    seed=args.seed,
    #)
    orchestrator(args)
    #run_grid(run_suffix=args.run_suffix)	

if __name__ == "__main__":
    main()
