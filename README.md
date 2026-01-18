# greenbionet

Part 1 - GNN Benchmarking

# GNN Benchmarking Pipeline (Jetson / x86)

This repository contains a benchmarking pipeline for evaluating Graph Neural Networks (GNNs) across **performance, energy consumption, and resource usage** on heterogeneous hardware platforms.

The pipeline was developed to support reproducible experimental studies on embedded GPUs (NVIDIA Jetson AGX Orin) and standard x86 systems.

---

## Features

- Multiple GNN models:
  - GCN
  - GraphSAGE
  - GAT
  - GIN
  - GraphGPS
- Mixed precision execution:
  - FP32
  - FP16 (AMP)
- Configurable batch sizes and random seeds
- Training + repeated inference benchmarking
- Energy and power measurement (platform-aware)
- Unified CSV logging for all runs
- Disk caching of expensive preprocessing (e.g., Laplacian eigenpairs)

---

## Repository structure

- run_benchmark.py # Main execution script
outputs/ # CSV logs (created automatically)
eigs_cache/ # Cached Laplacian eigenpairs
README.md
neuroimaging/fmri_preprocessing.py #code to preprocess fmriprep derivatives from Openneuro

---

## Requirements

- Python ≥ 3.9
- PyTorch
- PyTorch Geometric
- CUDA (for GPU execution)
- CodeCarbon (for energy measurement)

Exact package versions depend on the target platform (Jetson vs x86).

---

## Running the benchmark

All experiments are executed through a single script:

python run_benchmark.py [OPTIONS]

Example: single configuration

python run_benchmark.py \
  --dataset COLLAB \
  --model gcn \
  --precision fp32 \
  --batch 16 \
  --infer-batch 16 \
  --seed 42 \
  --run-suffix __final


Recorded data are then saved into a single csv file:runs_all.csv

