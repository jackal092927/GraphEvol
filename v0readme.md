# GraphEvol v0 Pipeline (PA Ordering Match Classification)

This v0 pipeline builds an end-to-end experiment for **binary classification**:  
given a graph and a vertex ordering, decide whether the ordering matches a **preferential attachment (PA)** growth process.

## What the code does
- **Data generation**
  - Generate PA graphs (optionally with light ER noise).
  - Positive samples: graph + true PA ordering.
  - Negative samples: same graph + randomly permuted ordering.
- **TDA features**
  - Lower-star filtration on the vertex ordering.
  - Extended persistence diagrams (ordinary, relative, extended+, extended-).
  - Persistence images (fixed grid) + simple graph stats.
- **Model**
  - Default: MLP classifier on TDA + graph stats.
  - Optional: GAT graph classifier (as a branch; TDA still computed for plots).
- **Evaluation + visualization**
  - Metrics: accuracy/precision/recall/F1 + confusion matrix.
  - Plots: sample graphs, persistence images, class-mean TDA grids, feature norm histogram, training curves.

## How many samples are generated
Default settings in `v0_pipeline.py`:
- `num_graphs = 120` PA graphs
- Each graph gives **1 positive + 1 negative sample**
- Total: **120 positive** and **120 negative** samples (240 total)

You can change this in `main()` or by editing `make_dataset(...)`.

## Train/val/test split
Default split in `v0_pipeline.py`:
- Train: 70%
- Val: 15%
- Test: 15% (remainder)

## How to run (conda env)
Use the following to reproduce the experiments in the `tda` conda environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tda

# MLP (default, TDA-focused)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python v0_pipeline.py --model mlp

# Optional GAT branch
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python v0_pipeline.py --model gat
```

## Outputs
- `outputs_tda/` (default) or `outputs_gat/`
- Key files:
  - `sample_graphs.png` – example positive vs negative graph coloring
  - `persistence_images.png` – single-sample TDA images (ord H0/H1)
  - `tda_class_means.png` – **mean persistence images** for positives vs negatives + diff
  - Quick link: `outputs_tda/tda_class_means.png`
  - `feature_norms.png` – quick separation check of TDA feature norms
  - `training_curves.png` – loss/accuracy curves
  - `confusion_matrix.png`

## Notes on TDA features (what is being learned)
- Filtration: **lower-star** on the ordering values assigned to vertices.
- Extended persistence diagrams: captures connected components (H0) and cycles (H1) across sublevel and superlevel structure.
- Persistence images: fixed-size grids (12x12) for each diagram type and dimension.
- Final feature vector = concatenated persistence images + graph stats.

## Files of interest
- `v0_pipeline.py` – all data generation, TDA features, models, evaluation, plots.
- `init_sample_code.py` – original PA ordering regression demo.
- `test0.py` – extended persistence examples on grid graphs.

If you want to adjust the ordering definition or the filtration, the hook is in:
`ordering_from_pa(...)` and `simplex_tree_lower_star(...)` in `v0_pipeline.py`.
