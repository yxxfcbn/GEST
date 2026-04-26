# TSP GNN Sparse Solver

This project uses a graph neural network (GNN) to score candidate edges for Traveling Salesman Problem (TSP) instances, then builds a sparse graph and solves the resulting TSP model with SCIP.

The main workflow is:

1. Read TSP coordinate instances and edge labels.
2. Build graph features including distance, iterative MST weight, KNN indicator, and q-ratio feature.
3. Train a GATv2-based edge classifier to predict whether an edge belongs to a good/optimal tour.
4. Use the trained model to sparsify a TSP graph.
5. Add Delaunay and Christofides edges for feasibility support.
6. Solve the sparse TSP formulation with PySCIPOpt.

## Main files

- `tsp_train.py`  
  Trains the edge-scoring GNN model and saves `tsp_gnn_model.pt`.

- `tsp_scip_solve.py`  
  Loads the trained model, builds a sparse graph, and solves the sparse TSP with SCIP.

- `requirements.txt`  
  Python dependencies for training and solving.

## Project structure

The scripts expect data under the project root in a layout similar to:

```text
.
├── tsp_train.py
├── tsp_scip_solve.py
├── requirements.txt
├── tsp/
│   ├── CLKhard/
│   │   ├── coords/ or *.tsp
│   │   └── labels/ or *_label.txt files
│   ├── LKCChard/
│   └── CLKeasy/
└── tsp_gnn_model.pt
```

A `.tsp` coordinate file should contain a `NODE_COORD_SECTION`. Label files are expected to be named like:

```text
<instance_id>_label.txt
```

or:

```text
<instance_id>_labels.txt
```

## Installation

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## PyTorch and CUDA note

The `requirements.txt` file contains a generic `torch` dependency. If you need GPU acceleration, install the CUDA-specific PyTorch wheel first from the official PyTorch selector:

<https://pytorch.org/get-started/locally/>

For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## SCIP / PySCIPOpt note

`tsp_scip_solve.py` depends on `pyscipopt`. On some systems, especially Windows, installing `pyscipopt` may require installing the SCIP Optimization Suite first:

<https://www.scipopt.org/index.php#download>

After SCIP is installed, run:

```bash
pip install pyscipopt
```

## Training

Run:

```bash
python tsp_train.py
```

The training script currently uses data from:

- `tsp/CLKhard`
- `tsp/LKCChard`
- `tsp/CLKeasy`

It trains a `TSP_EdgeGNN` model and saves the best model state as:

```text
tsp_gnn_model.pt
```

## Solving with SCIP

After training, run the sparse SCIP solver:

```bash
python tsp_scip_solve.py
```

To solve a specific instance id from the configured data directory:

```bash
python tsp_scip_solve.py --id 000
```

Common options:

```bash
python tsp_scip_solve.py --id 000 --coarse_K 15 --p 0.5 --topk 10
```

Arguments:

- `--id`: instance id, for example `000`
- `--coarse_K`: KNN/coarse candidate parameter
- `--p`: GNN edge-score threshold
- `--topk`: keep top-k high-scoring incident edges per node
- `--no_first_round_coarse`: disable the Delaunay + MST first-round coarse graph construction
- `--no_full`: skip complete graph solving logic if enabled in your script version

## Method overview

The GNN model is an edge classifier based on `GATv2Conv`. For every directed edge, it uses:

- normalized node coordinates
- Euclidean distance feature
- iterative MST feature
- KNN indicator feature
- q-ratio feature based on nearest-neighbor distance

During solving, the model predicts edge probabilities. Edges above threshold `p` are kept, optionally filtered by `topk`. The sparse graph is then augmented with Delaunay and Christofides edges before SCIP solves the TSP model with degree constraints and subtour elimination cuts.

## Notes

- Large `.pt` model files and generated images may be better stored with Git LFS or excluded from Git if they are not required for reproduction.
- Dataset files may be large; consider documenting whether they are included in the repository or should be downloaded separately.
- If `scipy.spatial.Delaunay` is unavailable, some coarse graph features may be skipped or fail depending on the script path used.
