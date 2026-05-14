# MOSAIC

Official PyTorch implementation of:

> **Interpretable Stability-Aware Multimodal Connectome Learning With Structure-Anchored Fusion for Brain Disorder Diagnosis**

MOSAIC jointly models functional (FC), structural (SC) and effective (EC)
connectivity with (i) stability-aware grouped RL for subnetwork mask
discovery, (ii) topology-matched encoders (Transformer / DHT-HyperGNN /
signed Transformer), and (iii) SC-anchored hierarchical fusion.

## Repository layout

```
code/
├── configs/        # YAML config per dataset
├── datasets/       # Dataset adapters
├── models/mosaic.py  # Mask policy + encoders + SC-anchored fusion
├── main.py         # Train / eval entry point
└── README.md
```

## Installation

```bash
conda create -n mosaic python=3.10 -y
conda activate mosaic
pip install torch>=2.1 numpy pyyaml scikit-learn
```

## Datasets

All cohorts are publicly available; due to data-sharing agreements they
are not redistributed here. Place per-subject `.npz` files (containing
`FC`, `SC`, `EC`, `label`) under `data/<cohort>/`.

| Cohort | Source | Atlas |
|---|---|---|
| ABIDE-I  | http://preprocessed-connectomes-project.org/abide/ | Schaefer100×7 |
| ABIDE-II | http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html | Schaefer100×7 |
| ADHD-200 | http://fcon_1000.projects.nitrc.org/indi/adhd200/ | AAL116 |
| OASIS-3  | https://www.oasis-brains.org/ | DK68 |
| HCP (healthy reference only) | https://www.humanconnectome.org/study/hcp-young-adult | — |

Preprocessing follows Section 3.1 of the paper (CPAC for rs-fMRI; MIND /
DTI for SC; regularized VAR(1) for EC).

## Training & evaluation

```bash
# Train (5-fold group-stratified CV)
python main.py --config configs/abide1.yaml --mode train

# Evaluate
python main.py --config configs/abide1.yaml --mode eval --ckpt runs/abide1/best.pt
```

Replace `abide1` with `abide2`, `adhd200`, or `oasis` for the other
cohorts. Reported metrics: ACC, SEN, SPE.

## License

MIT.
