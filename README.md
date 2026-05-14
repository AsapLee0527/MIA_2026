# MOSAIC: Interpretable Stability-Aware Multimodal Connectome Learning

Official PyTorch implementation of:

> **Interpretable Stability-Aware Multimodal Connectome Learning With Structure-Anchored Fusion for Brain Disorder Diagnosis**

---

## At a glance

| Aspect | Detail |
|---|---|
| **Task** | Subject-level binary classification of brain disorders (ASD / ADHD / Dementia vs. healthy controls) from multimodal connectomes. |
| **Input** | Three connectivity matrices per subject — functional `FC`, structural `SC`, and effective `EC` — defined on a cohort-specific brain atlas. |
| **Output** | (i) Diagnostic prediction, (ii) per-modality hard 0/1 subnetwork mask (interpretable subset of edges), (iii) fused multimodal representation. |
| **Cohorts** | ABIDE-I, ABIDE-II, ADHD-200, OASIS-3 (+ HCP as healthy reference only). |
| **Core ideas** | (1) Stability-aware grouped RL for mask discovery; (2) Topology-matched heterogeneous encoders (Transformer / DHT-HyperGNN / signed Transformer); (3) SC-anchored hierarchical cross-modal fusion. |
| **Eval protocol** | Group-stratified 5-fold cross-validation; metrics: ACC, SEN, SPE. |

MOSAIC jointly models functional (FC), structural (SC), and effective (EC)
connectivity through:

1. **Stability-aware grouped reinforcement learning** for modality-specific
   subnetwork mask discovery — hard 0/1 masks trained with a
   group-relative Monte-Carlo policy gradient and a
   *perturbation-consistency* reward, so that retained edges are
   simultaneously *task-relevant* and *robust to distributional
   perturbations*.
2. **Topology-matched heterogeneous encoders** that respect each
   modality's geometry: a vanilla Transformer for the dense, undirected
   FC; a Directed-Hypergraph-Transform (DHT)-based HyperGNN for the
   sparse, hub-driven SC; and a direction-aware (signed) Transformer for
   the asymmetric EC.
3. **SC-anchored hierarchical fusion**: the structural backbone SC is
   used as the *key/value context* in cross-modal attention to
   recalibrate the noisier FC/EC streams, yielding a stable shared
   representation for downstream classification.

---

## Repository layout

```
code/
├── configs/                  # YAML configs per dataset
│   ├── abide1.yaml
│   ├── abide2.yaml
│   ├── adhd200.yaml
│   └── oasis.yaml
├── datasets/                 # Cohort-specific dataset adapters
│   ├── __init__.py
│   ├── base.py
│   └── connectome_dataset.py
├── models/                   # MOSAIC model definition
│   ├── __init__.py
│   └── mosaic.py             # Mask policy + topology-matched encoders
│                             # + SC-anchored hierarchical fusion
├── main.py                   # Train / evaluate entry point
│                             # (perturbation, losses, metrics inlined)
├── .gitignore
└── README.md
```

The implementation is **intentionally compact** so that every component
described in the paper is easy to locate. We deliberately avoid splitting
the codebase into many small files; instead:

* All four MOSAIC building blocks — (i) the mask policy
  $\pi_\theta^{(m)}$, (ii) the FC Transformer, (iii) the SC DHT-HyperGNN,
  (iv) the EC signed Transformer, plus (v) the SC-anchored hierarchical
  fusion module — are implemented in a single file
  [`models/mosaic.py`](models/mosaic.py).
* The modality-aware perturbation operator $\mathcal{T}(\cdot)$, the
  joint loss $\mathcal{L}_{\text{total}}$ (classification + entropy
  regularization + RL loss + sparsity prior) and the ACC / SEN / SPE
  metrics are inlined in [`main.py`](main.py).

### What each directory contains

* **`configs/`** — One YAML per cohort. A YAML fully specifies a run:
  atlas / number of ROIs, train-eval split, optimizer, RL hyper-parameters
  (group size $K$, perturbation strength), encoder widths, and fusion
  options. The four shipped configs reproduce the four diagnostic rows of
  Table 2 in the paper. To run on a new cohort, copy any existing YAML
  and edit the data path / atlas size.
* **`datasets/`** — Lightweight adapters that map each cohort's released
  format (per-subject `.npz` containing `FC`, `SC`, `EC`, `label`) into a
  unified PyTorch `Dataset`. `base.py` defines the abstract interface;
  `connectome_dataset.py` is the concrete implementation used for all
  four cohorts. Cohort-specific quirks (e.g. atlas size, label encoding)
  are handled here so that everything downstream is cohort-agnostic.
* **`models/`** — The MOSAIC network. `mosaic.py` exposes a single
  `MOSAIC` class whose `forward(FC, SC, EC)` returns the logits, the
  three masks, and the fused embedding. Internal sub-modules
  (`MaskPolicy`, `FCTransformer`, `SCHyperGNN`, `ECSignedTransformer`,
  `SCAnchoredFusion`) can be imported independently for analysis.
* **`main.py`** — Training / evaluation / ablation entry point. All
  command-line switches required to reproduce the tables and figures of
  the paper are routed through here (see *Reproducing the paper* below).

---

## Paper ↔ code mapping

The table below points each major equation / table / figure in the
paper to the exact place in the code where it is realized.

| Paper element | Where in the code |
|---|---|
| Eq. (1)–(3): mask policy $\pi_\theta^{(m)}$ and group-relative reward | `MaskPolicy` in `models/mosaic.py`; reward / advantage assembled in `main.py::compute_rl_loss` |
| Eq. (4): perturbation operator $\mathcal{T}(\cdot)$ | `main.py::perturb_modality` (modality-aware: rewiring for SC, sign flip for EC, edge dropout for FC) |
| Eq. (5): joint loss $\mathcal{L}_{\text{total}}$ | `main.py::training_step` |
| Sec. 3.4 / FC Transformer | `FCTransformer` in `models/mosaic.py` |
| Sec. 3.5 / SC DHT-HyperGNN | `SCHyperGNN` in `models/mosaic.py` |
| Sec. 3.6 / EC signed Transformer | `ECSignedTransformer` in `models/mosaic.py` |
| Sec. 3.7 / SC-anchored fusion | `SCAnchoredFusion` in `models/mosaic.py` |
| Tab. 2 (diagnostic performance) | `python main.py --mode train` per cohort |
| Tab. 3 (fusion design comparison) | `--fusion {concat,gated,bidir,reverse,sc_anchor}` |
| Tab. 4 (ablation) | `--ablation {no_pert,G1,stgumbel,topk,softgate,l1}` |
| Fig. 5 (modality contribution) | `--modalities {FC,SC,EC,FC+SC,FC+EC,SC+EC,FC+SC+EC}` |
| Fig. 6 (SC-anchor ablation) | `--fusion {sc_anchor,sc2fc_only,sc2ec_only,shared}` |
| Sec. 3.8–3.9 (HCP healthy-reference analyses) | Operates on the per-subject mask/saliency dumps produced by `--mode eval --dump_masks`; downstream statistics scripts will be released with the camera-ready version. |

---

## Installation

```bash
conda create -n mosaic python=3.10 -y
conda activate mosaic
pip install torch>=2.1 numpy pyyaml scikit-learn
```

Tested with PyTorch ≥ 2.1 and CUDA 11.8 / 12.1. CPU fallback is supported
but training is significantly slower. A single GPU with ≥ 12 GB memory
is sufficient for all four cohorts at the default batch size (16); see
the FAQ below for memory-saving options.

---

## Dataset acquisition

All four diagnostic cohorts and the HCP healthy-reference cohort are
**publicly available**; due to data-sharing agreements they are not
redistributed in this repository. Please obtain access through the
official channels below and place the raw files under `data/<cohort>/`.

### Preprocessing pipeline (shared across cohorts)

```
raw scans  ──►  CPAC-based rs-fMRI preproc  ──►  ROI-wise BOLD time series
   │                                                       │
   │                                                       ├─►  FC : Pearson + Fisher z
   │                                                       └─►  EC : regularized VAR(1)
   │
   └────────►  T1 (or DTI) preproc  ──►  SC :  MIND morphometric similarity
                                                (or DTI streamline count for ADHD-200)
```

The exact CPAC / FreeSurfer settings follow Section 3.1 of the paper.

### 1. ABIDE-I

* Source: **ABIDE Preprocessed Initiative**
  ([http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/))
* Sample after QC: 965 subjects (502 NC, 463 ASD).
* Atlas: Schaefer100 × 7-network parcellation.
* Modalities:
  * `FC` — Pearson correlation of ROI-wise BOLD time series, Fisher
    z-transformed.
  * `SC` — MIND-based morphometric similarity computed from cortical
    thickness and regional volume on T1.
  * `EC` — Regularized VAR(1) effective connectivity estimated from
    BOLD.
* Citation: Di Martino et al., 2014.

### 2. ABIDE-II

* Source: **ABIDE-II via NITRC / Preprocessed Connectomes Project**
  ([http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html))
* Sample after QC: 863 subjects (478 NC, 385 ASD).
* Atlas / modalities: identical to ABIDE-I (Schaefer100 × 7).
* Citation: Di Martino et al., 2017.

### 3. ADHD-200

* Source: **ADHD-200 Sample**
  ([http://fcon_1000.projects.nitrc.org/indi/adhd200/](http://fcon_1000.projects.nitrc.org/indi/adhd200/))
* Sample after QC: 762 subjects (484 NC, 278 ADHD).
* Atlas: AAL116.
* Modalities:
  * `FC` — Pearson correlation + Fisher z.
  * `SC` — Diffusion-derived structural connectivity (DTI streamline
    count, log-normalized).
  * `EC` — Regularized VAR(1) from BOLD.
* Citation: ADHD-200 Consortium, 2012.

### 4. OASIS-3

* Source: **OASIS-3** ([https://www.oasis-brains.org/](https://www.oasis-brains.org/))
  (registration and DUA required).
* Sample after QC: 669 subjects (542 NC, 127 Dementia).
* Atlas: DK68.
* Modalities: We use the released **DK68 multimodal tensor package**,
  which provides FC, the structural prior, and EC pre-computed in DK68
  space.
* Citation: LaMontagne et al., 2019; Jack et al., 2018.

### 5. HCP (healthy reference only)

* Source: **HCP Young Adult S1200 release**
  ([https://www.humanconnectome.org/study/hcp-young-adult](https://www.humanconnectome.org/study/hcp-young-adult))
  (free registration and DUA required).
* Sample: 853 multimodal subjects.
* Use: HCP is **only** used for the healthy-reference / normative
  reference statistics analyses (Sec. 3.8–3.9 of the paper). It is **not**
  used to train MOSAIC.

After downloading and preprocessing, the expected layout is:

```
data/
├── abide1/   {subject_id}.npz   # arrays: FC, SC, EC, label
├── abide2/   {subject_id}.npz
├── adhd200/  {subject_id}.npz
├── oasis/    {subject_id}.npz
└── hcp/      {subject_id}.npz   # healthy reference only
```

Each `.npz` stores three connectivity matrices (`FC`, `SC`, `EC`) on the
cohort's atlas, plus a binary `label` (NC = 0, disease = 1) where
applicable.

---

## Configuration files

Each cohort YAML has the following structure (annotated):

```yaml
data:
  root: data/abide1            # directory of per-subject .npz files
  atlas_size: 100              # number of ROIs (Schaefer100 / AAL116 / DK68)

train:
  folds: 5                     # group-stratified CV
  batch_size: 16
  epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-5
  seed: 42

model:
  d_model: 128                 # encoder hidden width
  n_heads: 4                   # Transformer / signed-Transformer heads
  n_layers: 2
  fusion: sc_anchor            # {concat, gated, bidir, reverse, sc_anchor}

rl:                            # stability-aware grouped policy gradient
  group_size: 8                # K samples per subject per step
  pert_strength: 0.1           # strength of T(.)
  sparsity_lambda: 1.0e-3      # L0/L1 prior on the mask
  entropy_beta: 1.0e-2         # entropy regularizer on pi_theta
```

Override any field from the command line, e.g.
`python main.py --config configs/abide1.yaml --rl.group_size 16`.

---

## Training

```bash
# ABIDE-I (Schaefer100x7, 5-fold group-stratified CV)
python main.py --config configs/abide1.yaml --mode train

# ABIDE-II
python main.py --config configs/abide2.yaml --mode train

# ADHD-200
python main.py --config configs/adhd200.yaml --mode train

# OASIS
python main.py --config configs/oasis.yaml --mode train
```

By default each run writes to `runs/<cohort>/` and logs per-fold ACC / SEN / SPE.

## Evaluation

```bash
python main.py --config configs/abide1.yaml --mode eval --ckpt runs/abide1/best.pt
```

To dump per-subject masks (used by Sec. 3.8–3.9 analyses):

```bash
python main.py --config configs/abide1.yaml --mode eval \
               --ckpt runs/abide1/best.pt --dump_masks runs/abide1/masks/
```

Reported metrics: ACC, SEN, SPE under group-stratified 5-fold cross
validation.

---

## Reproducing the paper

| Section | Command |
|---------|---------|
| Tab. 2 / Diagnostic Performance         | `python main.py --config configs/<cohort>.yaml --mode train` |
| Tab. 4 / Ablation                       | `python main.py --config configs/<cohort>.yaml --ablation {no_pert,G1,stgumbel,topk,softgate,l1}` |
| Fig. 5 / Connectivity contribution      | `python main.py --config configs/<cohort>.yaml --modalities {FC,SC,EC,FC+SC,FC+EC,SC+EC,FC+SC+EC}` |
| Tab. 3 / Fusion design comparison       | `python main.py --config configs/<cohort>.yaml --fusion {concat,gated,bidir,reverse,sc_anchor}` |
| Fig. 6 / SC-anchor ablation             | `python main.py --config configs/<cohort>.yaml --fusion {sc_anchor,sc2fc_only,sc2ec_only,shared}` |

Cross-cohort reproducibility (Fig. 7) and HCP healthy-reference analyses
(Fig. 8–10) operate on the per-subject mask/saliency outputs produced by
`--dump_masks`. The downstream statistics scripts are available from the
authors upon request and will be released alongside the camera-ready
version.

---

## FAQ

**Q1. Can I run a single modality (e.g. FC only)?**
Yes — pass `--modalities FC`. Internally this disables the SC/EC
encoders and replaces SC-anchored fusion by a pass-through, exactly as
in the Fig. 5 ablation.

**Q2. My GPU only has 8 GB. What can I do?**
Lower `train.batch_size` to 8 and `rl.group_size` to 4. The default
configs target a 12 GB card; halving both keeps the model well within
8 GB at the cost of slightly noisier RL gradients.

**Q3. How do I plug in a new atlas / cohort?**
(i) Pre-compute `FC`, `SC`, `EC` for each subject into a `.npz` with the
schema described above; (ii) copy `configs/abide1.yaml`, change
`data.root` and `data.atlas_size`; (iii) `python main.py --config
configs/<your>.yaml --mode train`. No code change is required as long as
all three matrices share the same ROI ordering.

**Q4. Why a single `mosaic.py` instead of one file per module?**
The five components (mask policy + three encoders + fusion) share a
common dimension and tensor protocol, and several of them re-use small
helpers (e.g. the masked attention kernel). Keeping them in one file
makes the dimension flow easy to audit, and the file is still under a
few hundred lines.

**Q5. Are pretrained checkpoints released?**
Not in this repository, since the underlying cohorts have DUAs that
forbid releasing subject-derived models. After running the training
commands above, your `runs/<cohort>/best.pt` should match the numbers
reported in Tab. 2 within the standard CV variance.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mosaic2026,
  title   = {Interpretable Stability-Aware Multimodal Connectome Learning
             With Structure-Anchored Fusion for Brain Disorder Diagnosis},
  author  = {Anonymous},
  journal = {Medical Image Analysis},
  year    = {2026}
}
```

## License

Released under the MIT License.
