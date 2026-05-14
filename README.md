# MOSAIC: Interpretable Stability-Aware Multimodal Connectome Learning

Official PyTorch implementation of:

> **Interpretable Stability-Aware Multimodal Connectome Learning With Structure-Anchored Fusion for Brain Disorder Diagnosis**

MOSAIC jointly models functional (FC), structural (SC), and effective (EC)
connectivity through:

1. **Stability-aware grouped reinforcement learning** for modality-specific
   subnetwork mask discovery (hard 0/1 masks, group-relative Monte Carlo
   policy gradient with perturbation-consistency reward).
2. **Topology-matched heterogeneous encoders**: a Transformer for FC, a
   DHT-based HyperGNN for SC, and a direction-aware (signed) Transformer
   for EC.
3. **SC-anchored hierarchical fusion**: SC serves as the key/value context
   that recalibrates FC and EC through cross-modal attention.

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
└── README.md
```

The implementation is intentionally compact: the modality-aware
perturbation operator $\mathcal{T}(\cdot)$, the joint loss
$\mathcal{L}_{\text{total}}$ and the ACC/SEN/SPE metrics are inlined in
`main.py`; the four MOSAIC building blocks (mask policy, FC Transformer,
SC DHT-HyperGNN, EC direction-aware Transformer, SC-anchored fusion) all
live in `models/mosaic.py`.

---

## Installation

```bash
conda create -n mosaic python=3.10 -y
conda activate mosaic
pip install torch>=2.1 numpy pyyaml scikit-learn
```

Tested with PyTorch >= 2.1 and CUDA 11.8 / 12.1. CPU fallback is supported
but training is significantly slower.

---

## Dataset acquisition

All four diagnostic cohorts and the HCP healthy-reference cohort are
**publicly available**; due to data-sharing agreements they are not
redistributed in this repository. Please obtain access through the
official channels below and place the raw files under `data/<cohort>/`.

### 1. ABIDE-I

* Source: **ABIDE Preprocessed Initiative**
  ([http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/))
* Sample: 965 subjects (502 NC, 463 ASD).
* Atlas: Schaefer100 × 7-network parcellation.
* Required outputs:
  * `FC`: Pearson correlation of ROI-wise BOLD time series + Fisher's
    z-transform.
  * `SC`: MIND-based morphometric similarity from cortical thickness and
    regional volume (T1).
  * `EC`: Regularized VAR(1) effective connectivity from BOLD.
* Citation: Di Martino et al., 2014.

### 2. ABIDE-II

* Source: **ABIDE-II via NITRC / Preprocessed Connectomes Project**
  ([http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html))
* Sample: 863 subjects (478 NC, 385 ASD).
* Atlas / modalities: identical to ABIDE-I (Schaefer100 × 7).
* Citation: Di Martino et al., 2017.

### 3. ADHD-200

* Source: **ADHD-200 Sample**
  ([http://fcon_1000.projects.nitrc.org/indi/adhd200/](http://fcon_1000.projects.nitrc.org/indi/adhd200/))
* Sample: 762 subjects (484 NC, 278 ADHD).
* Atlas: AAL116.
* Required outputs:
  * `FC`: Pearson correlation + Fisher z.
  * `SC`: Diffusion-derived structural connectivity (DTI-based).
  * `EC`: Regularized VAR(1) from BOLD.
* Citation: ADHD-200 Consortium, 2012.

### 4. OASIS-3

* Source: **OASIS-3** ([https://www.oasis-brains.org/](https://www.oasis-brains.org/))
  (registration and DUA required).
* Sample: 669 subjects (542 NC, 127 Dementia).
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
* Use: HCP is **only** used for the healthy-reference and normative
  reference statistics analyses (Sec. 3.8–3.9 of the paper). It is **not**
  used for training MOSAIC.

> **Note.** Re-running the rs-fMRI preprocessing pipeline requires
> CPAC; we follow the standard pipeline described in
> Section 3.1 of the paper.

After downloading, the expected layout is:

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

## Evaluation

```bash
python main.py --config configs/abide1.yaml --mode eval --ckpt runs/abide1/best.pt
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
(Fig. 8–10) operate on per-subject mask/saliency outputs produced by
the commands above; the analysis scripts are available from the authors
upon request and will be released alongside the camera-ready version.

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
