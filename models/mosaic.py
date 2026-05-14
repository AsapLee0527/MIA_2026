"""MOSAIC: Multimodal cOnnectome learning with Stability-Aware
modaliIty-specific masking and Cross-modal SC-anchored fusion.

This single file contains the four building blocks described in
Section 2 of the paper:

  1. Modality-specific Bernoulli mask policy (Sec. 2.3)
  2. Stability-aware grouped REINFORCE optimization (Sec. 2.4)
  3. Topology-matched encoders for FC / SC / EC (Sec. 2.5)
       - FC: Transformer
       - SC: DHT-based HyperGNN (lightweight ROI-aggregating variant)
       - EC: direction-aware Transformer (signed-channel aware)
  4. SC-anchored hierarchical cross-attention fusion (Sec. 2.6)

The implementation is intentionally self-contained and modest in size so
that reviewers and downstream users can read the model end-to-end without
chasing across many files. Hyperparameters are read from the YAML config
passed to the top-level ``MOSAIC`` constructor.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Modality-specific mask policy (Sec. 2.3)
# =============================================================================
class MaskPolicy(nn.Module):
    """Connectome-conditioned Bernoulli policy over R ROI retentions.

    For modality m and subject i, reads the full R x R connectome row by row
    and emits one logit per ROI; the per-ROI Bernoulli marginal is therefore
    *jointly* conditioned on the global multivariate context even though the
    sampling distribution is factorized for tractability.

    Forward returns ``(p, m, logp)`` where
        p     : (B, R)        retention probability per ROI
        m     : (B, R)        sampled hard 0/1 mask (or argmax if deterministic)
        logp  : (B,)          log Pi(m | A) summed over R Bernoulli factors
    """

    def __init__(self, num_rois: int, hidden: int = 128) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.fc1 = nn.Linear(num_rois, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, A: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        # A: (B, R, R); use rows as per-ROI feature vectors
        h = F.gelu(self.fc1(A))
        logits = self.fc2(h).squeeze(-1)        # (B, R)
        p = torch.sigmoid(logits)
        if deterministic:
            m = (p >= 0.5).float()
        else:
            m = torch.bernoulli(p)
        # log-prob of the sampled mask
        eps = 1e-8
        logp = (m * torch.log(p + eps)
                + (1 - m) * torch.log(1 - p + eps)).sum(dim=-1)
        return {"p": p, "m": m, "logp": logp}


def apply_mask(A: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """tilde A = M A M with M = diag(m)."""
    return A * m.unsqueeze(-1) * m.unsqueeze(-2)


# =============================================================================
# 2. Topology-matched encoders (Sec. 2.5)
# =============================================================================
class FCTransformer(nn.Module):
    """Transformer encoder over ROI tokens for FC."""

    def __init__(self, num_rois: int, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(num_rois, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: (B, R, R) -> tokens (B, R, d)
        h = self.proj(A)
        return self.encoder(h)


class SCDHTHyperGNN(nn.Module):
    """Lightweight DHT-based HyperGNN for SC.

    Implements a single round of edge-node <-> ROI-hyperedge message
    passing on the dual hypergraph induced by the SC adjacency matrix.
    For ROI-level downstream heads we aggregate edge-node messages back
    to nodes via row/column incidence.
    """

    def __init__(self, num_rois: int, d_model: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.proj_node = nn.Linear(num_rois, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # node features from row context
        x = self.proj_node(A)                                  # (B, R, d)
        B, R, d = x.shape
        # build edge-node features: concat[u, v, w_uv]
        u = x.unsqueeze(2).expand(B, R, R, d)
        v = x.unsqueeze(1).expand(B, R, R, d)
        w = A.unsqueeze(-1)
        e = self.edge_mlp(torch.cat([u, v, w], dim=-1))        # (B, R, R, d)
        # ROI-hyperedge aggregation: sum over edges incident on each ROI
        deg = (A != 0).float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        msg = e.sum(dim=2) / deg                               # (B, R, d)
        return self.node_update(torch.cat([x, msg], dim=-1))


class ECSignedTransformer(nn.Module):
    """Direction-aware Transformer for EC.

    When signed effective interactions are available, EC is split into
    A^{E,+} and A^{E,-} and concatenated with their transposes so that
    incoming and outgoing dependencies are preserved as separate channels.
    """

    def __init__(self, num_rois: int, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1,
                 signed: bool = True) -> None:
        super().__init__()
        self.signed = signed
        in_dim = 4 * num_rois if signed else 2 * num_rois
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        if self.signed:
            Ap = F.relu(A)
            An = F.relu(-A)
            x = torch.cat([Ap, Ap.transpose(-1, -2),
                           An, An.transpose(-1, -2)], dim=-1)
        else:
            x = torch.cat([A, A.transpose(-1, -2)], dim=-1)
        h = self.proj(x)
        return self.encoder(h)


# =============================================================================
# 3. SC-anchored hierarchical fusion (Sec. 2.6)
# =============================================================================
class SCAnchoredFusion(nn.Module):
    """SC-as-key/value cross-attention that refines FC and EC under SC.

    Variants supported via ``variant``:
        sc_anchor    : default; SC is the key/value context for both FC and EC
        sc2fc_only   : refine only FC under SC
        sc2ec_only   : refine only EC under SC
        shared       : single shared cross-attention without SC anchoring
        concat       : naive concatenation of pooled representations
        gated        : per-modality scalar gating over pooled representations
        bidir        : bidirectional cross-attention (SC also queries FC/EC)
        reverse      : SC-as-query, FC/EC as key/value (reverse direction)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, variant: str = "sc_anchor") -> None:
        super().__init__()
        self.variant = variant
        self.fc_attn = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.ec_attn = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.fc_ln = nn.LayerNorm(d_model)
        self.ec_ln = nn.LayerNorm(d_model)
        if variant == "bidir":
            self.sc_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
            self.sc_ln = nn.LayerNorm(d_model)
        if variant == "gated":
            self.gate = nn.Linear(3 * d_model, 3)

    def forward(self, hF: torch.Tensor, hS: torch.Tensor,
                hE: torch.Tensor) -> torch.Tensor:
        if self.variant == "concat":
            return torch.cat([hF.mean(1), hS.mean(1), hE.mean(1)], dim=-1)
        if self.variant == "gated":
            pooled = torch.cat([hF.mean(1), hS.mean(1), hE.mean(1)], dim=-1)
            w = F.softmax(self.gate(pooled), dim=-1)
            stack = torch.stack([hF.mean(1), hS.mean(1), hE.mean(1)], dim=1)
            return torch.cat([(w.unsqueeze(-1) * stack).sum(1)], dim=-1)
        if self.variant == "reverse":
            # SC queries FC/EC instead of being a fixed reference
            ctx = torch.cat([hF, hE], dim=1)
            hS_new, _ = self.fc_attn(hS, ctx, ctx)
            hS_new = self.fc_ln(hS + hS_new)
            return torch.cat([hF.mean(1), hS_new.mean(1), hE.mean(1)], dim=-1)

        # sc_anchor / sc2fc_only / sc2ec_only / shared / bidir
        if self.variant in ("sc_anchor", "sc2fc_only", "shared", "bidir"):
            f_ref, _ = self.fc_attn(hF, hS, hS)
            hF = self.fc_ln(hF + f_ref)
        if self.variant in ("sc_anchor", "sc2ec_only", "shared", "bidir"):
            e_ref, _ = self.ec_attn(hE, hS, hS)
            hE = self.ec_ln(hE + e_ref)
        if self.variant == "bidir":
            ctx = torch.cat([hF, hE], dim=1)
            s_ref, _ = self.sc_attn(hS, ctx, ctx)
            hS = self.sc_ln(hS + s_ref)
        return torch.cat([hF.mean(1), hS.mean(1), hE.mean(1)], dim=-1)


# =============================================================================
# 4. Top-level MOSAIC model
# =============================================================================
class MOSAIC(nn.Module):
    """End-to-end MOSAIC pipeline: mask policy + encoders + fusion."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        R = cfg["num_rois"]
        d = cfg.get("d_model", 128)
        self.modalities = cfg.get("modalities", ["FC", "SC", "EC"])
        ablation = cfg.get("ablation", {}).get("mask_variant", None)
        self.mask_variant = ablation
        fusion_variant = cfg.get("fusion", {}).get("variant", "sc_anchor")

        # 1. modality-specific policies (one per active modality)
        self.policies = nn.ModuleDict({
            m: MaskPolicy(num_rois=R, hidden=cfg.get("policy_hidden", 128))
            for m in self.modalities
        })

        # 2. topology-matched encoders
        self.fc_enc = FCTransformer(R, d_model=d) if "FC" in self.modalities else None
        self.sc_enc = SCDHTHyperGNN(R, d_model=d) if "SC" in self.modalities else None
        self.ec_enc = ECSignedTransformer(
            R, d_model=d, signed=cfg.get("ec_signed", True),
        ) if "EC" in self.modalities else None

        # 3. SC-anchored fusion (or chosen variant)
        self.fusion = SCAnchoredFusion(d_model=d, variant=fusion_variant)

        # 4. classifier head
        self.classifier = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.get("dropout", 0.2)),
            nn.Linear(d, cfg["num_classes"]),
        )

    # -----------------------------------------------------------------
    # Mask generation
    # -----------------------------------------------------------------
    def _sample_masks(self, x: Dict[str, torch.Tensor],
                      deterministic: bool) -> Dict[str, dict]:
        out = {}
        for m in self.modalities:
            out[m] = self.policies[m](x[m], deterministic=deterministic)
        return out

    # -----------------------------------------------------------------
    # One forward pass given a fixed mask tuple
    # -----------------------------------------------------------------
    def _forward_once(self, x: Dict[str, torch.Tensor],
                      masks: Dict[str, dict]) -> torch.Tensor:
        hF = hS = hE = None
        if "FC" in self.modalities:
            hF = self.fc_enc(apply_mask(x["FC"], masks["FC"]["m"]))
        if "SC" in self.modalities:
            hS = self.sc_enc(apply_mask(x["SC"], masks["SC"]["m"]))
        if "EC" in self.modalities:
            hE = self.ec_enc(apply_mask(x["EC"], masks["EC"]["m"]))
        # placeholder zeros for missing branches keep the fusion API uniform
        if hF is None: hF = torch.zeros_like(hS if hS is not None else hE)
        if hS is None: hS = torch.zeros_like(hF)
        if hE is None: hE = torch.zeros_like(hF)
        z = self.fusion(hF, hS, hE)
        return self.classifier(z)

    # -----------------------------------------------------------------
    # Training-time forward: sample G mask tuples, build group-relative
    # advantages, and produce the auxiliary loss components consumed by
    # ``main.total_loss``.
    # -----------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor],
                x_perturbed: Optional[Dict[str, torch.Tensor]] = None,
                group_size: int = 1,
                deterministic: bool = False,
                y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # ---- Deterministic path (eval) -----------------------------------
        if deterministic or group_size <= 1:
            masks = self._sample_masks(x, deterministic=deterministic)
            logits = self._forward_once(x, masks)
            return {
                "logits": logits,
                "rl_loss": logits.new_zeros(()),
                "compactness": self._mean_retention(masks),
                "consistency": logits.new_zeros(()),
                "entropy": self._mean_entropy(masks),
                "masks": masks,
            }

        # ---- Grouped sampling (train) ------------------------------------
        rewards, logps, retentions, entropies, last_logits = [], [], [], [], None
        cons_terms = []
        for _ in range(group_size):
            masks = self._sample_masks(x, deterministic=False)
            logits = self._forward_once(x, masks)
            with torch.no_grad():
                # Reward = -CE(logits, y) (filled in by main loop via y if given)
                if y is not None:
                    reward = -F.cross_entropy(logits, y, reduction="none")
                else:
                    reward = -logits.logsumexp(-1)
            rewards.append(reward)
            logps.append(sum(masks[m]["logp"] for m in self.modalities))
            retentions.append(self._mean_retention(masks))
            entropies.append(self._mean_entropy(masks))
            last_logits = logits

            if x_perturbed is not None:
                masks_p = self._sample_masks(x_perturbed, deterministic=False)
                logits_p = self._forward_once(x_perturbed, masks_p)
                p = F.softmax(logits, dim=-1)
                q = F.softmax(logits_p, dim=-1)
                kl = 0.5 * (F.kl_div(q.log(), p, reduction="batchmean")
                            + F.kl_div(p.log(), q, reduction="batchmean"))
                dice_term = 1.0 - torch.stack([
                    self._dice(masks[m]["m"], masks_p[m]["m"])
                    for m in self.modalities
                ]).mean()
                cons_terms.append(kl + dice_term)

        R = torch.stack(rewards, dim=0)                       # (G, B)
        L = torch.stack(logps, dim=0)                         # (G, B)
        adv = (R - R.mean(0, keepdim=True)) / (R.std(0, keepdim=True) + 1e-6)
        rl_loss = -(adv.detach() * L).mean()

        return {
            "logits": last_logits,
            "rl_loss": rl_loss,
            "compactness": torch.stack(retentions).mean(),
            "consistency": (torch.stack(cons_terms).mean()
                            if cons_terms else last_logits.new_zeros(())),
            "entropy": torch.stack(entropies).mean(),
        }

    # -----------------------------------------------------------------
    # Bookkeeping helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _mean_retention(masks: Dict[str, dict]) -> torch.Tensor:
        return torch.stack([masks[m]["m"].mean() for m in masks]).mean()

    @staticmethod
    def _mean_entropy(masks: Dict[str, dict]) -> torch.Tensor:
        ents = []
        for m in masks:
            p = masks[m]["p"].clamp(1e-6, 1 - 1e-6)
            ents.append(-(p * p.log() + (1 - p) * (1 - p).log()).mean())
        return torch.stack(ents).mean()

    @staticmethod
    def _dice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        inter = (a * b).sum(-1)
        denom = a.sum(-1) + b.sum(-1) + 1e-6
        return (2 * inter / denom).mean()
