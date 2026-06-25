"""
bigearth_retrieval/models/losses.py
=====================================
Loss for three-modality cross-modal retrieval.

Total loss = Σ over all modality PAIRS of λ * NT-Xent(A, B)
           + λ_same * Σ over each modality of NT-Xent(v1, v2)
           + λ_cls  * Σ over each modality of BCE(logits, labels)

Modality pairs covered during training (6 cross-modal + 3 same-modal):
  Cross: optical↔ms, optical↔sar, ms↔sar
  Same:  optical v1↔v2,  ms v1↔v2,  sar v1↔v2

This lets the model learn a single embedding space where all 7 problem
retrieval modes (and the 2 bonus SAR↔MS modes) are well-aligned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from bigearth_retrieval.data.dataset import MODALITY_OPTICAL, MODALITY_MS, MODALITY_SAR, ALL_MODALITIES

# All unordered cross-modal pairs
CROSS_PAIRS = [
    (MODALITY_OPTICAL, MODALITY_MS),
    (MODALITY_OPTICAL, MODALITY_SAR),
    (MODALITY_MS,      MODALITY_SAR),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core NT-Xent with Jaccard hard-negative reweighting
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """
    la: (B, C)  lb: (N, C)  →  (B, N) Jaccard similarity matrix.
    Uses binary multi-hot vectors.
    """
    inter = la.float() @ lb.float().T                        # (B, N)
    a_sum = la.float().sum(-1, keepdim=True)                 # (B, 1)
    b_sum = lb.float().sum(-1, keepdim=True).T               # (1, N)
    union = (a_sum + b_sum - inter).clamp(min=1.0)
    return inter / union                                     # (B, N) ∈ [0,1]


def nt_xent(
    proj_a:    torch.Tensor,            # (B, D) L2-normed query projections
    proj_b:    torch.Tensor,            # (N, D) L2-normed key  projections
    temperature: float = 0.07,
    label_a:   Optional[torch.Tensor] = None,  # (B, 19)
    label_b:   Optional[torch.Tensor] = None,  # (N, 19)
    hard_neg_w: float = 2.0,
    symmetric:  bool  = True,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss.
    Positives are the diagonal when B==N (same-index pairs).
    Hard negative reweighting: logits for same-label pairs are SCALED DOWN
    so the model is not harshly penalised for keeping semantically similar
    images close.
    """
    B, N   = proj_a.shape[0], proj_b.shape[0]
    device = proj_a.device

    logits = (proj_a @ proj_b.T) / temperature     # (B, N) cosine-sim / T

    # ── Hard-negative reweighting mask ────────────────────────────────────────
    if label_a is not None and label_b is not None:
        overlap   = _jaccard(label_a, label_b)      # (B, N)
        # Entries with overlap get logit scaled DOWN (soft negatives)
        # Entries with no overlap get logit scaled UP (hard negatives)
        # Diagonal (positives) will be excluded from scaling below
        scale = 1.0 + (hard_neg_w - 1.0) * (1.0 - overlap)
        if B == N:
            # Protect positives: set their scale to 1
            eye = torch.eye(B, device=device)
            scale = scale * (1 - eye) + eye
        logits = logits * scale

    if B == N:
        # Standard in-batch NT-Xent (symmetric)
        log_pos   = logits.diag()                              # (B,)
        log_denom = torch.logsumexp(logits, dim=1)             # (B,)
        loss_fwd  = -(log_pos - log_denom).mean()

        if symmetric:
            log_denom2 = torch.logsumexp(logits.T, dim=1)
            loss_bwd   = -(log_pos - log_denom2).mean()
            return (loss_fwd + loss_bwd) / 2.0
        return loss_fwd
    else:
        # Queue-based: rows are queries, no guaranteed diagonal positive
        # (positives are marked by pos_mask in queue usage — simplified here)
        log_denom = torch.logsumexp(logits, dim=1)
        # Take max logit as pseudo-positive (simplification)
        loss = (-logits.max(dim=1).values + log_denom).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# TripleModalLoss
# ─────────────────────────────────────────────────────────────────────────────

class TripleModalLoss(nn.Module):
    """
    Combined loss for three-modality cross-modal retrieval.

    L_total = λ_cross * mean(cross-modal NT-Xent over all 3 pairs)
            + λ_same  * mean(same-modal  NT-Xent over all 3 modalities)
            + λ_cls   * mean(BCE cls loss over all 3 modalities)
    """

    def __init__(
        self,
        temperature:    float = 0.07,
        lambda_cross:   float = 1.0,
        lambda_same:    float = 0.5,
        lambda_cls:     float = 0.3,
        hard_neg_weight: float = 2.0,
        use_hard_neg:   bool  = True,
        pos_weight:     Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.temperature   = temperature
        self.lambda_cross  = lambda_cross
        self.lambda_same   = lambda_same
        self.lambda_cls    = lambda_cls
        self.hard_neg_w    = hard_neg_weight
        self.use_hard_neg  = use_hard_neg
        self.bce           = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        labels:    torch.Tensor,              # (B, 19)
    ) -> Dict[str, torch.Tensor]:

        losses: Dict[str, torch.Tensor] = {}
        total  = torch.zeros(1, device=labels.device).squeeze()

        lbl_a = labels if self.use_hard_neg else None

        # ── Cross-modal NT-Xent (3 pairs, symmetric each) ────────────────────
        cross_vals = []
        for mod_a, mod_b in CROSS_PAIRS:
            ka = f"{mod_a}_proj"
            kb = f"{mod_b}_proj"
            if ka not in model_out or kb not in model_out:
                continue
            l = nt_xent(
                model_out[ka], model_out[kb],
                self.temperature,
                lbl_a, lbl_a,
                self.hard_neg_w, symmetric=True,
            )
            losses[f"cross_{mod_a}_{mod_b}"] = l
            cross_vals.append(l)

        if cross_vals:
            l_cross = torch.stack(cross_vals).mean()
            losses["cross"] = l_cross
            total = total + self.lambda_cross * l_cross

        # ── Same-modal NT-Xent (v1 ↔ v2 for each modality) ───────────────────
        same_vals = []
        for mod in ALL_MODALITIES:
            k1 = f"{mod}_proj"
            k2 = f"{mod}_v2_proj"
            if k1 not in model_out or k2 not in model_out:
                continue
            l = nt_xent(
                model_out[k1], model_out[k2],
                self.temperature,
                lbl_a, lbl_a,
                self.hard_neg_w, symmetric=True,
            )
            losses[f"same_{mod}"] = l
            same_vals.append(l)

        if same_vals:
            l_same = torch.stack(same_vals).mean()
            losses["same"] = l_same
            total = total + self.lambda_same * l_same

        # ── Classification auxiliary loss (each modality) ─────────────────────
        cls_vals = []
        for mod in ALL_MODALITIES:
            lk = f"{mod}_logits"
            if lk not in model_out:
                continue
            l = self.bce(model_out[lk], labels)
            losses[f"cls_{mod}"] = l
            cls_vals.append(l)

        if cls_vals:
            l_cls = torch.stack(cls_vals).mean()
            losses["cls"] = l_cls
            total = total + self.lambda_cls * l_cls

        losses["total"] = total
        return losses


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute per-class BCE pos_weight from label counts
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weights(label_counts: torch.Tensor, total: int) -> torch.Tensor:
    """pos_weight_i = (total - count_i) / count_i, capped at 10."""
    return ((total - label_counts) / label_counts.clamp(min=1.0)).clamp(max=10.0)
