"""
bigearth_retrieval/models/dual_encoder.py
==========================================
THREE-BRANCH encoder with foundation model backbones.

  ┌───────────────────────────────────────────────────────────────────────┐
  │  Optical encoder  : DINOv2-Base (ViT-B/14, 768-D → 512-D)           │
  │                     Pretrained self-supervised on LVD-142M            │
  │                     Falls back to ResNet-50 if transformers absent    │
  │                                                                        │
  │  MS encoder       : Prithvi-100M (ViT-L, 768-D → 512-D)             │
  │                     Pretrained on HLS Sentinel-2 (NASA/IBM)           │
  │                     Falls back to ResNet-50 (8-ch) if model absent   │
  │                                                                        │
  │  SAR encoder      : ResNet-50 (conv1 patched for 2-ch)               │
  │                     No satellite SAR foundation model exists yet       │
  │                                                                        │
  │  Shared MLP projection head  512 → 2048 → 512  (BN · ReLU · L2)     │
  │  Shared classification head  512 → 19  (BCE auxiliary loss)           │
  └───────────────────────────────────────────────────────────────────────┘

Backbone selection is controlled by config.yaml:
  model:
    optical_backbone: "dinov2"     # "dinov2" | "resnet50"
    ms_backbone:      "prithvi"    # "prithvi" | "satmae" | "resnet50"
    sar_backbone:     "resnet50"   # "resnet50" | "swin_t"

Why these backbones:
  DINOv2  — self-supervised ViT, no label dependency, excellent spatial
             features for RGB, pre-tested on remote sensing downstream tasks.
  Prithvi — fine-tuned on Harmonised Landsat Sentinel-2, natively handles
             multi-band input, understands phenology and land-cover cues
             in the non-RGB bands (red-edge, SWIR).
  ResNet-50 — best available for SAR since no public SAR foundation model
              exists; SARDet and RingMo are SAR detection models, not
              retrieval embedding models.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Dict, Optional

from bigearth_retrieval.data.dataset import (
    MODALITY_OPTICAL, MODALITY_MS, MODALITY_SAR, ALL_MODALITIES
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared projection head
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    """SimCLR-v2 style 2-layer BN-MLP. Output is L2-normalised."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-50 fallback encoder (channel-adaptive)
# ─────────────────────────────────────────────────────────────────────────────
def _build_resnet50(in_channels: int, pretrained: bool) -> tuple:
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    base    = tv_models.resnet50(weights=weights)
    old_c   = base.conv1
    new_c   = nn.Conv2d(in_channels, old_c.out_channels,
                         kernel_size=old_c.kernel_size, stride=old_c.stride,
                         padding=old_c.padding, bias=False)
    with torch.no_grad():
        if in_channels <= 3:
            new_c.weight[:] = old_c.weight[:, :in_channels]
        else:
            reps   = in_channels // 3 + 1
            tiled  = old_c.weight.repeat(1, reps, 1, 1)[:, :in_channels]
            new_c.weight[:] = tiled / (in_channels / 3.0)
    base.conv1 = new_c
    base.fc    = nn.Identity()
    return base, 2048


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, pretrained: bool = True):
        super().__init__()
        backbone, feat_dim = _build_resnet50(in_channels, pretrained)
        self.backbone  = backbone
        self.adapter   = nn.Sequential(
            nn.Linear(feat_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return F.normalize(self.adapter(self.backbone(x)), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 encoder (optical — 3-ch RGB)
# ─────────────────────────────────────────────────────────────────────────────
class DINOv2Encoder(nn.Module):
    """
    facebook/dinov2-base (ViT-B/14, 768-D CLS token).
    Expects 3-ch input normalised to ImageNet mean/std.
    Frozen backbone + trainable adapter by default.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, embed_dim: int, freeze_backbone: bool = True):
        super().__init__()
        try:
            from transformers import Dinov2Model
            self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self._use_hf  = True
            print("[DINOv2Encoder] Loaded facebook/dinov2-base from HuggingFace")
        except Exception as e:
            print(f"[DINOv2Encoder] HuggingFace load failed ({e}), "
                  f"falling back to ResNet-50")
            self._fallback = ResNetEncoder(3, embed_dim, pretrained=True)
            self._use_hf   = False

        if self._use_hf:
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
            # Register ImageNet normalisation as non-trainable buffer
            self.register_buffer("inp_mean",
                torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("inp_std",
                torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
            self.adapter = nn.Sequential(
                nn.Linear(768, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
            )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) BigEarthNet-normalised float32."""
        if not self._use_hf:
            return self._fallback(x)

        # Re-normalise from BigEarthNet z-score → ImageNet z-score
        # BigEarthNet stats were already applied in dataset.py;
        # DINOv2 needs ImageNet normalisation.
        # We un-normalise first, clamp to [0,1], then apply ImageNet stats.
        # In practice the means/stds are close enough that the adapter
        # handles the residual shift — but explicit re-normalisation is cleaner.
        x_01   = torch.sigmoid(x)                   # soft map to (0,1)
        x_norm = (x_01 - self.inp_mean) / self.inp_std

        out  = self.backbone(pixel_values=x_norm)
        cls  = out.last_hidden_state[:, 0, :]       # (B, 768) CLS token
        return F.normalize(self.adapter(cls), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Prithvi encoder (multispectral — 8-ch non-RGB bands)
# ─────────────────────────────────────────────────────────────────────────────
class PrithviEncoder(nn.Module):
    """
    ibm-nasa-geospatial/Prithvi-100M — ViT-L pretrained on HLS Sentinel-2.
    Natively supports multi-band input via a configurable patch-embed layer.
    We use mean-pooling of all patch tokens as the global feature.

    Falls back to ResNet-50 (8-ch) if the model cannot be loaded.
    """

    def __init__(self, embed_dim: int, in_channels: int = 8,
                 freeze_backbone: bool = True):
        super().__init__()
        self._use_hf = False
        try:
            from transformers import AutoModel, AutoConfig
            cfg = AutoConfig.from_pretrained(
                "ibm-nasa-geospatial/Prithvi-100M",
                trust_remote_code=True,
                num_frames=1,           # single-temporal (no time series)
                in_chans=in_channels,   # 8 non-RGB bands
            )
            self.backbone = AutoModel.from_pretrained(
                "ibm-nasa-geospatial/Prithvi-100M",
                config=cfg,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            self._use_hf = True
            print(f"[PrithviEncoder] Loaded Prithvi-100M ({in_channels}-ch input)")
        except Exception as e:
            print(f"[PrithviEncoder] Load failed ({e}), "
                  f"falling back to ResNet-50 ({in_channels}-ch)")

        if self._use_hf:
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
            # Prithvi ViT-L outputs 1024-D; use 768 as conservative fallback
            feat_dim = getattr(self.backbone.config, "hidden_size", 768)
            self.adapter = nn.Sequential(
                nn.Linear(feat_dim, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
            )
        else:
            self._fallback = ResNetEncoder(in_channels, embed_dim, pretrained=True)

        self.embed_dim   = embed_dim
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, H, W)."""
        if not self._use_hf:
            return self._fallback(x)

        # Prithvi expects (B, T, C, H, W); T=1 for single-temporal
        xb = x.unsqueeze(1)                           # (B, 1, 8, H, W)
        out  = self.backbone(xb)
        # Global average over all patch tokens (skip CLS if present)
        tokens = out.last_hidden_state                  # (B, N, D)
        feat   = tokens[:, 1:, :].mean(dim=1)          # skip CLS, mean pool
        return F.normalize(self.adapter(feat), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# SatMAE encoder (alternative MS backbone)
# ─────────────────────────────────────────────────────────────────────────────
class SatMAEEncoder(nn.Module):
    """
    SatMAE (ViT-Large) pretrained on fMoW-Sentinel multi-spectral imagery.
    Handles arbitrary band counts via group-channel embedding.
    Falls back to ResNet-50 if unavailable.
    """

    def __init__(self, embed_dim: int, in_channels: int = 8,
                 freeze_backbone: bool = True):
        super().__init__()
        self._use_hf = False
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                "sustainlab-group/satmae_pretrain_fmow_sentinel",
                trust_remote_code=True,
            )
            self._use_hf = True
            print(f"[SatMAEEncoder] Loaded SatMAE ({in_channels}-ch)")
        except Exception as e:
            print(f"[SatMAEEncoder] Load failed ({e}), "
                  f"falling back to ResNet-50 ({in_channels}-ch)")

        if self._use_hf:
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
            feat_dim = getattr(self.backbone.config, "hidden_size", 1024)
            self.adapter = nn.Sequential(
                nn.Linear(feat_dim, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
            )
        else:
            self._fallback = ResNetEncoder(in_channels, embed_dim, pretrained=True)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_hf:
            return self._fallback(x)
        out    = self.backbone(x)
        feat   = out.last_hidden_state[:, 1:, :].mean(1)
        return F.normalize(self.adapter(feat), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Swin-T encoder (alternative SAR backbone — better local texture)
# ─────────────────────────────────────────────────────────────────────────────
class SwinTEncoder(nn.Module):
    """Swin-Tiny with conv1 patched to 2-ch SAR input."""
    def __init__(self, embed_dim: int, in_channels: int = 2,
                 pretrained: bool = True):
        super().__init__()
        weights = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.swin_t(weights=weights)

        # Patch the first conv (PatchMerging stem) for in_channels
        old_proj = base.features[0][0]    # Conv2d(3, 96, 4, 4)
        new_proj = nn.Conv2d(in_channels, old_proj.out_channels,
                             kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride, bias=False)
        with torch.no_grad():
            if in_channels <= 3:
                new_proj.weight[:] = old_proj.weight[:, :in_channels]
            else:
                new_proj.weight[:] = old_proj.weight.mean(1, keepdim=True).expand_as(
                    new_proj.weight)
        base.features[0][0] = new_proj
        base.head = nn.Identity()

        self.backbone = base
        # Swin-T outputs 768-D after adaptive avg pool
        self.adapter  = nn.Sequential(
            nn.Linear(768, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.backbone(x)           # (B, 768)
        return F.normalize(self.adapter(feat), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder factory
# ─────────────────────────────────────────────────────────────────────────────
def build_encoder(
    modality: str,
    backbone_name: str,
    embed_dim: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Factory that returns the right encoder for each modality + backbone combo.
    All encoders share the same interface: forward(x) → (B, embed_dim) L2-normed.
    """
    if modality == MODALITY_OPTICAL:
        if backbone_name == "dinov2":
            return DINOv2Encoder(embed_dim, freeze_backbone=freeze_backbone)
        else:  # resnet50
            return ResNetEncoder(3, embed_dim, pretrained=pretrained)

    elif modality == MODALITY_MS:
        if backbone_name == "prithvi":
            return PrithviEncoder(embed_dim, in_channels=8,
                                  freeze_backbone=freeze_backbone)
        elif backbone_name == "satmae":
            return SatMAEEncoder(embed_dim, in_channels=8,
                                 freeze_backbone=freeze_backbone)
        else:  # resnet50
            return ResNetEncoder(8, embed_dim, pretrained=pretrained)

    elif modality == MODALITY_SAR:
        if backbone_name == "swin_t":
            return SwinTEncoder(embed_dim, in_channels=2, pretrained=pretrained)
        else:  # resnet50
            return ResNetEncoder(2, embed_dim, pretrained=pretrained)

    raise ValueError(f"Unknown modality: {modality}")


# ─────────────────────────────────────────────────────────────────────────────
# TripleEncoder — main model
# ─────────────────────────────────────────────────────────────────────────────
class TripleEncoder(nn.Module):
    """
    Three modality-specific encoders (configurable backbones) sharing one
    projection head and one classification head.
    All three embed into the same 512-D L2-normalised unit sphere.
    """

    def __init__(
        self,
        embed_dim:        int   = 512,
        proj_hidden:      int   = 2048,
        pretrained:       bool  = True,
        freeze_backbone:  bool  = True,
        dropout:          float = 0.1,
        num_classes:      int   = 19,
        use_cls_head:     bool  = True,
        optical_backbone: str   = "dinov2",
        ms_backbone:      str   = "prithvi",
        sar_backbone:     str   = "resnet50",
    ):
        super().__init__()

        print(f"\n[TripleEncoder] Building encoders:")
        print(f"  optical  : {optical_backbone}  (3-ch RGB)")
        print(f"  ms       : {ms_backbone}        (8-ch non-RGB)")
        print(f"  sar      : {sar_backbone}       (2-ch VV+VH)")

        self.optical_encoder = build_encoder(MODALITY_OPTICAL, optical_backbone,
                                             embed_dim, pretrained, freeze_backbone)
        self.ms_encoder      = build_encoder(MODALITY_MS,      ms_backbone,
                                             embed_dim, pretrained, freeze_backbone)
        self.sar_encoder     = build_encoder(MODALITY_SAR,     sar_backbone,
                                             embed_dim, pretrained, freeze_backbone)

        self._encoders = {
            MODALITY_OPTICAL: self.optical_encoder,
            MODALITY_MS:      self.ms_encoder,
            MODALITY_SAR:     self.sar_encoder,
        }

        # Shared projection head (contrastive loss input)
        self.projector = ProjectionMLP(embed_dim, proj_hidden, embed_dim, dropout)

        # Shared classification head (auxiliary BCE loss)
        self.use_cls_head = use_cls_head
        if use_cls_head:
            self.cls_head = nn.Linear(embed_dim, num_classes)

        self.embed_dim = embed_dim

    def encode(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """L2-normed embedding for one modality — no projection."""
        return self._encoders[modality](x)

    def forward(
        self,
        optical:    Optional[torch.Tensor] = None,
        ms:         Optional[torch.Tensor] = None,
        sar:        Optional[torch.Tensor] = None,
        optical_v2: Optional[torch.Tensor] = None,
        ms_v2:      Optional[torch.Tensor] = None,
        sar_v2:     Optional[torch.Tensor] = None,
        return_proj: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for mod, v1, v2 in [
            (MODALITY_OPTICAL, optical, optical_v2),
            (MODALITY_MS,      ms,      ms_v2),
            (MODALITY_SAR,     sar,     sar_v2),
        ]:
            enc = self._encoders[mod]
            if v1 is not None:
                emb = enc(v1)
                out[f"{mod}_emb"] = emb
                if return_proj:
                    out[f"{mod}_proj"] = self.projector(emb)
                if self.use_cls_head:
                    out[f"{mod}_logits"] = self.cls_head(emb)
            if v2 is not None:
                emb2 = enc(v2)
                out[f"{mod}_v2_emb"]  = emb2
                if return_proj:
                    out[f"{mod}_v2_proj"] = self.projector(emb2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MomentumTripleEncoder — MoCo-v2 with per-modality queues
# ─────────────────────────────────────────────────────────────────────────────
class MomentumTripleEncoder(nn.Module):
    """
    EMA key encoder + one momentum queue per modality (4096 entries each).
    Provides far more negatives than in-batch only, especially important
    when batch size is limited by 6 GB VRAM.
    """

    def __init__(self, encoder: TripleEncoder, queue_size: int = 4096,
                 momentum: float = 0.999):
        super().__init__()
        self.encoder_q  = encoder
        self.encoder_k  = copy.deepcopy(encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad_(False)

        self.queue_size = queue_size
        self.momentum   = momentum
        D = encoder.embed_dim

        for mod in ALL_MODALITIES:
            self.register_buffer(f"{mod}_queue",
                F.normalize(torch.randn(D, queue_size), dim=0))
        self.register_buffer("label_queue", torch.zeros(queue_size, 19))
        self.register_buffer("queue_ptr",   torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self):
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data = self.momentum * pk.data + (1 - self.momentum) * pq.data

    @torch.no_grad()
    def _enqueue(self, projs: Dict, labels: torch.Tensor):
        B   = labels.shape[0]
        ptr = int(self.queue_ptr)
        end = (ptr + B) % self.queue_size
        for mod in ALL_MODALITIES:
            k = f"{mod}_proj"
            if k not in projs: continue
            q = getattr(self, f"{mod}_queue")
            if ptr + B <= self.queue_size:
                q[:, ptr:ptr + B] = projs[k].T
            else:
                f2 = self.queue_size - ptr
                q[:, ptr:] = projs[k][:f2].T
                q[:, :end] = projs[k][f2:].T
            setattr(self, f"{mod}_queue", q)
        if ptr + B <= self.queue_size:
            self.label_queue[ptr:ptr + B] = labels
        else:
            f2 = self.queue_size - ptr
            self.label_queue[ptr:] = labels[:f2]
            self.label_queue[:end] = labels[f2:]
        self.queue_ptr[0] = end

    def forward(self, optical_v1, optical_v2, ms_v1, ms_v2,
                sar_v1, sar_v2, labels) -> Dict:
        q_out = self.encoder_q(
            optical=optical_v1, ms=ms_v1, sar=sar_v1,
            optical_v2=optical_v2, ms_v2=ms_v2, sar_v2=sar_v2,
            return_proj=True,
        )
        with torch.no_grad():
            self._ema_update()
            k_out = self.encoder_k(
                optical=optical_v2, ms=ms_v2, sar=sar_v2,
                return_proj=True,
            )
        self._enqueue(k_out, labels)
        extra = {}
        for mod in ALL_MODALITIES:
            extra[f"{mod}_key"]   = k_out.get(f"{mod}_proj")
            extra[f"{mod}_queue"] = getattr(self, f"{mod}_queue").clone().detach()
        extra["label_queue"] = self.label_queue.clone().detach()
        return {**q_out, **extra}
