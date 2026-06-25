"""
bigearth_retrieval/data/dataset.py
===================================
THREE-MODALITY BigEarthNet-v2 dataset — improved band split.

  ┌──────────────────────────────────────────────────────────────────────┐
  │  OPTICAL       = S2  B02 B03 B04              (3-ch  RGB only)      │
  │  MULTISPECTRAL = S2  B05 B06 B07 B08 B8A B09 B11 B12  (8-ch no RGB)│
  │  SAR           = S1  VV VH                    (2-ch  backscatter)   │
  └──────────────────────────────────────────────────────────────────────┘

Key change from v1: MS bands no longer include B02/B03/B04.
Each branch now carries UNIQUE spectral information:
  Optical  → human-visible colour, texture (3-ch)
  MS       → red-edge, NIR, SWIR — phenology, moisture, soil (8-ch)
  SAR      → structure, roughness, all-weather (2-ch)

This removes the redundant subset relationship (optical ⊂ ms) and forces
each encoder to specialise on genuinely different physical signals.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
from typing import Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Modality constants
# ─────────────────────────────────────────────────────────────────────────────
MODALITY_OPTICAL = "optical"
MODALITY_MS      = "ms"
MODALITY_SAR     = "sar"
ALL_MODALITIES   = [MODALITY_OPTICAL, MODALITY_MS, MODALITY_SAR]

# ── Band definitions (NON-OVERLAPPING) ───────────────────────────────────────
OPTICAL_BANDS = ["B02", "B03", "B04"]                    # 3-ch RGB @ 10m
MS_BANDS      = ["B05", "B06", "B07", "B08",            # 8-ch red-edge + SWIR
                  "B8A", "B09", "B11", "B12"]            # (no RGB bands)
SAR_POLS      = ["VV", "VH"]                             # 2-ch dual-pol SAR

# Channel counts per modality
CH = {MODALITY_OPTICAL: 3, MODALITY_MS: 8, MODALITY_SAR: 2}

# ── 19 canonical BigEarthNet label classes ────────────────────────────────────
LABELS_19 = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
LABEL2IDX   = {l: i for i, l in enumerate(LABELS_19)}
NUM_CLASSES = len(LABELS_19)

# ── Per-channel normalisation stats (BigEarthNet training split) ──────────────
# Optical: B02 B03 B04
OPT_MEAN = np.array([429.94, 614.21, 590.24], dtype=np.float32)
OPT_STD  = np.array([572.42, 582.30, 675.88], dtype=np.float32)

# Multispectral: B05 B06 B07 B08 B8A B09 B11 B12  (8 bands, RGB excluded)
MS_MEAN  = np.array([950.68, 1792.20, 2086.47, 2218.94,
                     2396.00, 2512.37, 1828.85, 1241.68], dtype=np.float32)
MS_STD   = np.array([729.89,  841.17,  938.39, 1344.90,
                     1124.43, 1175.74,  993.10,  821.74], dtype=np.float32)

# SAR: VV VH (dB scale)
SAR_MEAN = np.array([-12.619, -19.951], dtype=np.float32)
SAR_STD  = np.array([  5.262,   5.515], dtype=np.float32)

NORM_STATS = {
    MODALITY_OPTICAL: (OPT_MEAN, OPT_STD),
    MODALITY_MS:      (MS_MEAN,  MS_STD),
    MODALITY_SAR:     (SAR_MEAN, SAR_STD),
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as f:
        return f.read(1).astype(np.float32)

def _s2_tile(patch_id: str) -> str:
    return "_".join(patch_id.split("_")[:-2])

def _s1_tile(s1_name: str) -> str:
    return "_".join(s1_name.split("_")[:-2])

def _clip(arr: np.ndarray, pct: float) -> np.ndarray:
    lo, hi = np.percentile(arr, [pct, 100 - pct])
    return np.clip(arr, lo, hi)


def load_optical(s2_root: str, patch_id: str, clip_pct: float = 2.0) -> np.ndarray:
    """(H, W, 3) float32 — B02 B03 B04 only."""
    patch_dir = Path(s2_root) / _s2_tile(patch_id) / patch_id
    return np.stack(
        [_clip(read_tif(str(patch_dir / f"{patch_id}_{b}.tif")), clip_pct)
         for b in OPTICAL_BANDS], axis=-1
    ).astype(np.float32)


def load_ms(s2_root: str, patch_id: str, clip_pct: float = 2.0) -> np.ndarray:
    """(H, W, 8) float32 — B05 B06 B07 B08 B8A B09 B11 B12 (no RGB)."""
    patch_dir = Path(s2_root) / _s2_tile(patch_id) / patch_id
    return np.stack(
        [_clip(read_tif(str(patch_dir / f"{patch_id}_{b}.tif")), clip_pct)
         for b in MS_BANDS], axis=-1
    ).astype(np.float32)


def load_sar(s1_root: str, s1_name: str, clip_pct: float = 2.0) -> np.ndarray:
    """(H, W, 2) float32 — VV VH."""
    patch_dir = Path(s1_root) / _s1_tile(s1_name) / s1_name
    return np.stack(
        [_clip(read_tif(str(patch_dir / f"{s1_name}_{p}.tif")), clip_pct)
         for p in SAR_POLS], axis=-1
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────
class ChannelNorm:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean[None, None, :]
        self.std  = std[None, None, :]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return ((img - self.mean) / (self.std + 1e-6)).astype(np.float32)

NORMALIZERS = {mod: ChannelNorm(*NORM_STATS[mod]) for mod in ALL_MODALITIES}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _train_aug(sz: int) -> A.Compose:
    return A.Compose([
        A.RandomCrop(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=30, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.3),
        ToTensorV2(),
    ])

def _val_aug(sz: int) -> A.Compose:
    return A.Compose([A.CenterCrop(sz, sz), ToTensorV2()])


# ─────────────────────────────────────────────────────────────────────────────
# Label utility
# ─────────────────────────────────────────────────────────────────────────────
def labels_to_vec(labels) -> np.ndarray:
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for l in labels:
        if l in LABEL2IDX:
            vec[LABEL2IDX[l]] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# TripleModalDataset  (training)
# ─────────────────────────────────────────────────────────────────────────────
class TripleModalDataset(Dataset):
    """
    All 3 modalities per patch, each with 2 independent augmented views.
    optical: (3-ch), ms: (8-ch), sar: (2-ch)
    """

    def __init__(
        self,
        metadata_path: str,
        s2_root: str,
        s1_root: str,
        split: str = "train",
        image_size: int = 112,
        clip_pct: float = 2.0,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.s2_root   = s2_root
        self.s1_root   = s1_root
        self.clip_pct  = clip_pct
        self.augment   = augment

        df = pd.read_parquet(metadata_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)),
                           random_state=42).reset_index(drop=True)
        self.df = df

        if augment:
            self._aug1 = _train_aug(image_size)
            self._aug2 = _train_aug(image_size)
        else:
            self._aug = _val_aug(image_size)

        print(f"[TripleModalDataset] split={split} | N={len(df):,} | "
              f"optical=3ch ms=8ch sar=2ch (non-overlapping)")

    def __len__(self): return len(self.df)

    def _t(self, img, aug): return aug(image=img)["image"].float()

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        pid, s1n = row["patch_id"], row["s1_name"]
        lv = labels_to_vec(list(row["labels"]))

        opt = NORMALIZERS[MODALITY_OPTICAL](load_optical(self.s2_root, pid, self.clip_pct))
        ms  = NORMALIZERS[MODALITY_MS]     (load_ms    (self.s2_root, pid, self.clip_pct))
        sar = NORMALIZERS[MODALITY_SAR]    (load_sar   (self.s1_root, s1n, self.clip_pct))

        if self.augment:
            return {
                "optical_v1": self._t(opt, self._aug1),
                "optical_v2": self._t(opt, self._aug2),
                "ms_v1":      self._t(ms,  self._aug1),
                "ms_v2":      self._t(ms,  self._aug2),
                "sar_v1":     self._t(sar, self._aug1),
                "sar_v2":     self._t(sar, self._aug2),
                "labels":     torch.from_numpy(lv),
                "patch_id":   pid, "s1_name": s1n, "idx": idx,
            }
        return {
            "optical": self._t(opt, self._aug),
            "ms":      self._t(ms,  self._aug),
            "sar":     self._t(sar, self._aug),
            "labels":  torch.from_numpy(lv),
            "patch_id": pid, "s1_name": s1n, "idx": idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SingleModalDataset  (gallery / query embedding)
# ─────────────────────────────────────────────────────────────────────────────
class SingleModalDataset(Dataset):
    """Single modality for gallery building or query embedding at inference."""

    def __init__(
        self,
        metadata_path: str,
        s2_root: str,
        s1_root: str,
        split: str = "test",
        modality: str = MODALITY_OPTICAL,
        image_size: int = 112,
        clip_pct: float = 2.0,
        max_samples: Optional[int] = None,
    ):
        assert modality in ALL_MODALITIES
        self.modality = modality
        self.s2_root  = s2_root
        self.s1_root  = s1_root
        self.clip_pct = clip_pct
        self.norm     = NORMALIZERS[modality]
        self.aug      = _val_aug(image_size)

        df = pd.read_parquet(metadata_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)),
                           random_state=42).reset_index(drop=True)
        self.df = df
        print(f"[SingleModalDataset] {modality} | {split} | N={len(df):,} | "
              f"ch={CH[modality]}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        pid, s1n = row["patch_id"], row["s1_name"]

        if   self.modality == MODALITY_OPTICAL: raw = load_optical(self.s2_root, pid, self.clip_pct)
        elif self.modality == MODALITY_MS:      raw = load_ms(self.s2_root, pid, self.clip_pct)
        else:                                   raw = load_sar(self.s1_root, s1n, self.clip_pct)

        return {
            "image":    self.aug(image=self.norm(raw))["image"].float(),
            "labels":   torch.from_numpy(labels_to_vec(list(row["labels"]))),
            "patch_id": pid, "s1_name": s1n,
            "modality": self.modality, "idx": idx,
        }
