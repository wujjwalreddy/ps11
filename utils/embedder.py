"""
bigearth_retrieval/utils/embedder.py
======================================
Batch-embeds all three modalities and builds the FAISS gallery.
Updated for v3: reads backbone config from checkpoint or config.yaml.

Usage:
    python -m utils.embedder \
        --config configs/config.yaml \
        --checkpoint checkpoints/best.pt \
        --split test \
        --out_dir outputs/gallery_test
"""

import os, sys, argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bigearth_retrieval.data.dataset import SingleModalDataset, ALL_MODALITIES
from bigearth_retrieval.models.dual_encoder import TripleEncoder
from bigearth_retrieval.utils.faiss_index import GalleryIndex


@torch.no_grad()
def embed_modality(
    model:         TripleEncoder,
    metadata_path: str,
    s2_root:       str,
    s1_root:       str,
    split:         str,
    modality:      str,
    image_size:    int,
    clip_pct:      float,
    device:        torch.device,
    batch_size:    int = 128,
    num_workers:   int = 4,
    max_samples:   Optional[int] = None,
) -> Dict:
    ds = SingleModalDataset(
        metadata_path=metadata_path,
        s2_root=s2_root, s1_root=s1_root,
        split=split, modality=modality,
        image_size=image_size, clip_pct=clip_pct,
        max_samples=max_samples,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model.eval()
    embs, lbls, pids, s1ns = [], [], [], []

    for batch in tqdm(loader, desc=f"  Embed {modality.upper():>7}"):
        imgs = batch["image"].to(device, non_blocking=True)
        with autocast():
            emb = model.encode(modality, imgs)
        embs.append(emb.float().cpu().numpy())
        lbls.append(batch["labels"].numpy())
        pids.extend(batch["patch_id"])
        s1ns.extend(batch["s1_name"])

    return {
        "embeddings": np.concatenate(embs, 0),
        "labels":     np.concatenate(lbls, 0),
        "patch_ids":  pids,
        "s1_names":   s1ns,
        "modalities": [modality] * len(pids),
    }


def build_gallery(
    model:       TripleEncoder,
    metadata_path: str,
    s2_root:     str,
    s1_root:     str,
    split:       str,
    image_size:  int,
    clip_pct:    float,
    index_type:  str,
    nlist:       int,
    nprobe:      int,
    use_gpu:     bool,
    device:      torch.device,
    save_dir:    Optional[str] = None,
    batch_size:  int = 128,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> tuple:
    print(f"\n[Embedder] Embedding split='{split}' — 3 modalities")
    data = {}
    for mod in ALL_MODALITIES:
        data[mod] = embed_modality(
            model, metadata_path, s2_root, s1_root,
            split, mod, image_size, clip_pct, device,
            batch_size, num_workers, max_samples,
        )

    dim     = data[ALL_MODALITIES[0]]["embeddings"].shape[1]
    all_emb = np.concatenate([data[m]["embeddings"] for m in ALL_MODALITIES])

    gallery = GalleryIndex(dim=dim, index_type=index_type,
                           nlist=nlist, nprobe=nprobe, use_gpu=use_gpu)
    gallery.train_if_needed(all_emb)
    for mod in ALL_MODALITIES:
        d = data[mod]
        gallery.add_batch(d["embeddings"], d["modalities"], d["labels"],
                          d["patch_ids"], d["s1_names"])
    gallery.finalize()

    if save_dir:
        gallery.save(save_dir)

    return (
        gallery,
        {m: data[m]["embeddings"] for m in ALL_MODALITIES},
        {m: data[m]["labels"]     for m in ALL_MODALITIES},
    )


def load_model(cfg: dict, ckpt_path: str, device: torch.device) -> TripleEncoder:
    """Load TripleEncoder from checkpoint, respecting backbone config."""
    # Prefer backbone config stored inside the checkpoint (saved by train.py)
    ckpt      = torch.load(ckpt_path, map_location=device)
    ckpt_cfg  = ckpt.get("config", {})
    model_cfg = ckpt_cfg.get("model", cfg.get("model", {}))

    model = TripleEncoder(
        embed_dim=model_cfg.get("embed_dim", cfg["model"]["embed_dim"]),
        proj_hidden=model_cfg.get("projection_hidden",
                                  cfg["model"]["projection_hidden"]),
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
        use_cls_head=False,
        optical_backbone=model_cfg.get("optical_backbone",
                                       cfg["model"].get("optical_backbone","resnet50")),
        ms_backbone=model_cfg.get("ms_backbone",
                                  cfg["model"].get("ms_backbone","resnet50")),
        sar_backbone=model_cfg.get("sar_backbone",
                                   cfg["model"].get("sar_backbone","resnet50")),
    ).to(device)

    state   = ckpt.get("model", ckpt)
    cleaned = {k.replace("encoder_q.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print(f"[Embedder] Loaded checkpoint: {ckpt_path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split",      default="test")
    ap.add_argument("--out_dir",    default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(cfg, args.checkpoint, device)
    out_dir = args.out_dir or str(
        Path(cfg["paths"]["index_dir"]) / f"{args.split}_gallery"
    )

    build_gallery(
        model,
        cfg["paths"]["metadata"],
        cfg["paths"]["s2_root"],
        cfg["paths"]["s1_root"],
        args.split,
        cfg["data"]["image_size"],
        cfg["data"]["clip_percentile"],
        cfg["faiss"]["index_type"],
        cfg["faiss"]["nlist"],
        cfg["faiss"]["nprobe"],
        cfg["faiss"]["use_gpu"],
        device,
        save_dir=out_dir,
        batch_size=args.batch_size,
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"\n[Embedder] Done → {out_dir}")


if __name__ == "__main__":
    main()
