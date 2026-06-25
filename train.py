"""
bigearth_retrieval/train.py
============================
Training loop — now reads backbone config from config.yaml.

Run:
    python train.py --config configs/config.yaml [--resume checkpoints/last.pt]

Key changes from v2:
  • TripleEncoder reads optical_backbone / ms_backbone / sar_backbone from cfg
  • MS encoder now receives 8-ch (non-RGB) instead of 12-ch
  • image_size defaults to 224 to suit ViT-based foundation models
  • Separate (lower) LR for backbone layers vs adapter + head layers
"""

import os, sys, time, argparse, csv, random, math, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bigearth_retrieval.data.dataset import (
    TripleModalDataset, SingleModalDataset,
    ALL_MODALITIES, NUM_CLASSES, LABELS_19, LABEL2IDX,
)
from bigearth_retrieval.models.dual_encoder import (
    TripleEncoder, MomentumTripleEncoder
)
from bigearth_retrieval.models.losses import TripleModalLoss, compute_pos_weights
from bigearth_retrieval.utils.faiss_index import GalleryIndex, evaluate_all_modes
from bigearth_retrieval.utils.embedder import embed_modality


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class WarmupCosine:
    def __init__(self, opt, warmup, total, base_lr):
        self.opt = opt; self.warmup = warmup
        self.total = total; self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            p  = (epoch - self.warmup) / max(1, self.total - self.warmup)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * p))
        for pg in self.opt.param_groups:
            # backbone params get 10× lower LR
            scale = 0.1 if pg.get("is_backbone", False) else 1.0
            pg["lr"] = lr * scale
        return lr


def _param_groups(model: TripleEncoder, base_lr: float):
    """Separate param groups: backbone (low LR) vs adapter+head (normal LR)."""
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adapter" in name or "projector" in name or "cls_head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": head_params,     "is_backbone": False},
        {"params": backbone_params, "is_backbone": True},
    ]


@torch.no_grad()
def quick_val(model, cfg, device, max_val=5000):
    model.eval()
    meta = cfg["paths"]["metadata"]
    s2r  = cfg["paths"]["s2_root"]
    s1r  = cfg["paths"]["s1_root"]
    isz  = cfg["data"]["image_size"]
    cpct = cfg["data"]["clip_percentile"]

    emb_d, lbl_d = {}, {}
    for mod in ALL_MODALITIES:
        d = embed_modality(model, meta, s2r, s1r, "validation",
                           mod, isz, cpct, device,
                           batch_size=128, num_workers=4,
                           max_samples=max_val)
        emb_d[mod] = d["embeddings"]
        lbl_d[mod] = d["labels"]

    dim     = emb_d[ALL_MODALITIES[0]].shape[1]
    all_emb = np.concatenate([emb_d[m] for m in ALL_MODALITIES])
    gallery = GalleryIndex(dim=dim, index_type="Flat", use_gpu=False,
                           nlist=256, nprobe=16)
    gallery.train_if_needed(all_emb)
    for mod in ALL_MODALITIES:
        d2 = embed_modality(model, meta, s2r, s1r, "validation",
                            mod, isz, cpct, device,
                            batch_size=128, num_workers=4,
                            max_samples=max_val)
        gallery.add_batch(d2["embeddings"], [mod]*len(d2["patch_ids"]),
                          d2["labels"], d2["patch_ids"], d2["s1_names"])
    gallery.finalize()

    results = evaluate_all_modes(gallery, emb_d, lbl_d, Ks=[5, 10])
    print(f"\n  {'Mode':<22} {'F1@5':>7} {'F1@10':>7} {'ms/q':>7}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['mode']:<22} {r['F1@5']:>7.4f} {r['F1@10']:>7.4f} "
              f"{r['avg_query_time_ms']:>7.2f}")
    model.train()
    return results


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, accum_steps, epoch):
    model.train()
    run = {"total": 0., "cross": 0., "same": 0., "cls": 0.}
    nb  = len(loader)
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        to = lambda t: t.to(device, non_blocking=True)
        with autocast():
            out = model(
                to(batch["optical_v1"]), to(batch["optical_v2"]),
                to(batch["ms_v1"]),      to(batch["ms_v2"]),
                to(batch["sar_v1"]),     to(batch["sar_v2"]),
                to(batch["labels"]),
            )
            ld   = criterion(out, to(batch["labels"]))
            loss = ld["total"] / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == nb:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.encoder_q.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k in run:
            if k in ld: run[k] += ld[k].item()

        if (step + 1) % 50 == 0:
            print(f"  Ep{epoch:02d} [{step+1:4d}/{nb}]  "
                  f"L={run['total']/(step+1):.4f}  "
                  f"cross={run['cross']/(step+1):.4f}  "
                  f"same={run['same']/(step+1):.4f}  "
                  f"cls={run['cls']/(step+1):.4f}")

    return {k: v / nb for k, v in run.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["output_dir"],     exist_ok=True)

    train_ds = TripleModalDataset(
        metadata_path=cfg["paths"]["metadata"],
        s2_root=cfg["paths"]["s2_root"],
        s1_root=cfg["paths"]["s1_root"],
        split="train",
        image_size=cfg["data"]["image_size"],
        clip_pct=cfg["data"]["clip_percentile"],
        augment=True,
    )
    loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, drop_last=True, persistent_workers=True,
    )

    # Build TripleEncoder with configured backbones
    encoder = TripleEncoder(
        embed_dim=cfg["model"]["embed_dim"],
        proj_hidden=cfg["model"]["projection_hidden"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
        num_classes=NUM_CLASSES,
        use_cls_head=True,
        optical_backbone=cfg["model"]["optical_backbone"],
        ms_backbone=cfg["model"]["ms_backbone"],
        sar_backbone=cfg["model"]["sar_backbone"],
    )
    model = MomentumTripleEncoder(
        encoder, queue_size=cfg["training"]["loss"]["queue_size"],
    ).to(device)

    # Class-balanced BCE weights
    print("[Train] Computing label class weights…")
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for row in train_ds.df["labels"]:
        for l in row:
            if l in LABEL2IDX: counts[LABEL2IDX[l]] += 1
    pos_wt = compute_pos_weights(
        torch.from_numpy(counts).float(), len(train_ds)
    ).to(device)

    criterion = TripleModalLoss(
        temperature=cfg["training"]["loss"]["temperature"],
        lambda_cross=cfg["training"]["loss"]["lambda_cross"],
        lambda_same=cfg["training"]["loss"]["lambda_same"],
        lambda_cls=cfg["training"]["loss"]["lambda_cls"],
        hard_neg_weight=cfg["training"]["loss"]["hard_neg_weight"],
        use_hard_neg=cfg["training"]["loss"]["use_hard_negatives"],
        pos_weight=pos_wt,
    )

    # Param groups: backbone at 0.1× LR, adapter+head at 1×
    pg    = _param_groups(model.encoder_q, cfg["training"]["lr"])
    optimizer = optim.AdamW(pg, lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    scaler    = GradScaler(enabled=cfg["training"]["amp"])
    scheduler = WarmupCosine(optimizer,
                             cfg["training"]["warmup_epochs"],
                             cfg["training"]["epochs"],
                             cfg["training"]["lr"])

    start_ep, best_f1 = 0, 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["epoch"]; best_f1 = ckpt.get("best_f1", 0.0)
        print(f"[Train] Resumed ep={start_ep}, best_f1={best_f1:.4f}")

    log_path = Path(cfg["paths"]["output_dir"]) / "train_log.csv"
    fields   = ["epoch","lr","loss_total","loss_cross","loss_same","loss_cls",
                 "mean_cross_F1@10","mean_same_F1@10"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    opt_bb  = cfg["model"]["optical_backbone"]
    ms_bb   = cfg["model"]["ms_backbone"]
    sar_bb  = cfg["model"]["sar_backbone"]
    print(f"\n{'='*65}")
    print(f"  Backbones: optical={opt_bb} | ms={ms_bb} | sar={sar_bb}")
    print(f"  Bands:  optical=B02-B04(3ch) | ms=B05-B12(8ch) | sar=VV+VH(2ch)")
    print(f"  Epochs {start_ep}→{cfg['training']['epochs']} | "
          f"Batch {cfg['training']['batch_size']}×{cfg['training']['accum_steps']}")
    print(f"{'='*65}\n")

    for epoch in range(start_ep, cfg["training"]["epochs"]):
        lr = scheduler.step(epoch)
        t0 = time.time()
        tm = train_one_epoch(model, loader, criterion, optimizer,
                             scaler, device,
                             cfg["training"]["accum_steps"], epoch)
        print(f"\nEpoch {epoch:02d} | {(time.time()-t0)/60:.1f}min | "
              f"LR={lr:.2e} | Loss={tm['total']:.4f}")

        val_results = []
        if epoch % 2 == 0 or epoch == cfg["training"]["epochs"] - 1:
            print("  [Val]")
            val_results = quick_val(model.encoder_q, cfg, device)

        state = {
            "epoch": epoch + 1, "best_f1": best_f1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }
        ckpt_dir = cfg["paths"]["checkpoint_dir"]
        torch.save(state, f"{ckpt_dir}/last.pt")

        cross_f1s = [r["F1@10"] for r in val_results
                     if r["mode"].split("→")[0] != r["mode"].split("→")[1]]
        mean_cross = float(np.mean(cross_f1s)) if cross_f1s else 0.0
        same_f1s   = [r["F1@10"] for r in val_results
                      if r["mode"].split("→")[0] == r["mode"].split("→")[1]]
        mean_same  = float(np.mean(same_f1s)) if same_f1s else 0.0

        if mean_cross > best_f1:
            best_f1 = mean_cross
            torch.save(state, f"{ckpt_dir}/best.pt")
            print(f"  ★ New best mean cross-modal F1@10: {best_f1:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                "epoch": epoch, "lr": f"{lr:.2e}",
                "loss_total": f"{tm['total']:.4f}",
                "loss_cross": f"{tm.get('cross',0):.4f}",
                "loss_same":  f"{tm.get('same',0):.4f}",
                "loss_cls":   f"{tm.get('cls',0):.4f}",
                "mean_cross_F1@10": f"{mean_cross:.4f}",
                "mean_same_F1@10":  f"{mean_same:.4f}",
            })

    print(f"\n✓ Done. Best mean cross-modal F1@10: {best_f1:.4f}")


if __name__ == "__main__":
    main()
