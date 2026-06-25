"""
bigearth_retrieval/evaluate.py
================================
Full evaluation — all 9 retrieval modes.

  Same-modal  (3): optical→optical | ms→ms | sar→sar
  Cross-modal (6): optical→ms | ms→optical | optical→sar |
                   sar→optical | ms→sar | sar→ms

Run:
    python evaluate.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/best.pt \
        [--gallery_dir outputs/gallery_test]
"""

import os, sys, json, argparse
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bigearth_retrieval.data.dataset import ALL_MODALITIES
from bigearth_retrieval.utils.faiss_index import (
    GalleryIndex, evaluate_all_modes, EVAL_MODES
)
from bigearth_retrieval.utils.embedder import embed_modality, load_model, build_gallery


def print_table(results):
    print("\n" + "═" * 84)
    print(f"  {'Mode':<22} {'F1@5':>7} {'F1@10':>7} "
          f"{'P@5':>7} {'R@5':>7} {'P@10':>7} {'R@10':>7} {'ms/q':>6}")
    print("  " + "─" * 80)

    same  = [r for r in results
             if r["mode"].split("→")[0] == r["mode"].split("→")[1]]
    cross = [r for r in results
             if r["mode"].split("→")[0] != r["mode"].split("→")[1]]

    def row(r, prefix=""):
        print(f"  {r['mode']:<22} "
              f"{r.get('F1@5',0):>7.4f} {r.get('F1@10',0):>7.4f} "
              f"{r.get('P@5',0):>7.4f} {r.get('R@5',0):>7.4f} "
              f"{r.get('P@10',0):>7.4f} {r.get('R@10',0):>7.4f} "
              f"{r.get('avg_query_time_ms',0):>6.2f}")

    print("  ── Same-modal ──────────────────────────────────────────────────────")
    for r in same:  row(r)
    print("  ── Cross-modal ─────────────────────────────────────────────────────")
    for r in cross: row(r)
    print("═" * 84)

    mean_same  = np.mean([r.get("F1@10", 0) for r in same])
    mean_cross = np.mean([r.get("F1@10", 0) for r in cross])
    mean_time  = np.mean([r.get("avg_query_time_ms", 0) for r in results])
    print(f"\n  Mean same-modal  F1@10 : {mean_same:.4f}")
    print(f"  Mean cross-modal F1@10 : {mean_cross:.4f}")
    print(f"  Avg query time         : {mean_time:.2f} ms\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--split",       default="test")
    ap.add_argument("--gallery_dir", default=None)
    ap.add_argument("--out",         default="outputs/eval_results.json")
    ap.add_argument("--max_query",   type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device} | Split: {args.split}")
    model = load_model(cfg, args.checkpoint, device)

    if args.gallery_dir and (Path(args.gallery_dir) / "faiss.index").exists():
        print(f"[Eval] Loading pre-built gallery: {args.gallery_dir}")
        gallery = GalleryIndex.load(args.gallery_dir,
                                    use_gpu=cfg["faiss"]["use_gpu"])
        emb_d, lbl_d = {}, {}
        for mod in ALL_MODALITIES:
            d = embed_modality(model, cfg["paths"]["metadata"],
                               cfg["paths"]["s2_root"], cfg["paths"]["s1_root"],
                               args.split, mod,
                               cfg["data"]["image_size"],
                               cfg["data"]["clip_percentile"],
                               device, max_samples=args.max_query)
            emb_d[mod] = d["embeddings"]
            lbl_d[mod] = d["labels"]
    else:
        gallery, emb_d, lbl_d = build_gallery(
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
            save_dir=args.gallery_dir,
            max_samples=args.max_query,
        )

    results = evaluate_all_modes(
        gallery, emb_d, lbl_d,
        Ks=[5, 10],
        threshold=cfg["eval"]["relevance_threshold"],
    )

    print_table(results)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved → {args.out}")


if __name__ == "__main__":
    main()
