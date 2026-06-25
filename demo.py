"""
bigearth_retrieval/demo.py
===========================
Streamlit web demo for interactive cross-modal satellite image retrieval.

Run:
    streamlit run demo.py -- \
        --config configs/config.yaml \
        --checkpoint checkpoints/best.pt \
        --gallery_dir outputs/gallery_test

Features
────────
• Upload a GeoTIFF patch (S2 or S1) OR pick a random test patch
• Select query modality and target gallery modality
• Display top-5 / top-10 retrieved images with similarity scores
• Show RGB composites (B04/B03/B02 for S2; VV for S1)
• Report retrieval time and ground-truth labels
"""

import sys
import time
import argparse
import numpy as np
import torch
import yaml
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import io
import PIL.Image

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import (
    load_s2_patch, load_s1_patch,
    ChannelNorm, S2_MEAN, S2_STD, S1_MEAN, S1_STD,
    labels_to_vec, LABELS_19,
)
from models.dual_encoder import DualEncoder
from utils.faiss_index import GalleryIndex
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Config parsing (from CLI args passed via --) ──────────────────────────────
@st.cache_resource
def load_resources(config_path: str, ckpt_path: str, gallery_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualEncoder(
        s2_in_channels=cfg["model"]["s2_in_channels"],
        s1_in_channels=cfg["model"]["s1_in_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        proj_hidden=cfg["model"]["projection_hidden"],
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        dropout=0.0,
        use_cls_head=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    cleaned = {k.replace("encoder_q.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    gallery = GalleryIndex.load(gallery_dir, use_gpu=cfg["faiss"]["use_gpu"])

    return model, gallery, cfg, device


def preprocess_s2(patch_dir: str, patch_id: str, cfg: dict) -> torch.Tensor:
    bands = cfg["data"]["s2_bands"]
    img   = load_s2_patch(cfg["paths"]["s2_root"], patch_id,
                           bands, cfg["data"]["clip_percentile"])
    norm  = ChannelNorm(S2_MEAN, S2_STD)
    img   = norm(img).astype("float32")
    aug   = A.Compose([A.CenterCrop(cfg["data"]["image_size"], cfg["data"]["image_size"]),
                       ToTensorV2()])
    return aug(image=img)["image"].unsqueeze(0).float()


def preprocess_s1(s1_name: str, cfg: dict) -> torch.Tensor:
    img  = load_s1_patch(cfg["paths"]["s1_root"], s1_name, cfg["data"]["clip_percentile"])
    norm = ChannelNorm(S1_MEAN, S1_STD)
    img  = norm(img).astype("float32")
    aug  = A.Compose([A.CenterCrop(cfg["data"]["image_size"], cfg["data"]["image_size"]),
                      ToTensorV2()])
    return aug(image=img)["image"].unsqueeze(0).float()


def make_rgb_s2(patch_id: str, cfg: dict) -> np.ndarray:
    """Create uint8 RGB image from B04, B03, B02."""
    root = cfg["paths"]["s2_root"]
    tile = "_".join(patch_id.split("_")[:-2])
    patch_dir = Path(root) / tile / patch_id
    r = rasterio.open(str(patch_dir / f"{patch_id}_B04.tif")).read(1).astype(float)
    g = rasterio.open(str(patch_dir / f"{patch_id}_B03.tif")).read(1).astype(float)
    b = rasterio.open(str(patch_dir / f"{patch_id}_B02.tif")).read(1).astype(float)
    rgb = np.stack([r, g, b], axis=-1)
    # Percentile stretch
    for i in range(3):
        lo, hi = np.percentile(rgb[..., i], [2, 98])
        rgb[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def make_vis_s1(s1_name: str, cfg: dict) -> np.ndarray:
    """Visualise VV channel as greyscale."""
    root = cfg["paths"]["s1_root"]
    tile = "_".join(s1_name.split("_")[:-2])
    patch_dir = Path(root) / tile / s1_name
    vv = rasterio.open(str(patch_dir / f"{s1_name}_VV.tif")).read(1).astype(float)
    lo, hi = np.percentile(vv, [2, 98])
    vv = np.clip((vv - lo) / (hi - lo + 1e-6), 0, 1)
    grey = (vv * 255).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main_ui():
    st.set_page_config(
        page_title="Satellite Image Retrieval",
        layout="wide",
        page_icon="🛰️",
    )

    st.title("🛰️ Cross-Modal Satellite Image Retrieval")
    st.markdown("""
    Retrieves semantically similar satellite images across sensor modalities  
    *(Sentinel-2 Optical ↔ Sentinel-1 SAR)* using deep contrastive embeddings + FAISS.
    """)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    config_path  = st.sidebar.text_input("Config path",  "configs/config.yaml")
    ckpt_path    = st.sidebar.text_input("Checkpoint",   "checkpoints/best.pt")
    gallery_dir  = st.sidebar.text_input("Gallery dir",  "outputs/gallery_test")
    K            = st.sidebar.slider("Top-K results", 5, 20, 10)
    query_mod    = st.sidebar.selectbox("Query modality",   ["s2", "s1"])
    gallery_mod  = st.sidebar.selectbox("Gallery modality", ["s1", "s2", "both"])
    g_mod        = None if gallery_mod == "both" else gallery_mod

    if st.sidebar.button("🔄 Load / Reload Model + Gallery"):
        st.cache_resource.clear()

    # ── Load resources ────────────────────────────────────────────────────────
    try:
        model, gallery, cfg, device = load_resources(config_path, ckpt_path, gallery_dir)
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.info("Make sure config, checkpoint, and gallery_dir paths are correct.")
        return

    st.sidebar.success(f"✅ Gallery: {gallery._n:,} vectors")

    # ── Query selection ───────────────────────────────────────────────────────
    st.subheader("🔍 Query Image")
    query_mode = st.radio("Select query source", ["Random test patch", "Enter patch ID"], horizontal=True)

    import pandas as pd
    meta = pd.read_parquet(cfg["paths"]["metadata"])
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    if query_mode == "Random test patch":
        if st.button("🎲 Draw random patch"):
            st.session_state["rand_idx"] = np.random.randint(len(test_meta))
        idx = st.session_state.get("rand_idx", 0)
        row = test_meta.iloc[idx]
        patch_id = row["patch_id"]
        s1_name  = row["s1_name"]
        true_labels = list(row["labels"])
    else:
        patch_id = st.text_input("Patch ID (S2)", test_meta.iloc[0]["patch_id"])
        s1_name  = test_meta[test_meta["patch_id"] == patch_id]["s1_name"].values
        s1_name  = s1_name[0] if len(s1_name) else ""
        true_labels = list(test_meta[test_meta["patch_id"] == patch_id]["labels"].values[0]) \
            if patch_id in test_meta["patch_id"].values else []

    # ── Display query image ───────────────────────────────────────────────────
    col_q, col_r = st.columns([1, 3])
    with col_q:
        st.markdown(f"**Patch ID:** `{patch_id}`")
        st.markdown(f"**S1 name:** `{s1_name}`")
        try:
            if query_mod == "s2":
                vis = make_rgb_s2(patch_id, cfg)
            else:
                vis = make_vis_s1(s1_name, cfg)
            st.image(vis, caption=f"Query ({query_mod.upper()})", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not render image: {e}")
        st.markdown("**True labels:**  \n" + "  \n".join(f"• {l}" for l in true_labels))

    # ── Run retrieval ─────────────────────────────────────────────────────────
    with col_r:
        if st.button("🔎 Retrieve"):
            try:
                if query_mod == "s2":
                    inp = preprocess_s2(patch_id, patch_id, cfg).to(device)
                    with torch.no_grad():
                        qe = model.s2_encoder(inp).float().cpu().numpy()
                else:
                    inp = preprocess_s1(s1_name, cfg).to(device)
                    with torch.no_grad():
                        qe = model.s1_encoder(inp).float().cpu().numpy()

                t0 = time.perf_counter()
                idx_arr, scores, _ = gallery.search(qe, query_mod, g_mod, K=K)
                elapsed = (time.perf_counter() - t0) * 1000

                st.markdown(f"⚡ Retrieval time: **{elapsed:.2f} ms**")

                top_idx = idx_arr[0]
                top_scr = scores[0]

                cols = st.columns(min(K, 5))
                for rank, (gi, sc) in enumerate(zip(top_idx, top_scr)):
                    if gi < 0:
                        continue
                    gal_pid  = gallery.pid_arr[gi]
                    gal_s1n  = gallery.s1n_arr[gi] if gallery.s1n_arr is not None else ""
                    gal_mod  = gallery.mod_arr[gi]
                    gal_lbls = [LABELS_19[j] for j, v in enumerate(gallery.lbl_arr[gi]) if v]

                    col = cols[rank % 5]
                    with col:
                        try:
                            if gal_mod == "s2":
                                vis = make_rgb_s2(gal_pid, cfg)
                            else:
                                vis = make_vis_s1(gal_s1n, cfg)
                            col.image(vis, caption=f"#{rank+1} ({gal_mod.upper()}) {sc:.3f}",
                                      use_column_width=True)
                        except Exception:
                            col.warning(f"#{rank+1} no render")
                        col.caption("  \n".join(f"• {l}" for l in gal_lbls[:3]))

                    if (rank + 1) % 5 == 0 and rank < K - 1:
                        cols = st.columns(5)

            except Exception as e:
                st.error(f"Retrieval error: {e}")


if __name__ == "__main__":
    # Parse CLI args passed after `--` in streamlit run
    import sys
    args_raw = sys.argv[1:]
    # Streamlit passes its own args; find ours after "--"
    if "--" in args_raw:
        idx = args_raw.index("--")
        our_args = args_raw[idx+1:]
    else:
        our_args = []

    main_ui()
