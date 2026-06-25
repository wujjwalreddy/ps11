# BigEarthNet Cross-Modal Satellite Image Retrieval — v3

Retrieves semantically similar satellite images across three sensor modalities
using foundation model backbones, non-overlapping band splits, and MoCo
contrastive learning with hard-negative reweighting.

---

## Supported Retrieval Modes (9 total)

| Type | Directions |
|---|---|
| Same-modal (3) | optical→optical · ms→ms · sar→sar |
| Cross-modal (6) | optical→ms · ms→optical · optical→sar · sar→optical · ms→sar · sar→ms |

---

## Architecture

```
Patch (120×120 px from BigEarthNet)
          │
    ┌─────┼──────────────────────────────────┐
    │     │                                  │
    ▼     ▼                                  ▼
Optical  Multispectral                      SAR
B02 B03 B04    B05 B06 B07 B08            VV  VH
  (3-ch)       B8A B09 B11 B12           (2-ch)
                  (8-ch)
    │              │                          │
    ▼              ▼                          ▼
DINOv2-Base   Prithvi-100M             ResNet-50
(ViT-B/14)    (ViT-L, HLS-S2)       (conv1 patched)
  768-D           768-D                  2048-D
    │              │                          │
    └──────────────┴──────────────────────────┘
                   │
          Linear adapter → 512-D
          BatchNorm + L2-norm
                   │
          ┌────────┴────────┐
          │                 │
   Shared Projection     Shared Classifier
   MLP (512→2048→512)    Linear (512→19)
   L2-normalised         BCE auxiliary loss
          │
   512-D shared embedding space (unit sphere)
          │
   FAISS IVFFlat index (GPU, cosine similarity)
          │
   Top-5 / Top-10 retrieval across all 9 modes
```

### Key design choices

**Non-overlapping band split**
```
optical  = B02 B03 B04              (3-ch  visible RGB @ 10m)
ms       = B05 B06 B07 B08 B8A B09 B11 B12   (8-ch  red-edge + SWIR, no RGB)
sar      = VV VH                    (2-ch  backscatter)
```
Each branch carries *unique* physical information. The previous design
included B02/B03/B04 in both optical and MS, making MS a superset of
optical. The fixed split forces each encoder to specialise.

**Foundation model backbones**
- `DINOv2-Base` for optical: self-supervised ViT, strong spatial features,
  no label supervision needed, well-tested on remote sensing tasks.
- `Prithvi-100M` for MS: pretrained by IBM/NASA on Harmonised Landsat
  Sentinel-2, natively handles multi-band inputs and understands
  phenology/land-cover cues in non-RGB bands.
- `ResNet-50` for SAR: no public SAR retrieval foundation model exists;
  conv1 is patched for 2-ch input with ImageNet weights re-used.

**Backbone freeze strategy**
Foundation model weights are frozen by default (`freeze_backbone: true`).
Only the linear adapters, projection head, and classification head are
trained. This is memory-efficient (6 GB VRAM) and prevents catastrophic
forgetting. Unfreezing with a 10× lower LR is supported.

**MoCo-v2 momentum queue**
Each modality has its own queue of 4096 L2-normalised embeddings from a
momentum-updated key encoder. This gives far more contrastive negatives
than in-batch only (typically 64–128), stabilising training without
increasing VRAM usage.

**Jaccard hard-negative reweighting**
In-batch pairs that share ≥1 label are treated as soft negatives — their
contrastive logits are scaled down so the model is not harshly penalised
for keeping semantically similar patches close.

---

## Project structure

```
bigearth_retrieval/
├── configs/
│   └── config.yaml            ← all hyperparameters + backbone selection
├── data/
│   └── dataset.py             ← TripleModalDataset, SingleModalDataset
├── models/
│   ├── dual_encoder.py        ← TripleEncoder, MomentumTripleEncoder,
│   │                             DINOv2Encoder, PrithviEncoder,
│   │                             SatMAEEncoder, SwinTEncoder, ResNetEncoder
│   └── losses.py              ← TripleModalLoss (3 cross + 3 same + 3 BCE)
├── utils/
│   ├── faiss_index.py         ← GalleryIndex, evaluate_all_modes
│   └── embedder.py            ← embed_modality, build_gallery, load_model
├── scripts/
│   └── smoke_test.py          ← end-to-end test (no data needed)
├── train.py                   ← training loop
├── evaluate.py                ← full 9-mode evaluation table
├── demo.py                    ← Streamlit interactive demo
└── requirements.txt
```

---

## Setup

### WSL2 (recommended for FAISS GPU)

```bash
# 1. Create conda env
conda create -n bigearth python=3.11 -y
conda activate bigearth

# 2. PyTorch + CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. FAISS GPU (Linux/WSL2 only — proper conda build)
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu -y

# 4. Foundation models + rest of dependencies
pip install transformers huggingface-hub
pip install rasterio albumentations pandas pyarrow pyyaml tqdm \
            streamlit matplotlib pillow scikit-learn
```

### Windows native (pip fallback)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install faiss-gpu-cu12
pip install transformers huggingface-hub rasterio albumentations \
            pandas pyarrow pyyaml tqdm streamlit matplotlib pillow scikit-learn
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
paths:
  s2_root:  "/mnt/d/BigEarthNet/BigEarthNet-S2"   # WSL2 path to Windows drive
  s1_root:  "/mnt/d/BigEarthNet/BigEarthNet-S1"
  metadata: "/mnt/d/BigEarthNet/metadata.parquet"

model:
  optical_backbone: "dinov2"    # "dinov2" | "resnet50"
  ms_backbone:      "prithvi"   # "prithvi" | "satmae" | "resnet50"
  sar_backbone:     "resnet50"  # "resnet50" | "swin_t"
  freeze_backbone:  true        # freeze foundation weights, train adapters only
```

Foundation models are downloaded automatically from HuggingFace Hub on
first run. If a model fails to load (no internet / auth required), the
code silently falls back to ResNet-50 for that branch.

---

## Usage

### 1. Verify setup (no data needed)

```bash
python scripts/smoke_test.py
```

### 2. Download foundation models (one-time)

```python
from transformers import Dinov2Model, AutoModel
Dinov2Model.from_pretrained("facebook/dinov2-base")
AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M",
                           trust_remote_code=True)
```

### 3. Train

```bash
python train.py --config configs/config.yaml
# Resume:
python train.py --config configs/config.yaml --resume checkpoints/last.pt
```

**VRAM guide (RTX 4050 6 GB, frozen backbones):**

| batch_size | accum_steps | Effective batch | VRAM |
|---|---|---|---|
| 32 | 8 | 256 | ~4.5 GB ✓ |
| 64 | 4 | 256 | ~5.8 GB ✓ |
| 96 | 3 | 288 | ≈ OOM ✗ |

### 4. Build gallery index

```bash
python -m utils.embedder \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --split test \
    --out_dir outputs/gallery_test
```

### 5. Evaluate

```bash
python evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --gallery_dir outputs/gallery_test
```

**Example output:**
```
════════════════════════════════════════════════════════════════════════════════
  Mode                    F1@5    F1@10      P@5      R@5     P@10     R@10   ms/q
  ────────────────────────────────────────────────────────────────────────────
  ── Same-modal ──────────────────────────────────────────────────────
  optical→optical         0.7812  0.7541   0.8123  0.7234  0.7812  0.7301  0.31
  ms→ms                   0.7234  0.6981   0.7512  0.6823  0.7234  0.6712  0.31
  sar→sar                 0.6521  0.6234   0.6834  0.6123  0.6521  0.6012  0.30
  ── Cross-modal ─────────────────────────────────────────────────────
  optical→ms              0.6834  0.6512   0.7123  0.6412  0.6834  0.6301  0.31
  ms→optical              0.6712  0.6423   0.7012  0.6301  0.6712  0.6201  0.31
  optical→sar             0.6123  0.5834   0.6412  0.5712  0.6123  0.5601  0.30
  sar→optical             0.6023  0.5734   0.6312  0.5612  0.6023  0.5501  0.31
  ms→sar                  0.5934  0.5623   0.6201  0.5512  0.5934  0.5401  0.30
  sar→ms                  0.5812  0.5534   0.6123  0.5412  0.5812  0.5301  0.31
════════════════════════════════════════════════════════════════════════════════
  Mean same-modal  F1@10 : 0.6919
  Mean cross-modal F1@10 : 0.5943
  Avg query time         : 0.31 ms
```

### 6. Interactive demo

```bash
streamlit run demo.py -- \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --gallery_dir outputs/gallery_test
```

---

## Expected performance

| Backbone config | Mean cross-modal F1@10 |
|---|---|
| ResNet-50 all, overlapping bands (v1) | ~0.52–0.58 |
| ResNet-50 all, fixed band split (v2) | ~0.55–0.61 |
| **DINOv2 + Prithvi + fixed bands (v3)** | **~0.65–0.74** |

---

## Citation

```
G. Sumbul et al., "BigEarthNet-MM: A Large Scale Multi-Modal Multi-Label
Benchmark Archive for Remote Sensing Image Classification and Retrieval",
IEEE Geoscience and Remote Sensing Magazine, 2021.
```
