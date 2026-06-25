"""
bigearth_retrieval/utils/faiss_index.py
=========================================
FAISS gallery that stores all THREE modalities in a single index and
supports filtered search across all 9 retrieval directions:

  Same-modal  (3): optical→optical, ms→ms, sar→sar
  Cross-modal (6): optical→ms, ms→optical,
                   optical→sar, sar→optical,
                   ms→sar, sar→ms

Evaluation reports F1@5 and F1@10 for every direction independently.
"""

import os
import time
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bigearth_retrieval.data.dataset import ALL_MODALITIES, MODALITY_OPTICAL, MODALITY_MS, MODALITY_SAR

# All evaluation modes: (query_modality, gallery_modality, label)
EVAL_MODES = [
    # ── Same-modal ──────────────────────────────────────────────────────────
    (MODALITY_OPTICAL, MODALITY_OPTICAL, "optical→optical"),
    (MODALITY_MS,      MODALITY_MS,      "ms→ms"),
    (MODALITY_SAR,     MODALITY_SAR,     "sar→sar"),
    # ── Cross-modal ─────────────────────────────────────────────────────────
    (MODALITY_OPTICAL, MODALITY_MS,      "optical→ms"),
    (MODALITY_MS,      MODALITY_OPTICAL, "ms→optical"),
    (MODALITY_OPTICAL, MODALITY_SAR,     "optical→sar"),
    (MODALITY_SAR,     MODALITY_OPTICAL, "sar→optical"),
    (MODALITY_MS,      MODALITY_SAR,     "ms→sar"),
    (MODALITY_SAR,     MODALITY_MS,      "sar→ms"),
]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index factory
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(
    dim: int,
    index_type: str = "IVFFlat",
    nlist: int = 1024,
    nprobe: int = 64,
    use_gpu: bool = True,
) -> faiss.Index:
    """Build a cosine-similarity FAISS index (inner product on L2-normed vecs)."""
    metric = faiss.METRIC_INNER_PRODUCT

    if index_type == "Flat":
        idx = faiss.IndexFlatIP(dim)
    elif index_type == "IVFFlat":
        quant = faiss.IndexFlatIP(dim)
        idx   = faiss.IndexIVFFlat(quant, dim, nlist, metric)
        idx.nprobe = nprobe
    elif index_type == "IVFPQ":
        quant = faiss.IndexFlatIP(dim)
        idx   = faiss.IndexIVFPQ(quant, dim, nlist, 64, 8)
        idx.nprobe = nprobe
    elif index_type == "HNSW":
        idx = faiss.IndexHNSWFlat(dim, 32, metric)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch       = 128
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(res, 0, idx)
        print(f"[FAISS] GPU index ({index_type}, dim={dim})")
    else:
        print(f"[FAISS] CPU index ({index_type}, dim={dim})")

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# GalleryIndex
# ─────────────────────────────────────────────────────────────────────────────

class GalleryIndex:
    """
    Mixed-modality gallery backed by a single FAISS index.

    All embeddings (optical + ms + sar) are stored together.
    At query time, results are filtered by modality tag so each of the 9
    retrieval directions returns only the target modality.
    """

    def __init__(
        self,
        dim:        int,
        index_type: str  = "IVFFlat",
        nlist:      int  = 1024,
        nprobe:     int  = 64,
        use_gpu:    bool = True,
    ):
        self.dim     = dim
        self.index   = build_faiss_index(dim, index_type, nlist, nprobe, use_gpu)
        self._trained = index_type in ("Flat", "HNSW")

        # Metadata arrays — filled via add_batch, frozen by finalize()
        self._modalities: List[str]       = []
        self._labels:     List[np.ndarray] = []
        self._patch_ids:  List[str]       = []
        self._s1_names:   List[str]       = []
        self._n = 0

    # ── Population ────────────────────────────────────────────────────────────

    def train_if_needed(self, embeddings: np.ndarray):
        if not self._trained:
            print(f"[FAISS] Training on {len(embeddings):,} vectors…")
            self.index.train(embeddings.astype(np.float32))
            self._trained = True

    def add_batch(
        self,
        embeddings: np.ndarray,          # (B, D) float32, L2-normed
        modalities: List[str],           # each in ALL_MODALITIES
        labels:     np.ndarray,          # (B, 19)
        patch_ids:  List[str],
        s1_names:   Optional[List[str]] = None,
    ):
        emb = embeddings.astype(np.float32)
        # Re-normalise just in case
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   = emb / (norms + 1e-8)
        self.index.add(emb)
        self._modalities.extend(modalities)
        self._labels.extend(list(labels))
        self._patch_ids.extend(patch_ids)
        self._s1_names.extend(s1_names if s1_names else [""] * len(patch_ids))
        self._n += len(embeddings)

    def finalize(self):
        """Convert lists → numpy for fast fancy-indexing."""
        self.mod_arr  = np.array(self._modalities)
        self.lbl_arr  = np.stack(self._labels)       # (N, 19)
        self.pid_arr  = np.array(self._patch_ids)
        self.s1n_arr  = np.array(self._s1_names)
        counts = {m: int((self.mod_arr == m).sum()) for m in ALL_MODALITIES}
        print(f"[FAISS] Gallery ready: {self._n:,} vectors | {counts}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_emb:   np.ndarray,          # (Q, D)
        query_mod:   str,                 # modality of query (for exclusion)
        gallery_mod: str,                 # target modality to retrieve
        K:           int = 10,
        exclude_self_mod: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search gallery_mod embeddings for each query.

        Strategy: over-retrieve K*6 candidates, then keep only those whose
        modality tag == gallery_mod.  This avoids building per-modality
        sub-indexes while still being fast.

        Returns:
            indices  (Q, K)  gallery positions (−1 = not enough results)
            scores   (Q, K)  cosine similarities
            avg_ms   float   average ms per query
        """
        Q       = query_emb.shape[0]
        qe      = query_emb.astype(np.float32)
        qe      = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-8)

        # How many candidates to retrieve before filtering
        mod_frac   = max((self.mod_arr == gallery_mod).mean(), 0.05)
        K_search   = min(int(K / mod_frac * 1.5) + 32, self._n)

        t0 = time.perf_counter()
        raw_scores, raw_idx = self.index.search(qe, K_search)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out_idx    = np.full((Q, K), -1, dtype=np.int64)
        out_scores = np.full((Q, K), -1.0, dtype=np.float32)

        for q in range(Q):
            ri = raw_idx[q]    # (K_search,)
            rs = raw_scores[q]
            # Valid + target modality
            valid_mask = (ri >= 0) & (self.mod_arr[np.where(ri >= 0, ri, 0)] == gallery_mod)
            valid_mask[ri < 0] = False
            fi = ri[valid_mask][:K]
            fs = rs[valid_mask][:K]
            out_idx[q,    :len(fi)] = fi
            out_scores[q, :len(fs)] = fs

        return out_idx, out_scores, elapsed_ms / Q

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        cpu_idx = (faiss.index_gpu_to_cpu(self.index)
                   if hasattr(self.index, "getDevice") else self.index)
        faiss.write_index(cpu_idx, str(Path(save_dir) / "faiss.index"))
        meta = {
            "modalities": self._modalities,
            "labels":     [l.tolist() for l in self._labels],
            "patch_ids":  self._patch_ids,
            "s1_names":   self._s1_names,
            "dim":        self.dim,
        }
        with open(Path(save_dir) / "gallery_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"[FAISS] Saved → {save_dir}")

    @classmethod
    def load(cls, save_dir: str, use_gpu: bool = True) -> "GalleryIndex":
        cpu_idx = faiss.read_index(str(Path(save_dir) / "faiss.index"))
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        else:
            idx = cpu_idx

        with open(Path(save_dir) / "gallery_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        inst = cls.__new__(cls)
        inst.dim         = meta["dim"]
        inst.index       = idx
        inst._trained    = True
        inst._modalities = meta["modalities"]
        inst._labels     = [np.array(l, dtype=np.float32) for l in meta["labels"]]
        inst._patch_ids  = meta["patch_ids"]
        inst._s1_names   = meta.get("s1_names", [""] * len(meta["patch_ids"]))
        inst._n          = len(inst._modalities)
        inst.finalize()
        print(f"[FAISS] Loaded from {save_dir} ({inst._n:,} vectors)")
        return inst


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def f1_at_k(
    retrieved_idx:  np.ndarray,    # (Q, K)
    query_labels:   np.ndarray,    # (Q, 19)
    gallery_labels: np.ndarray,    # (N, 19)
    K: int,
    threshold: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Mean Precision@K, Recall@K, F1@K over Q queries.
    Relevance = Jaccard(query_labels, gallery_labels) > threshold.
    """
    Q = query_labels.shape[0]
    prec, rec, f1 = [], [], []

    for q in range(Q):
        ql   = query_labels[q]
        rk   = retrieved_idx[q, :K]
        valid = rk[rk >= 0]

        if len(valid) == 0:
            prec.append(0.); rec.append(0.); f1.append(0.)
            continue

        # Relevance of retrieved items
        ret_lbls = gallery_labels[valid]
        inter    = (ql * ret_lbls).sum(axis=1)
        union    = (ql + ret_lbls - ql * ret_lbls).sum(axis=1).clip(min=1.)
        rel_k    = (inter / union > threshold).astype(float)

        # Total relevant in full gallery (for recall denominator)
        g_inter  = ql @ gallery_labels.T
        g_union  = (ql.sum() + gallery_labels.sum(axis=1) - g_inter).clip(min=1.)
        total_rel = int((g_inter / g_union > threshold).sum()) - 1  # -1 for self

        n_rel = int(rel_k.sum())
        P = n_rel / len(valid)
        R = n_rel / max(total_rel, 1)
        F = 2 * P * R / (P + R + 1e-8) if (P + R) > 0 else 0.

        prec.append(P); rec.append(R); f1.append(F)

    return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))


def evaluate_all_modes(
    gallery:         GalleryIndex,
    embeddings:      Dict[str, np.ndarray],   # mod → (N_mod, D)
    labels:          Dict[str, np.ndarray],   # mod → (N_mod, 19)
    Ks:              List[int] = [5, 10],
    threshold:       float = 0.0,
) -> List[Dict]:
    """
    Run retrieval evaluation for all 9 modes in EVAL_MODES.

    embeddings / labels dicts must contain keys for all three modalities.

    Returns list of result dicts, one per mode.
    """
    results = []
    K_max   = max(Ks)

    for q_mod, g_mod, label in EVAL_MODES:
        qe = embeddings[q_mod]
        ql = labels[q_mod]

        idx, scores, avg_ms = gallery.search(qe, q_mod, g_mod, K=K_max)

        r = {"mode": label, "avg_query_time_ms": avg_ms}
        for K in Ks:
            P, R, F1 = f1_at_k(idx, ql, gallery.lbl_arr, K=K,
                                threshold=threshold)
            r[f"P@{K}"]  = P
            r[f"R@{K}"]  = R
            r[f"F1@{K}"] = F1
        results.append(r)

    return results
