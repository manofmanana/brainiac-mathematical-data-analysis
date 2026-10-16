r"""Linear algebra on real data: SVD image compression, PCA, and Fourier filtering.

* **SVD**: :math:`A = U \Sigma V^\top`. Keeping the top-k singular triplets gives
  :math:`A_k`, the best rank-k approximation in Frobenius norm (Eckart–Young),
  with relative error :math:`\sqrt{\sum_{i>k}\sigma_i^2 / \sum_i \sigma_i^2}`.
* **PCA**: eigen-decomposition of the covariance :math:`C = \frac{1}{N-1} X_c X_c^\top`;
  eigenvalues are the variance captured by each principal axis.
* **FFT**: :math:`\hat A = \mathcal{F}(A)`; zeroing high frequencies blurs, zeroing
  low frequencies keeps edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: str | Path, max_side: int = 320) -> np.ndarray:
    img = Image.open(path).convert("L")
    img.thumbnail((max_side, max_side))
    return np.asarray(img, dtype=float) / 255.0


@dataclass
class SVDCompression:
    reconstruction: np.ndarray
    singular_values: np.ndarray
    energy: float  # fraction of squared Frobenius norm kept
    relative_error: float
    compression_ratio: float  # stored numbers vs original pixels


def svd_compress(A: np.ndarray, k: int) -> SVDCompression:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(np.clip(k, 1, s.size))
    recon = (U[:, :k] * s[:k]) @ Vt[:k]
    total = float(np.sum(s**2))
    kept = float(np.sum(s[:k] ** 2))
    m, n = A.shape
    return SVDCompression(
        reconstruction=np.clip(recon, 0.0, 1.0),
        singular_values=s,
        energy=kept / total,
        relative_error=float(np.sqrt(max(total - kept, 0.0) / total)),
        compression_ratio=(m * n) / (k * (m + n + 1)),
    )


@dataclass
class PCAResult:
    eigenvalues: np.ndarray
    components: np.ndarray  # (d, d) columns are principal axes
    explained_ratio: np.ndarray
    projection: np.ndarray  # (2, N) coordinates on the first two axes


def pca(X: np.ndarray) -> PCAResult:
    """PCA of column-sample data ``X`` (d, N)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ Xc.T / (X.shape[1] - 1)
    vals, vecs = np.linalg.eigh(C)  # symmetric -> real, ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Fix sign so results are deterministic.
    vecs *= np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(vecs.shape[1])])
    return PCAResult(vals, vecs, vals / vals.sum(), vecs[:, :2].T @ Xc)


def fft_filter(A: np.ndarray, radius: float, mode: str = "low") -> tuple[np.ndarray, np.ndarray]:
    """Keep frequencies inside (low-pass) or outside (high-pass) a radius (fraction of Nyquist)."""
    F = np.fft.fftshift(np.fft.fft2(A))
    m, n = A.shape
    yy, xx = np.ogrid[-(m // 2) : m - m // 2, -(n // 2) : n - n // 2]
    dist = np.sqrt((yy / (m / 2)) ** 2 + (xx / (n / 2)) ** 2)
    mask = dist <= radius if mode == "low" else dist > radius
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    if mode == "high":
        filtered = filtered - filtered.min()
        filtered = filtered / (filtered.max() + 1e-12)
    spectrum = np.log1p(np.abs(F)) * mask
    return np.clip(filtered, 0.0, 1.0), spectrum
