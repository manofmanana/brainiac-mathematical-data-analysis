"""Plotly figures for the Streamlit app (dark, transparent, neon accents)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brainiac.datasets import Dataset
from brainiac.network import History, NeuralNetwork

GREEN, CYAN, PURPLE, PINK, AMBER = "#00ff88", "#49f3e1", "#a855f7", "#ff4fd8", "#ffb347"
PALETTE = [GREEN, PURPLE, CYAN, AMBER, PINK, "#7dd3fc", "#f87171", "#a3e635"]
TEXT = "#d8f5ea"


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, font=dict(size=15, color=GREEN)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,18,0.55)",
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=50, r=20, t=50 if title else 20, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    return fig


# ------------------------------------------------------------------ training


def training_curves(hist: History, task: str) -> go.Figure:
    epochs = np.arange(1, len(hist.loss) + 1)
    fig = make_subplots(specs=[[{"secondary_y": task == "classification"}]])
    fig.add_trace(go.Scatter(x=epochs, y=hist.loss, name="training loss", line=dict(color=GREEN, width=2.5)))
    if task == "classification":
        fig.add_trace(
            go.Scatter(x=epochs, y=hist.accuracy, name="training accuracy", line=dict(color=PURPLE, width=2)),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="accuracy",
            range=[0, 1.02],
            tickformat=".0%",
            tickmode="array",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            secondary_y=True,
            showgrid=False,
        )
    fig.update_yaxes(title_text="loss (log scale)", type="log", secondary_y=False)
    fig.update_xaxes(title_text="epoch")
    return _style(fig, "Loss & accuracy per epoch")


def grad_norms(hist: History) -> go.Figure:
    norms = np.array(hist.grad_norms)
    epochs = np.arange(1, len(norms) + 1)
    fig = go.Figure()
    for j in range(norms.shape[1]):
        fig.add_trace(
            go.Scatter(x=epochs, y=norms[:, j], name=f"layer {j + 1}", line=dict(color=PALETTE[j % len(PALETTE)]))
        )
    fig.update_yaxes(type="log", title_text="‖∂L/∂W‖ (Frobenius, log)")
    fig.update_xaxes(title_text="epoch")
    return _style(fig, "Gradient norm per layer")


def decision_boundary(net: NeuralNetwork, data: Dataset) -> go.Figure:
    """Predicted class over the (standardised) 2-D input plane, with train/test points."""
    pts = np.hstack([data.X_train, data.X_test])
    lo, hi = pts.min(axis=1) - 0.6, pts.max(axis=1) + 0.6
    xs, ys = np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160)
    XX, YY = np.meshgrid(xs, ys)
    proba = net.predict_proba(np.vstack([XX.ravel(), YY.ravel()]))
    pred = proba.argmax(axis=0).reshape(XX.shape)
    conf = proba.max(axis=0).reshape(XX.shape)
    k = data.n_outputs
    colors = PALETTE[:k]
    # Colour = predicted class, brightness = confidence.
    z = pred + 0.999 * (conf - 1.0 / k) / (1.0 - 1.0 / k)
    scale = []
    for c in range(k):
        scale += [(c / k, "#0b0d10"), ((c + 1) / k - 1e-6, colors[c])]
    scale[-1] = (1.0, colors[-1])  # plotly colorscales must end exactly at 1
    fig = go.Figure(
        go.Heatmap(x=xs, y=ys, z=z, zmin=0, zmax=k, colorscale=scale, showscale=False, opacity=0.55, hoverinfo="skip")
    )
    for split, X, labels, symbol in (
        ("train", data.X_train, data.labels_train, "circle"),
        ("test", data.X_test, data.labels_test, "diamond"),
    ):
        for c in range(k):
            m = labels == c
            fig.add_trace(
                go.Scatter(
                    x=X[0, m],
                    y=X[1, m],
                    mode="markers",
                    name=f"{data.class_names[c]} ({split})",
                    marker=dict(color=colors[c], symbol=symbol, size=7, line=dict(color="#0b0d10", width=1)),
                )
            )
    fig.update_xaxes(title_text=f"{data.feature_names[0]} (standardised)")
    fig.update_yaxes(title_text=f"{data.feature_names[1]} (standardised)", scaleanchor="x", scaleratio=1)
    return _style(fig, "Decision regions (brightness = confidence)", height=460)


def regression_fit(net: NeuralNetwork, data: Dataset) -> go.Figure:
    xs = np.linspace(data.X_train.min() - 0.3, data.X_train.max() + 0.3, 300).reshape(1, -1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.X_train[0],
            y=data.Y_train[0],
            mode="markers",
            name="train",
            marker=dict(color=CYAN, size=5, opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.X_test[0],
            y=data.Y_test[0],
            mode="markers",
            name="test",
            marker=dict(color=AMBER, size=6, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter(x=xs[0], y=net.predict(xs)[0], mode="lines", name="network", line=dict(color=GREEN, width=3))
    )
    fig.update_xaxes(title_text="x (standardised)")
    fig.update_yaxes(title_text="y")
    return _style(fig, "Network fit", height=460)


def confusion_matrix(net: NeuralNetwork, data: Dataset) -> go.Figure:
    k = data.n_outputs
    pred, true = net.predict(data.X_test), data.labels_test
    cm = np.zeros((k, k), int)
    np.add.at(cm, (true, pred), 1)
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=data.class_names,
            y=data.class_names,
            colorscale=[[0, "#0b0d10"], [1, GREEN]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_xaxes(title_text="predicted")
    fig.update_yaxes(title_text="actual", autorange="reversed")
    return _style(fig, "Test-set confusion matrix", height=460)


def weight_histograms(net: NeuralNetwork, init_std: list[float]) -> go.Figure:
    fig = go.Figure()
    for j, layer in enumerate(net.dense_layers):
        fig.add_trace(
            go.Histogram(
                x=layer.W.ravel(),
                name=f"W{j + 1} (init σ={init_std[j]:.2f})",
                opacity=0.6,
                marker_color=PALETTE[j % len(PALETTE)],
                nbinsx=40,
            )
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="weight value")
    return _style(fig, "Trained weight distributions")


# ------------------------------------------------------------- verification


def gradcheck_scatter(analytic: np.ndarray, numeric: np.ndarray) -> go.Figure:
    lim = float(max(np.abs(analytic).max(), np.abs(numeric).max(), 1e-8)) * 1.1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="y = x", line=dict(color=PURPLE, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(
            x=numeric, y=analytic, mode="markers", name="parameters", marker=dict(color=GREEN, size=6, opacity=0.8)
        )
    )
    fig.update_xaxes(title_text="numerical gradient (finite differences)")
    fig.update_yaxes(title_text="analytic gradient (backprop)")
    return _style(fig, "Every point should sit on the diagonal", height=420)


# ---------------------------------------------------------------- landscape


def surface_3d(alphas: np.ndarray, betas: np.ndarray, Z: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Surface(
            x=alphas,
            y=betas,
            z=np.log10(Z + 1e-12),
            colorscale=[[0, "#0b0d10"], [0.35, PURPLE], [0.7, CYAN], [1, GREEN]],
            colorbar=dict(title="log₁₀ L"),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="α (direction 1)",
            yaxis_title="β (direction 2)",
            zaxis_title="log₁₀ loss",
            bgcolor="rgba(0,0,0,0)",
        )
    )
    return _style(fig, "Loss surface around the trained weights", height=520)


def interpolation(ts: np.ndarray, losses: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Scatter(x=ts, y=losses, mode="lines", line=dict(color=GREEN, width=3)))
    for x, label in ((0, "θ₀ (init)"), (1, "θ* (trained)")):
        fig.add_vline(x=x, line=dict(color=PURPLE, dash="dot"), annotation_text=label)
    fig.update_xaxes(title_text="t in θ(t) = (1 − t)·θ₀ + t·θ*")
    fig.update_yaxes(title_text="loss", type="log")
    return _style(fig, "Straight line from initialisation to solution")


# -------------------------------------------------------------- experiments


def experiment_curves(runs) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        # Losses span orders of magnitude, so summarise in log space:
        # geometric mean with a multiplicative ±1 sd band.
        logc = np.log10(np.clip(r.curves, 1e-12, None))
        mean = 10 ** np.nanmean(logc, axis=0)
        sd = np.nanstd(logc, axis=0)
        x = np.arange(1, mean.size + 1)
        upper, lower = mean * 10**sd, mean / 10**sd
        rgba = _hex_to_rgba(color, 0.15)
        fig.add_trace(
            go.Scatter(
                x=np.r_[x, x[::-1]],
                y=np.r_[upper, lower[::-1]],
                fill="toself",
                fillcolor=rgba,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(go.Scatter(x=x, y=mean, name=r.label, line=dict(color=color, width=2.5)))
    fig.update_yaxes(type="log", title_text="training loss (geometric mean ± 1 sd over seeds)")
    fig.update_xaxes(title_text="epoch")
    return _style(fig, "Convergence by activation × initialisation", height=460)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# --------------------------------------------------------------------- lab


def image(img: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    return _style(fig, title, height=340)


def singular_values(s: np.ndarray, k: int) -> go.Figure:
    energy = np.cumsum(s**2) / np.sum(s**2)
    idx = np.arange(1, s.size + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=idx, y=s, name="σᵢ", line=dict(color=GREEN)))
    fig.add_trace(go.Scatter(x=idx, y=energy, name="cumulative energy", line=dict(color=PURPLE)), secondary_y=True)
    fig.add_vline(x=k, line=dict(color=AMBER, dash="dot"), annotation_text=f"k = {k}")
    fig.update_yaxes(type="log", title_text="singular value", secondary_y=False)
    fig.update_yaxes(
        title_text="energy kept",
        range=[0, 1.02],
        tickformat=".0%",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        secondary_y=True,
        showgrid=False,
    )
    fig.update_xaxes(title_text="index i")
    return _style(fig, "Singular value spectrum", height=340)


def scree(explained: np.ndarray) -> go.Figure:
    idx = [f"PC{i + 1}" for i in range(explained.size)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=explained, name="explained variance", marker_color=GREEN))
    fig.add_trace(go.Scatter(x=idx, y=np.cumsum(explained), name="cumulative", line=dict(color=PURPLE)))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return _style(fig, "Variance explained by each principal component", height=360)


def pca_scatter(proj: np.ndarray, labels: np.ndarray, names: list[str]) -> go.Figure:
    fig = go.Figure()
    for c, name in enumerate(names):
        m = labels == c
        fig.add_trace(
            go.Scatter(
                x=proj[0, m],
                y=proj[1, m],
                mode="markers",
                name=name,
                marker=dict(color=PALETTE[c % len(PALETTE)], size=7),
            )
        )
    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    return _style(fig, "Data projected onto the top two principal axes", height=360)
