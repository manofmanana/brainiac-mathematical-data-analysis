"""Brainiac: an interactive lab for a neural network built from first principles.

Run with:  streamlit run main.py
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from brainiac import ACTIVATIONS, DATASETS, INIT_SCHEMES, NeuralNetwork, load_dataset
from brainiac.curvature import hutchinson_trace, top_eigenvalue
from brainiac.experiments import compare_configs
from brainiac.gradcheck import gradient_check
from brainiac.initializers import target_std
from brainiac.landscape import interpolation_path, loss_surface
from brainiac.linalg_lab import fft_filter, load_grayscale, pca, svd_compress
from brainiac.network import History

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

st.set_page_config(page_title="Brainiac · neural network from first principles", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- styling
@st.cache_resource
def _css() -> str:
    css = (ROOT / "app" / "static" / "style.css").read_text(encoding="utf-8")
    bg = ASSETS / "space.webp"
    if bg.exists():
        b64 = base64.b64encode(bg.read_bytes()).decode()
        css += (
            '\n[data-testid="stApp"] { background: linear-gradient(rgba(6,8,12,0.82), rgba(6,8,12,0.9)), '
            f'url("data:image/webp;base64,{b64}") center / cover fixed; }}'
        )
    return f"<style>{css}</style>"


st.markdown(_css(), unsafe_allow_html=True)


# --------------------------------------------------------------- training
@st.cache_data(show_spinner=False, max_entries=64)
def train(
    dataset: str,
    noise: float,
    hidden: tuple[int, ...],
    activation: str,
    init: str,
    optimizer: str,
    lr: float,
    batch_size: int | None,
    epochs: int,
    seed: int,
):
    """Train once per configuration; cached so every tab reuses the same model."""
    data = load_dataset(dataset, noise=noise, seed=seed)
    net = NeuralNetwork([data.n_features, *hidden, data.n_outputs], activation, init, data.task, seed=seed)
    theta0 = net.get_flat_params().copy()
    t = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        hist = net.fit(data.X_train, data.Y_train, epochs=epochs, lr=lr, optimizer=optimizer, batch_size=batch_size)
    return theta0, net.get_flat_params(), hist, time.perf_counter() - t


def rebuild(cfg: dict, theta: np.ndarray):
    data = load_dataset(cfg["dataset"], noise=cfg["noise"], seed=cfg["seed"])
    net = NeuralNetwork(
        [data.n_features, *cfg["hidden"], data.n_outputs], cfg["activation"], cfg["init"], data.task, seed=cfg["seed"]
    )
    net.set_flat_params(theta)
    return data, net


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚙️ Experiment controls")
    dataset = st.selectbox("Dataset", list(DATASETS), help="All datasets are generated locally and seeded.")
    default_noise = 0.1 if dataset.startswith("Noisy") else (0.0 if dataset.startswith("Iris") else 0.15)
    noise = st.slider("Noise", 0.0, 0.5, default_noise, 0.01)
    hidden_txt = st.text_input("Hidden layers (comma-separated widths)", "16, 16")
    try:
        hidden = tuple(int(h) for h in hidden_txt.replace(" ", "").split(",") if h)
        if not hidden or min(hidden) < 1 or max(hidden) > 256 or len(hidden) > 6:
            raise ValueError
    except ValueError:
        st.error("Use 1–6 positive widths up to 256, e.g. `16, 16`.")
        hidden = (16, 16)
    c1, c2 = st.columns(2)
    activation = c1.selectbox("Activation", list(ACTIVATIONS)[:-1], index=3)
    init = c2.selectbox("Initialisation", INIT_SCHEMES, index=1)
    c3, c4 = st.columns(2)
    optimizer = c3.selectbox("Optimiser", ["adam", "momentum", "sgd"])
    lr = c4.select_slider("Learning rate", [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], value=0.01)
    c5, c6 = st.columns(2)
    batch_label = c5.selectbox("Batch size", ["16", "32", "64", "full batch"], index=1)
    batch_size = None if batch_label == "full batch" else int(batch_label)
    seed = int(c6.number_input("Seed", 0, 9999, 0))
    epochs = st.slider("Epochs", 10, 600, 200, 10)

cfg = dict(
    dataset=dataset,
    noise=noise,
    hidden=hidden,
    activation=activation,
    init=init,
    optimizer=optimizer,
    lr=lr,
    batch_size=batch_size,
    epochs=epochs,
    seed=seed,
)
with st.spinner("Training…"):
    theta0, theta, hist, train_time = train(**cfg)
data, net = rebuild(cfg, theta)
hist: History
diverged = not np.isfinite(hist.loss[-1])

# ----------------------------------------------------------------- header
st.markdown(
    "<div class='brainiac-header'><h1><span class='title'>Brainiac</span></h1>"
    "<p>A neural network built from first principles in NumPy: every derivative written by hand and checked numerically.</p></div>",
    unsafe_allow_html=True,
)

tab_train, tab_check, tab_land, tab_exp, tab_lab, tab_math = st.tabs(
    [
        "🧠 Train & diagnose",
        "✅ Gradient check",
        "⛰️ Loss landscape",
        "⚖️ Experiments",
        "🧪 Linear algebra lab",
        "📐 The math",
    ]
)

# ------------------------------------------------------------ train tab
with tab_train:
    if diverged:
        st.error(
            f"Training diverged after {len(hist.loss)} epochs (the loss became infinite). "
            "Lower the learning rate, or switch to Adam."
        )
    m = st.columns(5)
    m[0].metric(
        "Training loss",
        f"{hist.loss[-1]:.4f}",
        delta=f"{hist.loss[-1] - hist.loss[0]:+.3f} since epoch 1",
        delta_color="inverse",
    )
    if data.task == "classification":
        m[1].metric("Train accuracy", f"{net.accuracy(data.X_train, data.Y_train):.1%}")
        m[2].metric("Test accuracy", f"{net.accuracy(data.X_test, data.Y_test):.1%}")
    else:
        m[1].metric("Train MSE", f"{net.loss(data.X_train, data.Y_train):.4f}")
        m[2].metric("Test MSE", f"{net.loss(data.X_test, data.Y_test):.4f}")
    m[3].metric("Parameters", f"{net.n_params:,}")
    m[4].metric("Training time", f"{train_time * 1000:.0f} ms")

    left, right = st.columns([1.1, 1])
    with left:
        if data.task == "regression":
            st.plotly_chart(charts.regression_fit(net, data), width="stretch")
        elif data.n_features == 2:
            st.plotly_chart(charts.decision_boundary(net, data), width="stretch")
        else:
            st.plotly_chart(charts.confusion_matrix(net, data), width="stretch")
    with right:
        st.plotly_chart(charts.training_curves(hist, data.task), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.grad_norms(hist), width="stretch")
        st.caption(
            "Gradients shrinking toward the first layer signal **vanishing gradients**: try sigmoid with "
            "`8, 8, 8, 8` hidden layers and *normal* initialisation, then switch to ReLU + He."
        )
    with right:
        init_std = [
            target_std(init, a, b) for a, b in zip([data.n_features, *hidden], [*hidden, data.n_outputs], strict=True)
        ]
        st.plotly_chart(charts.weight_histograms(net, init_std), width="stretch")
        st.caption(f"{init} initialisation draws each layer's weights with the σ shown in the legend.")

# ------------------------------------------------------------ gradcheck
with tab_check:
    st.markdown(
        "Backpropagation computes the exact gradient with the chain rule. To be sure it's implemented correctly, "
        "we compare it with a **central finite difference** for each parameter:"
    )
    st.latex(
        r"\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon}, \qquad \varepsilon = 10^{-6}"
    )
    n_check = min(64, data.X_train.shape[1])
    result = gradient_check(net, data.X_train[:, :n_check], data.Y_train[:, :n_check])
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative error", f"{result.overall:.1e}")
    c2.metric("Parameters compared", f"{result.checked.size:,} / {net.n_params:,}")
    c3.metric(
        "Verdict",
        "✅ Correct" if result.passed else "❌ Mismatch",
        help="Passes when the relative error is below 1e-5.",
    )
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(
            charts.gradcheck_scatter(result.analytic[result.checked], result.numeric[result.checked]), width="stretch"
        )
    with right:
        st.markdown("**Relative error per parameter tensor**")
        st.dataframe(
            pd.DataFrame(result.per_layer, columns=["tensor", "relative error"]).style.format(
                {"relative error": "{:.2e}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Below about 1e-7 is perfect agreement; the remaining error is floating-point rounding.")

# ------------------------------------------------------------ landscape
with tab_land:
    if diverged:
        st.info("Train a model that converges to explore its loss landscape.")
    else:
        c1, c2 = st.columns([1, 3])
        span = c1.slider("Slice width", 0.2, 2.0, 1.0, 0.1, help="Distance along each filter-normalised direction")
        res = c1.slider("Resolution", 11, 41, 25, 2)
        surf = loss_surface(net, data.X_train, data.Y_train, span=span, resolution=res, seed=seed)
        with c2:
            st.plotly_chart(charts.surface_3d(surf.alphas, surf.betas, surf.loss), width="stretch")
        st.caption(
            "A 2-D slice L(θ* + αδ + βη) along two random, filter-normalised directions (Li et al., 2018). "
            "Wide, flat basins tend to generalise better than sharp ones."
        )

        left, right = st.columns(2)
        with left:
            ts, losses = interpolation_path(net, theta0, data.X_train, data.Y_train)
            st.plotly_chart(charts.interpolation(ts, losses), width="stretch")
        with right:
            st.markdown("#### Curvature at the solution")
            grad_fn = lambda th: net.flat_grad(th, data.X_train, data.Y_train)  # noqa: E731
            lam = top_eigenvalue(grad_fn, theta, seed=seed)
            tr = hutchinson_trace(grad_fn, theta, n_samples=30, seed=seed)
            k1, k2 = st.columns(2)
            k1.metric("Top Hessian eigenvalue λₘₐₓ", f"{lam:.3f}")
            k2.metric("Hessian trace (Hutchinson)", f"{tr.mean:.2f} ± {tr.stderr:.2f}")
            st.markdown(
                "Both use **Hessian-vector products** "
                r"$Hv \approx \frac{\nabla L(\theta+\varepsilon v) - \nabla L(\theta-\varepsilon v)}{2\varepsilon}$, "
                "so the Hessian is never built. Power iteration gives λₘₐₓ; Hutchinson's estimator "
                r"$\mathbb{E}[v^\top H v] = \operatorname{tr}(H)$ gives the trace."
            )
            if lam > 0:
                bound = 2 / lam
                verdict = "within" if lr < bound else "**above**"
                st.markdown(
                    f"Plain gradient descent is stable only when η < 2/λₘₐₓ = **{bound:.3f}**. "
                    f"Your learning rate η = {lr} is {verdict} that bound."
                )

# ------------------------------------------------------------ experiments
with tab_exp:
    st.markdown(
        "Compare activation × initialisation choices on the current dataset and architecture. Each configuration "
        "trains on several seeds, so you see the **mean and spread** rather than one lucky run."
    )
    with st.form("experiment"):
        c1, c2 = st.columns(2)
        acts = c1.multiselect("Activations", list(ACTIVATIONS)[:-1], default=["relu", "tanh", "sigmoid"])
        inits = c2.multiselect("Initialisations", INIT_SCHEMES, default=["he", "xavier"])
        c3, c4 = st.columns(2)
        n_seeds = c3.slider("Seeds per configuration", 1, 5, 3)
        exp_epochs = c4.slider("Epochs", 20, 300, 120, 10)
        run = st.form_submit_button("▶ Run experiment", type="primary")
    if run:
        if not acts or not inits:
            st.warning("Pick at least one activation and one initialisation.")
        else:
            with st.spinner(f"Training {len(acts) * len(inits) * n_seeds} networks…"):
                target = 0.1 if data.task == "classification" else 0.05
                runs = compare_configs(
                    data,
                    list(hidden),
                    acts,
                    inits,
                    list(range(n_seeds)),
                    epochs=exp_epochs,
                    lr=lr,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    target=target,
                )
            st.plotly_chart(charts.experiment_curves(runs), width="stretch")
            metric = "test accuracy" if data.task == "classification" else "test MSE"
            fmt = "{:.1%}" if data.task == "classification" else "{:.4f}"
            rows = []
            for r in runs:
                rows.append(
                    {
                        "configuration": r.label,
                        "final loss": f"{np.mean(r.final_loss):.4f} ± {np.std(r.final_loss):.4f}",
                        metric: f"{fmt.format(np.mean(r.test_metric))} ± {fmt.format(np.std(r.test_metric))}",
                        f"epochs to loss < {target}": (
                            f"{np.nanmean(r.epochs_to_target):.0f}"
                            if np.isfinite(r.epochs_to_target).any()
                            else "never"
                        ),
                        "_sort": np.mean(r.final_loss),
                    }
                )
            table = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Values are mean ± standard deviation across seeds. Sorted by final training loss.")

# ------------------------------------------------------------ linalg lab
with tab_lab:
    st.markdown("#### SVD image compression")
    img = load_grayscale(ASSETS / "space.webp")
    k = st.slider("Rank k (number of singular values kept)", 1, 120, 20)
    comp = svd_compress(img, k)
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy kept", f"{comp.energy:.1%}")
    c2.metric("Relative error", f"{comp.relative_error:.1%}")
    c3.metric("Compression", f"{comp.compression_ratio:.1f}×")
    a, b, c = st.columns(3)
    a.plotly_chart(charts.image(img, f"Original ({img.shape[0]}×{img.shape[1]})"), width="stretch")
    b.plotly_chart(charts.image(comp.reconstruction, f"Rank-{k} reconstruction"), width="stretch")
    c.plotly_chart(charts.singular_values(comp.singular_values, k), width="stretch")
    st.caption(
        "By the Eckart–Young theorem, keeping the top k singular triplets is the best possible rank-k approximation."
    )

    st.markdown("#### Principal component analysis of the current dataset")
    p = pca(np.hstack([data.X_train, data.X_test]))
    if data.task == "classification":
        labels = np.r_[data.labels_train, data.labels_test]
        names = data.class_names
    else:
        labels, names = np.zeros(data.X_train.shape[1] + data.X_test.shape[1], int), ["samples"]
    left, right = st.columns(2)
    left.plotly_chart(charts.scree(p.explained_ratio), width="stretch")
    if p.eigenvalues.size >= 2:
        right.plotly_chart(charts.pca_scatter(p.projection, labels, names), width="stretch")
    else:
        right.info("PCA needs at least two features; pick Iris for the most interesting projection.")

    st.markdown("#### Fourier filtering")
    c1, c2 = st.columns([1, 3])
    mode = c1.radio("Filter", ["low", "high"], format_func=lambda m: f"{m}-pass", horizontal=True)
    radius = c1.slider("Cut-off radius (fraction of Nyquist)", 0.01, 1.0, 0.12, 0.01)
    filtered, spectrum = fft_filter(img, radius, mode)
    with c2:
        a, b = st.columns(2)
        a.plotly_chart(
            charts.image(spectrum / (spectrum.max() + 1e-12), "Kept frequencies (log magnitude)"), width="stretch"
        )
        b.plotly_chart(charts.image(filtered, f"{mode.title()}-pass result"), width="stretch")

# ------------------------------------------------------------ math
with tab_math:
    st.markdown((ROOT / "app" / "static" / "math.md").read_text(encoding="utf-8"))

st.caption(
    f"Brainiac © {time.strftime('%Y')} Alejandro Ines · NumPy, SciPy, Plotly & Streamlit · "
    "[source on GitHub](https://github.com/manofmanana/brainiac-mathematical-data-analysis)"
)
