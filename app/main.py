"""app/main.py
Clean, consolidated Streamlit entrypoint for Brainiac.

This version keeps a Brainiac aesthetic but avoids duplicated imports and
imports heavy modules lazily inside UI callbacks to reduce startup issues.
"""

import os
import sys
from pathlib import Path
import base64
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to silence noisy urllib3 / LibreSSL warnings (non-fatal);
# keep this before importing heavy network/data libraries where possible.
warnings.filterwarnings("ignore", message=r".*LibreSSL.*")
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Page config must run early
st.set_page_config(page_title="Brainiac: a mathematical neural network", layout="wide")


def load_css():
    css_path = ROOT / "app" / "static" / "style.css"
    if css_path.exists():
        try:
            st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
            return
        except Exception:
            pass
    # Minimal fallback styling
    st.markdown(
        """
        <style>
        :root {
          --brainiac-green: #49f3e1;
          --brainiac-purple: #8b00ff;
          --brainiac-text: #b7ffdb;
          --brainiac-bg: #0b0d10;
        }
        body, .stApp, .block-container { background-color: var(--brainiac-bg) !important; color: var(--brainiac-text) !important }
        .brainiac-header { font-weight: 800; font-family: 'Orbitron', sans-serif; text-align: center; color: var(--brainiac-green) !important; font-size:36px }
        .brainiac-sub { text-align:center; color: var(--brainiac-purple) !important; margin-top:4px }
        .glass-card { background: rgba(255,255,255,0.02); padding:12px; border-radius:8px }
        /* Tabs and general UI accents */
        [role="tab"] { color: var(--brainiac-text) !important }
        .stButton>button { background: linear-gradient(90deg, rgba(73,243,225,0.08), rgba(139,0,255,0.06)); color: var(--brainiac-text) !important }
        .stSidebar { background: rgba(11,13,16,0.95) }
        a { color: var(--brainiac-green) }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_background_image(image_path: str):
    p = Path(image_path)
    if not p.exists():
        return
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode()
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        st.markdown(
            f"<style>html, body, [data-testid=\"stAppViewContainer\"] {{ background: url(data:{mime};base64,{b64}) center/cover no-repeat; background-attachment: fixed; }}</style>",
            unsafe_allow_html=True,
        )
    except Exception:
        # don't crash the app for background failures
        pass


def ensure_input_dim(layers_list, Xarr):
    input_dim = Xarr.shape[0]
    if int(layers_list[0]) != int(input_dim):
        st.sidebar.warning(f"Adjusting input layer from {layers_list[0]} to match data ({input_dim})")
        layers_list[0] = int(input_dim)
    return layers_list


def brainiac_lab_art():
    # Create SVD/PCA/FFT based art and return a matplotlib Figure
    A = np.random.randn(128, 128)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    img_svd = (U @ np.diag(S) @ Vt)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # Use np.ptp for NumPy 2.0 compatibility
    ax[0].imshow((img_svd - img_svd.min()) / np.ptp(img_svd), cmap="plasma")
    ax[0].set_title("SVD Image")
    comps = U[:, :3] @ np.diag(S[:3])
    ax[1].imshow((comps - comps.min()) / np.ptp(comps), cmap="magma")
    ax[1].set_title("PCA-ish")
    F = np.fft.fftshift(np.fft.fft2(A))
    ax[2].imshow(np.log1p(np.abs(F)), cmap="inferno")
    ax[2].set_title("FFT Spectrogram")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    return fig


def main():
    load_css()

    # Prefer the explicit workspace background image if it exists
    preferred_bg = "/Users/alejandroines/mathematical-mind/space.jpg"
    if Path(preferred_bg).exists():
        set_background_image(preferred_bg)

    # Header (centered, neon styling handled via CSS)
    logo = ROOT / "app" / "static" / "logo.png"
    header_html = ""
    if logo.exists():
        # Inline styles here to force color/centering over Streamlit defaults
        header_html = f"""
        <div style='display:flex;align-items:center;justify-content:center;gap:20px;'>
            <h1 class='brainiac-header' style='color:#00FF88 !important; text-align:center !important; font-size:40px !important; font-weight:900 !important; font-style:italic !important; font-family: Georgia, "Times New Roman", serif !important; margin:0;'>Brainiac: Mathematical Neural Network</h1>
            <img src='data:image/png;base64,{base64.b64encode(logo.read_bytes()).decode()}' style='width:72px;height:auto;border-radius:8px;opacity:0.95'>
        </div>
        """
    else:
        header_html = """
        <div style='text-align:center;'>
            <h1 class='brainiac-header' style='color:#00FF88 !important; text-align:center !important; font-size:40px !important; font-weight:900 !important; font-style:italic !important; font-family: Georgia, "Times New Roman", serif !important; margin:0;'>Brainiac: Mathematical Neural Network</h1>
        </div>
        """
    st.markdown(header_html, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Experiment Controls")
    data_source = st.sidebar.radio("Data Source", ["Iris Dataset", "CoinGecko: Bitcoin"], index=0)
    layers_input = st.sidebar.text_input("Layer sizes (comma-separated)", "4,8,4,1")
    activation = st.sidebar.selectbox("Activation", ["relu", "sigmoid", "gelu"], index=0)
    init_scheme = st.sidebar.selectbox("Init Scheme", ["he", "xavier", "lecun", "normal", "uniform"], index=0)
    lr = st.sidebar.number_input("Learning rate", min_value=1e-6, value=0.01, format="%.6f")
    epochs = st.sidebar.slider("Epochs", 1, 200, 50)
    show_training = st.sidebar.checkbox("Show training progress", value=True)

    bg_choice = st.sidebar.selectbox("Background", ["Grey", "Workspace space.jpg", "None"], index=0)
    if bg_choice == "Workspace space.jpg":
        set_background_image(str(ROOT / "space.jpg"))
    elif bg_choice == "None":
        st.markdown("<style>body { background-color: #ffffff; }</style>", unsafe_allow_html=True)

    st.markdown("---")

    # Parse layers
    try:
        layers = [int(x.strip()) for x in layers_input.split(",") if x.strip()]
    except Exception:
        st.sidebar.error("Invalid layer sizes")
        layers = [4, 8, 4, 1]

    tabs = st.tabs(["Train & Diagnose", "Experiments", "Brainiac Lab"])

    with tabs[0]:
        st.header("Train & Diagnose")
        # Load data lazily
        if data_source.startswith("CoinGecko"):
            st.info("Fetching recent Bitcoin prices from CoinGecko...")
            try:
                import pandas as pd
                import requests

                resp = requests.get(
                    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                    params={"vs_currency": "usd", "days": "90"}, timeout=10,
                )
                prices = resp.json().get("prices", [])
                if not prices:
                    st.error("No price data returned")
                    st.stop()
                df = pd.DataFrame(prices, columns=["timestamp", "price"]).assign(
                    timestamp=lambda d: pd.to_datetime(d["timestamp"], unit="ms")
                )
                # Use numpy.ptp on the underlying array to avoid pandas.Series.ptp missing
                price_arr = df["price"].to_numpy()
                denom = np.ptp(price_arr)
                if denom == 0:
                    df["price_norm"] = 0.0
                else:
                    df["price_norm"] = (price_arr - price_arr.min()) / denom
                X = np.linspace(0, 1, len(df)).reshape(1, -1)
                y = df["price_norm"].to_numpy().reshape(1, -1)
                st.line_chart(df.set_index("timestamp")["price_norm"])
                data_label = "Bitcoin Normalized Price"
            except Exception as e:
                st.error(f"Failed to fetch CoinGecko data: {e}")
                st.stop()
        else:
            st.info("Using Iris dataset (local)")
            try:
                import pandas as pd
                import seaborn as sns

                iris = sns.load_dataset("iris")
                df = iris.rename(columns={"species": "label"})
                X = df.iloc[:, :-1].to_numpy().T
                y_raw = df["label"].astype("category").cat.codes.to_numpy()
                y = y_raw.reshape(1, -1) / y_raw.max()
                st.dataframe(df.head())
                data_label = "Iris Features (normalized)"
            except Exception as e:
                st.error(f"Failed to load Iris dataset: {e}")
                st.stop()

        st.write(f"### Training on: {data_label}")
        layers = ensure_input_dim(layers, X)

        if st.button("Start Training"):
            # Lazy imports of project modules
            from core.network import NeuralNetwork
            from utils.plotting import plot_loss, plot_gradient_heatmap, plot_3d_loss_surface

            model = NeuralNetwork(layer_sizes=layers, activation=activation, init_scheme=init_scheme)

            progress = st.progress(0)

            def cb(epoch, loss):
                if show_training and epoch % max(1, epochs // 50) == 0:
                    progress.progress(min(1.0, epoch / max(1, epochs)))

            model.train(X, y, lr=lr, epochs=epochs, verbose=False, callback=cb)
            progress.progress(1.0)
            st.success(f"Training complete — final loss: {model.loss_history[-1]:.6f}")

            st.subheader("Loss curve")
            st.pyplot(plot_loss(model.loss_history))

            st.subheader("Gradient norms (per epoch)")
            st.pyplot(plot_gradient_heatmap(model.grad_norms_per_epoch))

            st.subheader("Actual vs Predicted")
            preds = model.predict(X)
            fig2, ax = plt.subplots(figsize=(8, 4))
            if data_source.startswith("CoinGecko"):
                ax.plot(df["timestamp"], y.flatten(), label="Actual", color="#00ff88")
                ax.plot(df["timestamp"], preds.flatten(), label="Predicted", color="#8b00ff")
            else:
                ax.scatter(range(len(y.flatten())), y.flatten(), label="Actual", color="#00ff88")
                ax.scatter(range(len(preds.flatten())), preds.flatten(), label="Predicted", color="#8b00ff", alpha=0.6)
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig2)

            # 3D loss surface (if available)
            try:
                dense_layer = next((L for L in model.layers if hasattr(L, "W")), None)
                if dense_layer is not None:
                    st.subheader("3D loss surface (first Dense layer)")
                    st.pyplot(plot_3d_loss_surface(dense_layer, X, y, weight_idx_pair=(0, 1), grid_scale=0.5))
            except Exception:
                pass

    with tabs[1]:
        st.header("Experiments")
        st.write("Run convergence-speed comparisons across activations and init schemes")
        experiment_activations = [activation]
        experiment_inits = [init_scheme]
        if st.button("Run Experiments"):
            from core.network import run_convergence_experiments
            from utils.plotting import plot_convergence_comparison

            configs = [(a, i) for a in experiment_activations for i in experiment_inits]
            layers_exp = ensure_input_dim(layers.copy(), X)
            results = run_convergence_experiments(X, y, configs, layer_sizes=layers_exp, lr=lr, epochs=epochs)
            st.pyplot(plot_convergence_comparison(results))
            # summary
            import pandas as pd

            summary = {name: res["loss_history"][-1] if res["loss_history"] else None for name, res in results.items()}
            st.write(pd.DataFrame.from_dict(summary, orient="index", columns=["Final Loss"]))

    with tabs[2]:
        st.header("Brainiac Lab")
        st.write("Generate abstract art using linear algebra and signal transforms")
        if st.button("Generate Art"):
            fig = brainiac_lab_art()
            st.pyplot(fig)

    st.caption("Brainiac © 2025 — Live data optional. Built with NumPy, Streamlit, and pure math.")


if __name__ == "__main__":
    main()
