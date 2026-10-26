"""Tools of the notebook "Training Basics": MLP, Fourier features, training loop, images.

Author: Nassim Razaaly (nassim.razaaly@ensma.fr)
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.MSELoss()


# ============================================================ models
# Personalized class "module" with nice print/show
class modulePerso(nn.Module):
    def __init__(self):
      super().__init__()

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"{super().__repr__()}\n--> {self.n_params():,} parameters "

    # This method is for illustration purpose
    def show(self, max_nodes=8, path=None):
        mods = list(self.net)
        lin = [i for i, m in enumerate(mods) if isinstance(m, nn.Linear)]
        L = [mods[lin[0]].in_features] + [mods[i].out_features for i in lin]
        acts = [""] + [type(mods[i + 1]).__name__
                       if i + 1 < len(mods) and not isinstance(mods[i + 1], nn.Linear) else ""
                       for i in lin]
        n = len(L)

        def slots(k):  # neurones dessinés, None = "⋮"
            if k <= max_nodes:
                return list(range(k))
            t = max_nodes // 2
            return list(range(t)) + [None] + list(range(k - (max_nodes - t - 1), k))

        cols = [[(i, (len(s) - 1) / 2 - j, v) for j, v in enumerate(s)]
                for i, s in enumerate(map(slots, L))]

        fig, ax = plt.subplots(figsize=(max(6, 2.2 * n), 5.5))
        for a, b in zip(cols, cols[1:]):
            for xa, ya, va in a:
                for xb, yb, vb in b:
                    if va is not None and vb is not None:
                        ax.plot([xa, xb], [ya, yb], c="grey", lw=.5, alpha=.35, zorder=1)

        colors = ["#4CAF50"] + ["#2196F3"] * (n - 2) + ["#FF9800"]
        for i, col in enumerate(cols):
            for x, y, v in col:
                if v is None:
                    ax.text(x, y, "⋮", ha="center", va="center", fontsize=20)
                else:
                    ax.scatter(x, y, s=350, color=colors[i], edgecolor="k", zorder=3)

        names = ["Input"] + [f"Hidden {i}" for i in range(1, n - 1)] + ["Output"]
        lo, hi = -max_nodes / 2 - .6, max_nodes / 2 + .3
        for i, (name, k, act) in enumerate(zip(names, L, acts)):
            ax.text(i, lo, f"{name}\n{k}" + (f"\n{act}" if act else ""), ha="center", va="top")
        for i in range(n - 1):
            ax.text(i + .5, hi, f"{L[i] * L[i + 1] + L[i + 1]:,} params",
                    ha="center", fontsize=9, color="dimgrey")

        ax.set_title(f"MLP {L} — {self.n_params():,} paramètres")
        ax.set_xlim(-.6, n - .4)
        ax.set_ylim(lo - 1.6, hi + .6)
        ax.axis("off")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()


class MLP(modulePerso):
    def __init__(self, layers, activation, last_activation=None):
        super().__init__()
        blocks = []
        for n_in, n_out in zip(layers[:-2], layers[1:-1]):      # hidden layers
            blocks += [nn.Linear(n_in, n_out), activation()]
        blocks.append(nn.Linear(layers[-2], layers[-1]))        # output layer
        if last_activation is not None:
            blocks.append(last_activation())
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class FourierFeatures(nn.Module):
    def __init__(self, n_in, n_freq, sigma, trainable=False):
        super().__init__()
        self.B = nn.Parameter(sigma * torch.randn(n_freq, n_in), requires_grad=trainable) # (n_freq, n_in) Matrix

    def forward(self, v):                                     # (N, n_in) -> (N, 2 * n_freq)
        z = 2 * torch.pi * v @ self.B.T                       # (N, n_in) x (n_in, n_freq) -> (N, n_freq)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=1) # concat (N, n_freq) and (N, n_freq) into (N, 2*n_freq)


class MLP_FF(MLP):
    """Fourier features (n_in -> 2*n_freq) followed by MLP. `layers` = layers after FF."""
    def __init__(self, n_in, n_freq, sigma, layers, activation, last_activation=None, trainable=False):
        super().__init__([2 * n_freq] + list(layers), activation, last_activation)
        self.ff = FourierFeatures(n_in, n_freq, sigma, trainable)

    def forward(self, x):
        return self.net(self.ff(x))

    def _columns(self):
        cols, edges = super()._columns()
        n_freq, n_in = self.ff.B.shape
        cols[0] = [2 * n_freq, "Fourier", f"cos/sin\nσ={self.ff.sigma}", "#9C27B0"]
        cols.insert(0, [n_in, "Input", "", "#4CAF50"])
        etat = "entraînable" if self.ff.B.requires_grad else "fixe"
        edges.insert(0, f"B : {self.ff.B.numel():,} ({etat})")
        return cols, edges


# ============================================================ training
def train(model, optimizer, data, n_epochs, n_batch, plot_every=100, scheduler=None, snapshot=None):
    """Mini-batch training; tracks the training and validation MSE, with a live plot every `plot_every` epochs.
    hist: losses per epoch, best validation epoch, training time, snapshots (outputs of `snapshot()`)."""
    X, Y = data["train"]
    X_val, Y_val = data["val"]
    hist = {"train": [], "val": [], "snapshots": []}
    fig, ax = plt.subplots(figsize=(6, 3)); live = display(fig, display_id=True); plt.close(fig)
    t0 = time.perf_counter()

    for epoch in tqdm(range(n_epochs)):
        # 1. training, by mini-batches
        model.train()
        for idx in torch.randperm(len(X), device=X.device).chunk(n_batch):
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 2. monitoring
        model.eval()
        with torch.no_grad():
            hist["train"].append(loss_fn(model(X), Y).item())
            hist["val"].append(loss_fn(model(X_val), Y_val).item())

        # 3. live plot
        if (epoch + 1) % plot_every == 0 or epoch == n_epochs - 1:
            if snapshot is not None:
                hist["snapshots"].append(snapshot())
            ax.clear()
            ax.semilogy(hist["train"], label="train"); ax.semilogy(hist["val"], "--", label="validation")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()
            live.update(fig)

    hist["best_epoch"] = int(np.argmin(hist["val"])) + 1
    hist["time"] = time.perf_counter() - t0
    return hist


# ============================================================ images
def load_image(path, size=128, gray=False):
    raw = plt.imread(path)
    im = torch.tensor(raw, dtype=torch.float32)
    if raw.dtype == np.uint8:
        im = im / 255                                         # [0, 255] -> [0, 1]
    if im.ndim == 2:
        im = im[..., None]                                    # (H, W) -> (H, W, 1)
    if im.shape[2] == 4:                                      # transparency -> white background
        im = im[..., :3] * im[..., 3:] + 1 - im[..., 3:]
    H, W = im.shape[:2]; s = min(H, W)                        # central square
    im = im[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
    im = F.interpolate(im.permute(2, 0, 1)[None], size=(size, size), mode="area")[0].permute(1, 2, 0)
    if gray:
        im = im.mean(dim=2, keepdim=True)
    return im.squeeze(2)                                      # (n, n) if gray, (n, n, 3) if color


def show_images(images, titles, size=3):
    fig, ax = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for a, im, t in zip(np.atleast_1d(ax), images, titles):
        a.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.show()


def image_data(img, train=0.6, val=0.2, seed=0):
    n = img.shape[0]
    t = torch.linspace(-1, 1, n)
    yy, xx = torch.meshgrid(t, t, indexing="ij")             # y: rows, x: columns
    X = torch.stack([xx.flatten(), yy.flatten()], dim=1)      # (n*n, 2)
    Y = img.reshape(n * n, -1)                                # (n*n, 1) or (n*n, 3)
    perm = torch.randperm(n * n, generator=torch.Generator().manual_seed(seed))
    n_tr, n_va = int(train * n * n), int(val * n * n)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    data = {k: (X[i].to(device), Y[i].to(device)) for k, i in idx.items()}
    return data, X.to(device)                                 # X_all: all pixels, to draw the image


def predict_image(model, X_all, shape):
    model.eval()
    with torch.no_grad():
        return model(X_all).cpu().numpy().reshape(shape)


def psnr(mse):                                                # peak signal-to-noise ratio, in dB (peak = 1)
    return 10 * np.log10(1 / mse)


def animateImage(history):
    frames = history["snapshots"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5)); ax.axis("off")
    im_plot = ax.imshow(np.clip(frames[0], 0, 1), cmap="gray", vmin=0, vmax=1)
    def update(k):
        im_plot.set_data(np.clip(frames[k], 0, 1)); ax.set_title(f"snapshot {k + 1}/{len(frames)}")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=150)
    plt.close(fig)
    return HTML(anim.to_jshtml())
