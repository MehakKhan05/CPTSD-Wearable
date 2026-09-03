"""
WESAD stress-detection SNN — Phase 1 (algorithm layer)
--------------------------------------------------------
Preprocesses WESAD chest-worn ECG + EDA (GSR) signals, trains a small
LIF-based spiking neural network in snnTorch with hardware-aware
(quantized) weights, and exports the trained weights + LIF time
constants for use in the analog crossbar cell design.

Expected WESAD layout (from the official release):
    WESAD/
      S2/S2.pkl
      S3/S3.pkl
      ...
Each .pkl contains a dict with:
    data['signal']['chest']['ECG']  -> (N, 1) at 700 Hz
    data['signal']['chest']['EDA']  -> (N, 1) at 700 Hz
    data['label']                   -> (N,) per-sample condition label
        1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, ...
        (0/5/6/7 = transient/undefined, dropped)

Run:
    python wesad_snn_train.py --wesad_dir /path/to/WESAD --subjects S2 S3 S4
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader, random_split

# ----------------------------------------------------------------------
# 1. Data loading + windowing
# ----------------------------------------------------------------------

FS = 700  # WESAD chest sensor sample rate (Hz)
WINDOW_SEC = 10  # 10s windows are standard in WESAD stress-detection literature
WINDOW = FS * WINDOW_SEC
STEP = WINDOW // 2  # 50% overlap between windows


def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg = data["signal"]["chest"]["ECG"].squeeze()
    eda = data["signal"]["chest"]["EDA"].squeeze()
    label = data["label"].squeeze()
    return ecg, eda, label


def windows_from_subject(ecg, eda, label):
    """Slice into fixed windows, label = majority class in window.
    Keep only baseline (1) vs stress (2) windows -> binary task."""
    X, y = [], []
    for start in range(0, len(label) - WINDOW, STEP):
        end = start + WINDOW
        win_label = label[start:end]
        vals, counts = np.unique(win_label, return_counts=True)
        majority = vals[np.argmax(counts)]
        if majority not in (1, 2):  # only baseline / stress windows
            continue
        ecg_win = ecg[start:end]
        eda_win = eda[start:end]
        # z-score normalize per-window (removes subject/session offset)
        ecg_win = (ecg_win - ecg_win.mean()) / (ecg_win.std() + 1e-8)
        eda_win = (eda_win - eda_win.mean()) / (eda_win.std() + 1e-8)
        X.append(np.stack([ecg_win, eda_win], axis=0))  # (2, WINDOW)
        y.append(0 if majority == 1 else 1)  # 0=baseline, 1=stress
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def rate_encode(x, num_steps, gain=1.0):
    """Convert continuous windowed signal into spike trains via rate coding.
    x: (batch, channels, window) already z-scored -> squashed to [0,1] via sigmoid,
    then treated as per-timestep firing probability, downsampled to num_steps."""
    # Downsample window -> num_steps by simple average pooling
    b, c, w = x.shape
    pool = w // num_steps
    x = x[:, :, : pool * num_steps].reshape(b, c, num_steps, pool).mean(axis=3)
    prob = torch.sigmoid(torch.tensor(x) * gain)  # (batch, channels, num_steps)
    spikes = torch.bernoulli(prob)
    return spikes  # (batch, channels, num_steps)


class WESADWindows(Dataset):
    def __init__(self, X, y, num_steps=100):
        self.X = X
        self.y = y
        self.num_steps = num_steps

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spikes = rate_encode(self.X[idx : idx + 1], self.num_steps).squeeze(0)
        return spikes, self.y[idx]  # spikes: (channels, num_steps)


# ----------------------------------------------------------------------
# 2. Hardware-aware (quantized) linear layer
# ----------------------------------------------------------------------


class QuantLinear(nn.Module):
    """Linear layer with weights fake-quantized to n_bits during forward pass
    (straight-through estimator for the backward pass), so the network learns
    weights that will survive quantization for the analog mapping later."""

    def __init__(self, in_features, out_features, n_bits=4, w_range=1.0):
        super().__init__()
        # Init scale increased from the original 0.1: with only 2 input
        # channels and sparse binary spike inputs, weights that small never
        # drive the membrane potential past the LIF firing threshold, so the
        # network never spikes and never learns (loss sits at exactly ln(2)
        # forever, which is what "never fires" looks like under
        # CrossEntropyLoss on an all-zero output).
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.3)
        self.n_bits = n_bits
        self.w_range = w_range  # weights clamped to [-w_range, +w_range]

    def quantized_weight(self):
        levels = 2**self.n_bits - 1
        w = torch.clamp(self.weight, -self.w_range, self.w_range)
        w_q = torch.round((w + self.w_range) / (2 * self.w_range) * levels)
        w_q = w_q / levels * (2 * self.w_range) - self.w_range
        # straight-through estimator: forward uses w_q, backward uses w's gradient
        return w + (w_q - w).detach()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.quantized_weight())


# ----------------------------------------------------------------------
# 3. LIF-based SNN
# ----------------------------------------------------------------------


class StressSNN(nn.Module):
    def __init__(self, in_channels=2, hidden=32, out_classes=2, n_bits=4, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = QuantLinear(in_channels, hidden, n_bits=n_bits)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, threshold=0.6)
        self.fc2 = QuantLinear(hidden, out_classes, n_bits=n_bits)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, threshold=0.6)

    def forward(self, x):
        # x: (batch, channels, num_steps)
        batch, ch, num_steps = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for t in range(num_steps):
            cur1 = self.fc1(x[:, :, t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)  # (num_steps, batch, out_classes)


# ----------------------------------------------------------------------
# 4. Training loop
# ----------------------------------------------------------------------


def train(args):
    all_X, all_y = [], []
    for subj in args.subjects:
        ecg, eda, label = load_subject(f"{args.wesad_dir}/{subj}/{subj}.pkl")
        X, y = windows_from_subject(ecg, eda, label)
        all_X.append(X)
        all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total windows: {len(y)}  (stress: {y.sum()}, baseline: {(y==0).sum()})")

    dataset = WESADWindows(X, y, num_steps=args.num_steps)
    n_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = StressSNN(n_bits=args.n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Class-weighted loss: without this, the optimizer finds "always predict
    # baseline" (the majority class) as an easy local minimum and never
    # learns to separate the classes. Weight is inversely proportional to
    # class frequency so both classes contribute equally to the loss.
    class_counts = np.bincount(y)
    class_weights = torch.tensor(
        len(y) / (len(class_counts) * class_counts), dtype=torch.float32
    )
    print(f"Class counts: {class_counts}  ->  loss weights: {class_weights}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(args.epochs):
        model.train()
        for spikes, labels in train_loader:
            spk_out = model(spikes)  # (num_steps, batch, classes)
            out_rate = spk_out.mean(dim=0)  # rate-decode: mean firing rate per class
            loss = loss_fn(out_rate, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        fire_rate = spk_out.mean().item()  # avg output firing rate, last batch

        # validation
        model.eval()
        correct, total = 0, 0
        pred_counts = {0: 0, 1: 0}
        with torch.no_grad():
            for spikes, labels in val_loader:
                spk_out = model(spikes)
                pred = spk_out.mean(dim=0).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                for p in pred.tolist():
                    pred_counts[p] += 1
        print(
            f"Epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}  "
            f"val_acc={correct/total:.3f}  preds={pred_counts}  "
            f"out_fire_rate={fire_rate:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Export trained weights + LIF time constants (beta) for the
    #    analog crossbar mapping (this is what feeds your Xschem LUT)
    # ------------------------------------------------------------------
    export = {
        "fc1_weight_quantized": model.fc1.quantized_weight().detach().numpy(),
        "fc2_weight_quantized": model.fc2.quantized_weight().detach().numpy(),
        "lif1_beta": model.lif1.beta.detach().numpy(),
        "lif2_beta": model.lif2.beta.detach().numpy(),
        "n_bits": args.n_bits,
        "w_range": model.fc1.w_range,
    }
    np.savez("trained_weights.npz", **export)
    print("\nExported trained_weights.npz")
    print("fc1 weights (channel->hidden):\n", export["fc1_weight_quantized"])
    print("\nThese are your real quantized weight values for the crossbar LUT.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad_dir", type=str, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S2", "S3", "S4"])
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n_bits", type=int, default=4)
    args = parser.parse_args()
    train(args)