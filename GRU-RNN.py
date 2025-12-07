#!/usr/bin/env python3


import os
import argparse
import random
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# CLI / defaults
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="railway_realistic_dataset", help="dataset folder")
    p.add_argument("--out", type=str, default="rnn_output", help="output folder")
    p.add_argument("--sample-rate", type=int, default=1000)
    p.add_argument("--past", type=int, default=128)
    p.add_argument("--future", type=int, default=32)
    p.add_argument("--hop", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--max-per-file", type=int, default=50)
    p.add_argument("--fast-mode", action="store_true", help="even subsample windows per file")
    p.add_argument("--fft-n", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0, help="MAE z weight")
    p.add_argument("--beta", type=float, default=1.0, help="SDF z weight")
    p.add_argument("--k", type=float, default=3.0, help="threshold k for mean+ k*std")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="very quick debug run (fewer windows, fewer epochs)")
    p.add_argument("--save-top", type=int, default=10, help="how many top anomalous windows to save")
    p.add_argument("--spectral-loss-weight", type=float, default=0.02, help="weight for small spectral magnitude loss")
    p.add_argument("--force-threshold", type=float, default=2.23, help="If set, force CAS detection threshold to this value (useful for demos)")
    return p.parse_args()

args = parse_args()

DATASET_PATH = args.data
OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = args.sample_rate
PAST_SAMPLES = args.past
FUTURE_SAMPLES = args.future
HOP = args.hop

BATCH_SIZE = args.batch
EPOCHS = 6 if args.quick else args.epochs
LR = args.lr
HIDDEN_SIZE = args.hidden
NUM_LAYERS = args.layers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FFT_N = args.fft_n
ALPHA = args.alpha
BETA = args.beta
THRESH_K = args.k

MAX_PER_FILE = max(1, args.max_per_file // (4 if args.quick else 1))  # reduce for quick
FAST_MODE = args.fast_mode

NUM_WORKERS = args.num_workers
PIN_MEMORY = False

SAVE_TOP = args.save_top
SPECTRAL_LOSS_WEIGHT = args.spectral_loss_weight

# Force threshold override (None to disable)
THRESH_FORCE = float(args.force_threshold) if args.force_threshold is not None else None

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------------
# Utils
# -------------------------
def find_audio_files(base_path):
    exts = ("*.wav", "*.flac", "*.ogg", "*.mp3")
    files = []
    for ext in exts:
        files.extend(Path(base_path).rglob(ext))
    return sorted([str(p) for p in files])

def load_audio_file(path, sr=SAMPLE_RATE):
    try:
        y, fs = librosa.load(path, sr=None, mono=True)
        if y is None or len(y) == 0:
            return None
        if fs != sr:
            y = librosa.resample(y, orig_sr=fs, target_sr=sr)
        std = np.std(y)
        if std > 1e-9:
            y = (y - np.mean(y)) / (std + 1e-9)
        return y.astype(np.float32)
    except Exception as e:
        print("[load_audio_file] error", path, e)
        return None

def build_sequences_from_audio(wave, past_len=PAST_SAMPLES, future_len=FUTURE_SAMPLES, hop=HOP):
    sequences = []
    total = past_len + future_len
    if wave.shape[0] < total:
        return sequences
    for start in range(0, len(wave) - total + 1, hop):
        past = wave[start:start+past_len]
        future = wave[start+past_len:start+past_len+future_len]
        sequences.append((past.copy(), future.copy()))
    return sequences

# -------------------------
# Dataset
# -------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.seq = sequences
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        past, future = self.seq[idx]
        return torch.from_numpy(past).float(), torch.from_numpy(future).float()

def prepare_data(dataset_path):
    files = find_audio_files(dataset_path)
    print(f"[prepare_data] found {len(files)} files")
    sequences = []
    files_with_seqs = 0
    for p in tqdm(files, desc="loading"):
        w = load_audio_file(p)
        if w is None: continue
        seqs = build_sequences_from_audio(w)
        if not seqs: continue
        files_with_seqs += 1
        if MAX_PER_FILE <= 0:
            sequences.extend(seqs); continue
        if len(seqs) <= MAX_PER_FILE:
            sequences.extend(seqs); continue
        if FAST_MODE:
            stride = int(np.ceil(len(seqs) / MAX_PER_FILE))
            sampled = seqs[::stride][:MAX_PER_FILE]
            sequences.extend(sampled)
        else:
            sampled = random.sample(seqs, MAX_PER_FILE)
            sequences.extend(sampled)
    random.shuffle(sequences)
    print(f"[prepare_data] files with seqs {files_with_seqs}/{len(files)}  total windows {len(sequences)}")
    return sequences

# -------------------------
# Model
# -------------------------
class GRUPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, future_len=FUTURE_SAMPLES):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_size//2), future_len)
        )
    def forward(self, past):
        x = past.unsqueeze(-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        fut = self.fc(last)
        return fut

# -------------------------
# Loss helpers (MAE + small spectral)
# -------------------------
l1_loss = nn.L1Loss()

def spectral_mag_loss(preds, targets, fft_n=FFT_N):
    # preds, targets: (B, F) numpy arrays
    B = preds.shape[0]
    loss = 0.0
    for i in range(B):
        P = np.fft.rfft(preds[i], n=fft_n)
        A = np.fft.rfft(targets[i], n=fft_n)
        magP = np.abs(P)
        magA = np.abs(A)
        loss += np.mean((magP - magA)**2)
    return loss / float(B)

def compute_sdf_batch(preds, actuals, fft_n=FFT_N):
    B, F = preds.shape
    sdf_vals = np.zeros(B, dtype=np.float32)
    for i in range(B):
        p = preds[i]; a = actuals[i]
        P = np.fft.rfft(p, n=fft_n); A = np.fft.rfft(a, n=fft_n)
        magP = np.abs(P); magA = np.abs(A)
        sumP = magP.sum(); sumA = magA.sum()
        magP = (magP / (sumP + 1e-12)) if sumP > 0 else np.ones_like(magP)/len(magP)
        magA = (magA / (sumA + 1e-12)) if sumA > 0 else np.ones_like(magA)/len(magA)
        sdf_vals[i] = float(np.linalg.norm(magP - magA, ord=2))
    return sdf_vals

def compute_scores_on_loader(model, loader, fft_n=FFT_N):
    model.eval()
    maes = []; sdfs = []
    with torch.no_grad():
        for past, fut in loader:
            past = past.to(DEVICE); fut = fut.to(DEVICE)
            pred = model(past).cpu().numpy()
            fut_np = fut.cpu().numpy()
            mae_batch = np.mean(np.abs(pred - fut_np), axis=1)
            sdf_batch = compute_sdf_batch(pred, fut_np, fft_n=fft_n)
            maes.append(mae_batch); sdfs.append(sdf_batch)
    if len(maes) == 0: return np.array([]), np.array([])
    return np.concatenate(maes, axis=0), np.concatenate(sdfs, axis=0)

# -------------------------
# Training
# -------------------------
def train_model(model, train_loader, val_loader=None, epochs=EPOCHS):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    stats = {"train_loss": [], "val_loss": []}
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0; n = 0
        for past, fut in train_loader:
            past = past.to(DEVICE); fut = fut.to(DEVICE)
            pred = model(past)
            loss_l1 = l1_loss(pred, fut)
            # small spectral loss computed on CPU numpy to avoid heavy GPU ops; ok for small batches
            if SPECTRAL_LOSS_WEIGHT > 0:
                pred_np = pred.detach().cpu().numpy()
                fut_np = fut.detach().cpu().numpy()
                spec_l = spectral_mag_loss(pred_np, fut_np, fft_n=FFT_N)
                loss = loss_l1 + SPECTRAL_LOSS_WEIGHT * spec_l
            else:
                loss = loss_l1
            opt.zero_grad()
            # if spectral contribution is numpy float, convert:
            if isinstance(loss, float):
                loss = torch.tensor(loss, requires_grad=True).to(DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss_l1.item()) * past.size(0)  # report MAE-only for stability
            n += past.size(0)
        train_loss = running / max(1, n)
        stats["train_loss"].append(train_loss)

        # val
        val_loss = None
        if val_loader:
            model.eval()
            vrunning = 0.0; vn = 0
            with torch.no_grad():
                for past, fut in val_loader:
                    past = past.to(DEVICE); fut = fut.to(DEVICE)
                    pred = model(past)
                    l = float(l1_loss(pred, fut).item())
                    vrunning += l * past.size(0)
                    vn += past.size(0)
            val_loss = vrunning / max(1, vn)
            stats["val_loss"].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        print(f"[train] Epoch {ep}/{epochs} - train_mae: {train_loss:.6f}" + (f" - val_mae: {val_loss:.6f}" if val_loss is not None else ""))
    return stats

# -------------------------
# plotting & top-window saving
# -------------------------
def save_top_windows(an_wave, positions, cas_scores, topk=SAVE_TOP, prefix="top"):
    if len(positions) == 0:
        return []
    os.makedirs(os.path.join(OUT_DIR, "top_windows"), exist_ok=True)
    top_idx = np.argsort(-cas_scores)[:topk]
    saved = []
    for i in top_idx:
        s = int(positions[i])
        seg = an_wave[max(0, s-PAST_SAMPLES//2): s+PAST_SAMPLES+FUTURE_SAMPLES + PAST_SAMPLES//2]
        fname = os.path.join(OUT_DIR, "top_windows", f"{prefix}_pos{s}_cas{cas_scores[i]:.3f}.wav")
        sf.write(fname, seg, SAMPLE_RATE)
        saved.append(fname)
    return saved

# -------------------------
# main flow
# -------------------------
def main():
    sequences = prepare_data(DATASET_PATH)
    if len(sequences) == 0:
        print("No sequences found. Check DATASET_PATH.")
        return

    # split
    train_seqs, test_seqs = train_test_split(sequences, test_size=0.15, random_state=42) if len(sequences) > 1 else (sequences, [])
    train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.12, random_state=42) if len(train_seqs) > 1 else (train_seqs, [])
    print("Train/Val/Test:", len(train_seqs), len(val_seqs), len(test_seqs))

    # data loaders
    train_ds = SequenceDataset(train_seqs)
    val_ds = SequenceDataset(val_seqs)
    test_ds = SequenceDataset(test_seqs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b),
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b),
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b),
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = GRUPredictor(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, future_len=FUTURE_SAMPLES)
    print("[main] params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # train
    stats = train_model(model, train_loader, val_loader if len(val_ds)>0 else None, epochs=EPOCHS)

    # compute train distributions
    train_maes, train_sdfs = compute_scores_on_loader(model, train_loader, fft_n=FFT_N)
    if train_maes.size == 0:
        print("No train scores; aborting metric steps.")
        train_maes = np.array([]); train_sdfs = np.array([])
    else:
        ma_mean, ma_std = float(np.mean(train_maes)), float(np.std(train_maes))
        sd_mean, sd_std = float(np.mean(train_sdfs)), float(np.std(train_sdfs))
        print(f"[main] train mae mean/std {ma_mean:.6f}/{ma_std:.6f}, sdf mean/std {sd_mean:.6f}/{sd_std:.6f}")

    # combined CAS on test (or train if no test)
    eval_loader = test_loader if len(test_ds) > 0 else train_loader
    test_maes, test_sdfs = compute_scores_on_loader(model, eval_loader, fft_n=FFT_N)
    if test_maes.size == 0:
        print("[main] no eval scores")
        return

    # z-score using train stats (if absent use test stats)
    if train_maes.size == 0:
        ma_mean, ma_std = float(np.mean(test_maes)), float(np.std(test_maes))
        sd_mean, sd_std = float(np.mean(test_sdfs)), float(np.std(test_sdfs))

    test_ma_z = (test_maes - ma_mean) / (ma_std + 1e-12)
    test_sd_z = (test_sdfs - sd_mean) / (sd_std + 1e-12)
    test_cas = ALPHA * test_ma_z + BETA * test_sd_z

    # thresholds (computed from training CAS)
    train_cas = (ALPHA * ((train_maes - ma_mean)/(ma_std+1e-12)) + BETA * ((train_sdfs - sd_mean)/(sd_std+1e-12))) if train_maes.size>0 else test_cas
    thr_mean3 = float(np.mean(train_cas)) + float(THRESH_K) * float(np.std(train_cas))
    p95 = float(np.percentile(train_cas, 95))
    p99 = float(np.percentile(train_cas, 99))
    p995 = float(np.percentile(train_cas, 99.5))
    print(f"[thresholds] mean+{THRESH_K}std={thr_mean3:.4f}  p95={p95:.4f}  p99={p99:.4f}  p99.5={p995:.4f}")

    # choose threshold(s) to evaluate — if THRESH_FORCE provided, include it
    thr_map = {
        "mean+3std": thr_mean3,
        "p99": p99,
        "p99.5": p995
    }
    if THRESH_FORCE is not None:
        thr_map["forced"] = float(THRESH_FORCE)
        print(f"[info] Using forced threshold THRESH_FORCE={THRESH_FORCE:.4f} for detection counts")

    # if synthetic anomaly injection is to be done, pick an example audio and produce labels
    files = find_audio_files(DATASET_PATH)
    example_wave = None; example_path = None
    for p in files:
        w = load_audio_file(p)
        if w is None: continue
        if len(w) >= PAST_SAMPLES + FUTURE_SAMPLES + 2000:
            example_wave = w; example_path = p; break
    if example_wave is None and files:
        example_path = files[0]; example_wave = load_audio_file(example_path)

    if example_wave is not None:
        # inject synthetic anomaly (spectral_shift) and slide CAS over it
        an_wave = example_wave.copy()
        start_idx = len(an_wave)//3
        end_idx = start_idx + min(len(an_wave)-start_idx-1, 1200)
        t = np.arange(end_idx - start_idx) / float(SAMPLE_RATE)
        an_wave[start_idx:end_idx] += 0.8 * np.sin(2*np.pi*min(200, SAMPLE_RATE//3)*t)
        an_wave = an_wave / (np.max(np.abs(an_wave))+1e-12) * 0.95

        positions = []
        cas_scores = []; ma_scores = []; sd_scores = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(an_wave) - (PAST_SAMPLES + FUTURE_SAMPLES) + 1, HOP):
                past = an_wave[start:start+PAST_SAMPLES]
                future = an_wave[start+PAST_SAMPLES:start+PAST_SAMPLES+FUTURE_SAMPLES]
                past_t = torch.from_numpy(past).unsqueeze(0).to(DEVICE)
                pred = model(past_t).cpu().numpy().squeeze()
                mae = float(np.mean(np.abs(pred - future)))
                sdf = float(compute_sdf_batch(pred[np.newaxis,:], future[np.newaxis,:], fft_n=FFT_N)[0])
                ma_z_val = (mae - ma_mean) / (ma_std + 1e-12)
                sd_z_val = (sdf - sd_mean) / (sd_std + 1e-12)
                cas = ALPHA * ma_z_val + BETA * sd_z_val
                positions.append(start); cas_scores.append(cas); ma_scores.append(mae); sd_scores.append(sdf)
        positions = np.array(positions); cas_scores = np.array(cas_scores)
        ma_scores = np.array(ma_scores); sd_scores = np.array(sd_scores)

        # create labels for injected anomaly (window overlaps injected range)
        labels = np.zeros_like(positions, dtype=int)
        for i, pos in enumerate(positions):
            fut_start = pos + PAST_SAMPLES
            fut_end = fut_start + FUTURE_SAMPLES
            if not (fut_end < start_idx or fut_start > end_idx):
                labels[i] = 1

        # histogram & top windows & PR
        plt.figure(figsize=(6,3))
        plt.hist(cas_scores, bins=80)
        plt.axvline(np.mean(train_cas), color='k', linestyle='--', label='train mean')
        plt.axvline(thr_mean3, color='r', linestyle='--', label='mean+3std')
        plt.axvline(p99, color='m', linestyle=':', label='p99')
        plt.legend(); plt.title("CAS histogram (injected example)"); plt.tight_layout()
        hist_path = os.path.join(OUT_DIR, "cas_histogram.png")
        plt.savefig(hist_path, dpi=150); plt.close()
        print("[saved]", hist_path)

        # PR curve on injected labels
        if labels.sum() > 0:
            precision, recall, thr_pr = precision_recall_curve(labels, cas_scores)
            pr_auc = auc(recall, precision)
            print("[PR] PR-AUC (CAS) on injected example:", pr_auc)
            plt.figure(figsize=(5,4))
            plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve (injected)"); plt.legend(); plt.tight_layout()
            pr_path = os.path.join(OUT_DIR, "pr_curve_injected.png")
            plt.savefig(pr_path, dpi=150); plt.close()
            print("[saved]", pr_path)
        else:
            print("[info] no positive labels found for injected anomaly (weird)")

        # Save top windows for manual inspection
        saved_files = save_top_windows(an_wave, positions, cas_scores, topk=SAVE_TOP, prefix="injected_top")
        print("[saved top windows]", saved_files[:min(10, len(saved_files))])

        # compute detection counts for different thresholds
        for name, thr in thr_map.items():
            preds = (cas_scores > thr).astype(int)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            print(f"[thr {name}] thr={thr:.3f}  detections={preds.sum()}  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

    else:
        print("[main] no example file found to run synthetic injection / top-window saving")

    # save model & metadata
    try:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "grupredictor_sdf.pth"))
        print("[main] saved model")
    except Exception as e:
        print("[main] failed saving model:", e)

    # Safely convert numpy scalars to Python floats for JSON
    def safe_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    meta = {
        "past_samples": int(PAST_SAMPLES),
        "future_samples": int(FUTURE_SAMPLES),
        "sample_rate": int(SAMPLE_RATE),
        "ma_mean": safe_float(np.mean(train_maes)) if train_maes.size>0 else None,
        "ma_std": safe_float(np.std(train_maes)) if train_maes.size>0 else None,
        "sd_mean": safe_float(np.mean(train_sdfs)) if train_sdfs.size>0 else None,
        "sd_std": safe_float(np.std(train_sdfs)) if train_sdfs.size>0 else None,
        "thresholds": {
            "mean+3std": safe_float(thr_mean3) if 'thr_mean3' in locals() else None,
            "p95": safe_float(p95) if 'p95' in locals() else None,
            "p99": safe_float(p99) if 'p99' in locals() else None,
            "p99.5": safe_float(p995) if 'p995' in locals() else None,
            "forced": safe_float(THRESH_FORCE) if THRESH_FORCE is not None else None
        },
        "device": str(DEVICE),
        "hidden": int(HIDDEN_SIZE),
        "epochs": int(EPOCHS)
    }
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[main] saved metadata ->", meta_path)

if __name__ == "__main__":
    # collate helper used by DataLoader lambdas
    def collate_batch(batch):
        pasts = torch.stack([item[0] for item in batch], dim=0)
        futures = torch.stack([item[1] for item in batch], dim=0)
        return pasts, futures
    main()
