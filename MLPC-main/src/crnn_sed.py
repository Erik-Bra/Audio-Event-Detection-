"""
CRNN for polyphonic Sound Event Detection (MLPC 2026 Task 5, Bonus 1).

Pipeline:
  wav -> log-mel  (computed ONCE and cached) -> CNN (freq pooling) -> BiGRU
       -> per-frame sigmoid(15) -> max-pool to 1-second -> threshold -> onset/offset events.

SPEED: the log-mel spectrogram and the majority-voted targets are precomputed once per
file and cached to disk (and held in RAM during a run). Each epoch then only slices the
cached features and runs the GPU forward/backward -- no repeated wav decode / resample /
mel / annotation parsing. Mixed precision + cuDNN autotuner are on by default.

Point CONFIG.data_root at the raw dataset root containing train/ validation/ test/,
each with annotations.csv, metadata.csv, audio/*.wav (test has only audio/).
"""

import os, math, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
torch.backends.cudnn.benchmark = True            # autotune conv kernels for fixed-ish shapes

PROJECT_DIR = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# CONFIG  (the knobs you tune for Bonus 1b live here)
# --------------------------------------------------------------------------- #
class CFG:
    data_root   = str(PROJECT_DIR)
    out_dir     = str(PROJECT_DIR / "outputs" / "crnn")
    cache_root  = str(PROJECT_DIR / "outputs" / "crnn_cache")     # precomputed log-mel + targets land here

    sr          = 16_000
    n_fft       = 1024
    hop         = 320                 # 320/16000 = 20 ms -> 50 frames/sec
    n_mels      = 64
    fmin        = 20
    fmax        = 8000

    crop_sec    = 10                  # fixed-length training crops
    batch_size  = 16
    epochs      = 70
    lr          = 1e-3
    weight_decay= 1e-4
    gru_hidden  = 128
    gru_layers  = 2
    dropout     = 0.3
    pos_weight_cap = 50.0
    threshold   = 0.5

    amp         = True                # mixed precision on CUDA
    cache_in_ram= True                # load the whole feature cache into RAM (fast slicing)
    eval_every  = 1                   # run validation every N epochs (raise to skip evals)
    seed        = 0
    num_workers = 0                   # cached RAM slicing -> 0 is fastest & Windows-safe
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def fps(self):                    # frames per second
        return self.sr / self.hop


# --------------------------------------------------------------------------- #
# Class list (derived from data -> nothing hardcoded)
# --------------------------------------------------------------------------- #
def get_class_names(train_ann_csv):
    ann = pd.read_csv(train_ann_csv)
    return sorted(ann["annotation"].unique().tolist())


# --------------------------------------------------------------------------- #
# Majority-voted, frame-aligned targets  (built once, during caching)
# --------------------------------------------------------------------------- #
def build_targets(grp, n_frames, class_to_idx, cfg):
    """[n_frames, C] 0/1: keep a (frame, class) where >= half the file's annotators agree."""
    C = len(class_to_idx)
    y = np.zeros((n_frames, C), dtype=np.uint8)
    if grp is None:
        return y
    n_annot = grp["annotator_id"].nunique()
    if n_annot == 0:
        return y
    times = np.arange(n_frames) * cfg.hop / cfg.sr
    votes = np.zeros((n_frames, C), dtype=np.float32)
    for cls, cgrp in grp.groupby("annotation"):
        if cls not in class_to_idx:
            continue
        ci = class_to_idx[cls]
        for _, agrp in cgrp.groupby("annotator_id"):
            covered = np.zeros(n_frames, dtype=bool)
            for _, row in agrp.iterrows():
                covered |= (times >= row["onset"]) & (times < row["offset"])
            votes[:, ci] += covered
    return (votes * 2.0 >= n_annot).astype(np.uint8)


# --------------------------------------------------------------------------- #
# One-time feature cache:  wav -> log-mel + targets -> disk
# --------------------------------------------------------------------------- #
def _mel_signature(cfg):
    return f"mel{cfg.n_mels}_nfft{cfg.n_fft}_hop{cfg.hop}_sr{cfg.sr}_fmin{cfg.fmin}_fmax{cfg.fmax}"

def build_cache(split, audio_dir, ann_csv, class_names, cfg):
    """Compute (or reuse) the log-mel + target cache for one split. Returns the cache dir."""
    import soundfile as sf
    cdir = os.path.join(cfg.cache_root, _mel_signature(cfg), split)
    os.makedirs(cdir, exist_ok=True)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    ann_by_file = {}
    if ann_csv is not None and os.path.exists(ann_csv):
        for fn, grp in pd.read_csv(ann_csv).groupby("filename"):
            ann_by_file[fn] = grp

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
        n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax, center=True, power=2.0,
    ).to(cfg.device)

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(audio_dir, "*.wav")))
    todo = [f for f in files if not os.path.exists(os.path.join(cdir, f[:-4] + ".npz"))]
    if todo:
        print(f"  caching {len(todo)}/{len(files)} {split} files -> {cdir}")
    for k, fn_wav in enumerate(files):
        outp = os.path.join(cdir, fn_wav[:-4] + ".npz")
        if os.path.exists(outp):
            continue
        data, sr = sf.read(os.path.join(audio_dir, fn_wav), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
        if sr != cfg.sr:
            wav = torchaudio.functional.resample(wav, sr, cfg.sr)
        dur = int(math.ceil(wav.shape[0] / cfg.sr))
        with torch.no_grad():
            m = mel_tf(wav.to(cfg.device))                 # [n_mels, T]
            m = torch.log(m + 1e-6).cpu().numpy().astype(np.float16)
        n_frames = m.shape[1]
        fn = fn_wav[:-4]
        grp = ann_by_file.get(fn_wav, ann_by_file.get(fn))
        target = build_targets(grp, n_frames, class_to_idx, cfg)
        np.savez(outp, mel=m, target=target, dur=np.int32(dur))
        if todo and (k % 200 == 0):
            print(f"    {k}/{len(files)}")
    return cdir


# --------------------------------------------------------------------------- #
# Dataset reading the cache (optionally fully in RAM)
# --------------------------------------------------------------------------- #
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, cfg, train=True):
        self.cfg = cfg
        self.train = train
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        self.ram = None
        if cfg.cache_in_ram:
            self.ram = []
            for p in self.paths:
                d = np.load(p)
                self.ram.append((d["mel"], d["target"], int(d["dur"]), os.path.basename(p)[:-4]))

    def __len__(self):
        return len(self.paths)

    def _get(self, i):
        if self.ram is not None:
            return self.ram[i]
        d = np.load(self.paths[i])
        return d["mel"], d["target"], int(d["dur"]), os.path.basename(self.paths[i])[:-4]

    def __getitem__(self, i):
        mel, target, dur, fn = self._get(i)
        mel = torch.from_numpy(mel.astype(np.float32))     # [n_mels, T]
        target = torch.from_numpy(target.astype(np.float32))  # [T, C]
        if self.train:
            cf = int(round(self.cfg.crop_sec * self.cfg.fps))
            T = mel.shape[1]
            if T < cf:                                     # pad short clips
                mel = torch.nn.functional.pad(mel, (0, cf - T))
                target = torch.nn.functional.pad(target, (0, 0, 0, cf - target.shape[0]))
            else:                                          # random time crop
                t0 = int(np.random.randint(0, T - cf + 1))
                mel = mel[:, t0:t0 + cf]
                target = target[t0:t0 + cf]
        return mel, target, dur, fn


def collate(batch):
    mels, ys, durs, fns = zip(*batch)
    max_t = max(m.shape[1] for m in mels)
    n_mels = mels[0].shape[0]; n_cls = ys[0].shape[1]
    Mx = torch.zeros(len(mels), n_mels, max_t)
    Y  = torch.zeros(len(ys), max_t, n_cls)
    Msk= torch.zeros(len(ys), max_t)
    for i, (m, y) in enumerate(zip(mels, ys)):
        Mx[i, :, :m.shape[1]] = m
        Y[i, :y.shape[0]] = y
        Msk[i, :y.shape[0]] = 1.0
    return Mx, Y, Msk, list(durs), list(fns)


# --------------------------------------------------------------------------- #
# Model: (cached) log-mel -> CNN (freq pooling) -> BiGRU -> linear
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, freq_pool):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d((freq_pool, 1)),
        )
    def forward(self, x):
        return self.body(x)

class CRNN(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(ConvBlock(1, 32, 2), ConvBlock(32, 64, 2), ConvBlock(64, 128, 2))
        feat = 128 * (cfg.n_mels // 8)
        self.gru = nn.GRU(feat, cfg.gru_hidden, cfg.gru_layers, batch_first=True,
                          bidirectional=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * cfg.gru_hidden, n_classes))

    def forward(self, mel):                # mel: [B, n_mels, T] (log-mel)
        x = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        x = x.unsqueeze(1)                 # [B,1,M,T]
        x = self.cnn(x)                    # [B,128,M',T]
        x = x.permute(0, 3, 1, 2).flatten(2)   # [B,T,feat]
        x, _ = self.gru(x)
        return self.head(x)                # [B,T,C] logits


# --------------------------------------------------------------------------- #
# Per-second pooling + segment-based macro F1
# --------------------------------------------------------------------------- #
def frames_to_seconds(prob_frames, cfg, n_seconds):
    T, C = prob_frames.shape
    sec = np.floor(np.arange(T) * cfg.hop / cfg.sr).astype(int)
    out = np.zeros((n_seconds, C), dtype=np.float32)
    for s in range(n_seconds):
        m = sec == s
        if m.any():
            out[s] = prob_frames[m].max(0)
    return out

def macro_f1_from_seconds(pred_bin, true_bin):
    P = np.concatenate(pred_bin, 0); Y = np.concatenate(true_bin, 0)
    tp = (P * Y).sum(0); fp = (P * (1 - Y)).sum(0); fn = ((1 - P) * Y).sum(0)
    prec = tp / np.maximum(tp + fp, 1e-9); rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return f1.mean(), f1


@torch.no_grad()
def run_inference(model, ds, class_names, cfg, labeled):
    """Full-file inference over a cached dataset -> (events_df, sec_pred, sec_true)."""
    model.eval()
    rows, sec_pred, sec_true = [], [], []
    for i in range(len(ds)):
        mel, target, dur, fn = ds._get(i)
        mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(cfg.device)
        with torch.autocast(device_type="cuda", enabled=cfg.amp and cfg.device == "cuda"):
            logits = model(mel_t)[0]
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        pb = (frames_to_seconds(prob, cfg, dur) >= cfg.threshold).astype(np.float32)
        sec_pred.append(pb)
        if labeled:
            yt = target.astype(np.float32)
            sec_true.append((frames_to_seconds(yt, cfg, dur) >= 0.5).astype(np.float32))
        for ci, cls in enumerate(class_names):
            active = pb[:, ci] > 0; s = 0
            while s < len(active):
                if active[s]:
                    e = s
                    while e + 1 < len(active) and active[e + 1]:
                        e += 1
                    rows.append({"filename": fn + ".wav", "annotation": cls,
                                 "onset": float(s), "offset": float(e + 1)})
                    s = e + 1
                else:
                    s += 1
    df = pd.DataFrame(rows, columns=["filename", "annotation", "onset", "offset"])
    return df, sec_pred, sec_true


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def compute_pos_weight(ds, n_classes, cfg):
    pos = np.zeros(n_classes); tot = 0
    for i in range(len(ds)):
        _, target, _, _ = ds._get(i)
        pos += target.sum(0); tot += target.shape[0]
    w = (tot - pos) / np.maximum(pos, 1.0)
    return torch.tensor(np.minimum(w, cfg.pos_weight_cap), dtype=torch.float32)

def train(cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    tr_ann = os.path.join(cfg.data_root, "train", "annotations.csv")
    class_names = get_class_names(tr_ann); n_classes = len(class_names)
    print(f"{n_classes} classes: {class_names}")

    print("Preparing feature cache (one-time)...")
    tr_cdir = build_cache("train", os.path.join(cfg.data_root, "train", "audio"), tr_ann, class_names, cfg)
    va_cdir = build_cache("validation", os.path.join(cfg.data_root, "validation", "audio"),
                          os.path.join(cfg.data_root, "validation", "annotations.csv"), class_names, cfg)

    tr_ds = CachedDataset(tr_cdir, cfg, train=True)
    va_ds = CachedDataset(va_cdir, cfg, train=False)
    print(f"train files: {len(tr_ds)} | val files: {len(va_ds)}")

    pw = compute_pos_weight(tr_ds, n_classes, cfg).to(cfg.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    dl = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                                     collate_fn=collate, num_workers=cfg.num_workers,
                                     drop_last=True, pin_memory=(cfg.device == "cuda"))
    model = CRNN(cfg, n_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device == "cuda")

    best = -1.0
    for ep in range(cfg.epochs):
        model.train(); run = 0.0
        for Mx, Y, Msk, _, _ in dl:
            Mx, Y, Msk = Mx.to(cfg.device), Y.to(cfg.device), Msk.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and cfg.device == "cuda"):
                logits = model(Mx)
                T = min(logits.shape[1], Y.shape[1])
                l = loss_fn(logits[:, :T], Y[:, :T])
                l = (l * Msk[:, :T].unsqueeze(-1)).sum() / Msk[:, :T].sum().clamp(min=1)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            run += l.item()
        sched.step()
        if (ep % cfg.eval_every == 0) or (ep == cfg.epochs - 1):
            _, vp, vt = run_inference(model, va_ds, class_names, cfg, labeled=True)
            vf1, _ = macro_f1_from_seconds(vp, vt)
            print(f"epoch {ep:02d} | loss {run/len(dl):.4f} | val macroF1 {vf1:.4f}")
            if vf1 > best:
                best = vf1
                torch.save({"model": model.state_dict(), "classes": class_names},
                           os.path.join(cfg.out_dir, "best.pt"))
        else:
            print(f"epoch {ep:02d} | loss {run/len(dl):.4f}")
    print(f"best val macroF1 {best:.4f}")
    return class_names


if __name__ == "__main__":
    cfg = CFG()
    classes = train(cfg)
    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=cfg.device)
    model = CRNN(cfg, len(classes)).to(cfg.device); model.load_state_dict(ckpt["model"])
    te_cdir = build_cache("test", os.path.join(cfg.data_root, "test", "audio"), None, classes, cfg)
    te_ds = CachedDataset(te_cdir, cfg, train=False)
    sub, _, _ = run_inference(model, te_ds, classes, cfg, labeled=False)
    sub.to_csv(os.path.join(cfg.out_dir, "submission.csv"), index=False)
    print("wrote", os.path.join(cfg.out_dir, "submission.csv"), "rows:", len(sub))