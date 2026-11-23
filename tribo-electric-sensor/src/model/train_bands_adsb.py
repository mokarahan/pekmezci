#!/usr/bin/env python3
"""
Bands (20 x 1 kHz, 0-20 kHz) -> flights (lat, lon, altitude): residual CNN, train and predict.

DATA
    /DATA/BANDS/bands_<YYYYmmddTHHMMSS>.<ms>.npy   float matrix, 20 bands x N samples
                                                   (N x 20 is accepted too, it is transposed)
    /DATA/ADSB/adsb_<YYYYmmddTHHMMSS>.<ms>.csv     one file per second; both layouts of
                                                   adsb_lol_recorder.py work (columns lat, lon
                                                   and alt / alt_baro / alt_geom)
    The two files of one second are paired by the equal YYYYmmddTHHMMSS part.

TARGET (20 x 3)
    Up to 20 flights per second, (lat, lon, alt) each, empty slots = 0. The first 20 rows of
    the csv are used, the rest is ignored. Slots are ordered by --order (default: distance
    to the data centre, nearest first). Internally the targets are standardised
    (x_km, y_km, alt_ft) and the network also predicts a "slot is used" probability.
    predict writes the 20 x 3 array back in (lat, lon, alt_ft) with unused slots = 0.

NETWORK  (7 convolution layers, residual)
    layer 1      one kernel set PER BAND: band b (b..b+1 kHz) gets its own 1D kernel(s) that see
                 only that band. The kernel length follows the wavelength, base_kernel / (b+1):
                 100 samples for band 0 (1 kHz) ... 5 samples for band 19 (20 kHz).
                 --band-filters kernels per band (1 = exactly 20 kernels).
    layers 2-7   three residual blocks of two convolutions each (width, 2*width, 4*width
                 channels, each block downsamples by 4); 1x1 pooled shortcuts.
    then         global mean+max pooling over time -> small MLP -> 20 slots x (used, x, y, alt).

TRAINING
    Chronological train/val/test (70/15/15 %) with a gap of --gap files between the parts,
    because neighbouring seconds are almost identical (a random split would leak).
    Loss: Hungarian-matched set loss (slot order does not matter) or --loss slot.
    Metrics need no slot order either: count MAE / accuracy, precision / recall / F1 (a match
    counts if it is within --tol-km in 3D), horizontal error (km), altitude error (ft).
    A "no-audio prior" (average flights per slot) is evaluated for reference: if the network
    does not clearly beat it, it has not found a signal in the bands.

SMALL GPU
    --gpu-mem-mb 1700 keeps training inside that budget (see --ctx-overhead-mb): it caps
    PyTorch's allocator, finds the micro-batch that fits by probing real steps, uses gradient
    accumulation to keep --batch-size, switches gradient checkpointing on if needed, and uses
    mixed precision (fp16 with loss scaling before Ampere, bf16 on Ampere or newer).

Install:  pip install torch numpy scipy
Run:      python train_bands_adsb.py info
          python train_bands_adsb.py train --epochs 30 --gpu-mem-mb 1700
          python train_bands_adsb.py predict --ckpt runs/best.pt \
                 --input /DATA/BANDS/bands_20261122T121005.545.npy
"""

import argparse
import contextlib
import copy
import csv
import math
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

try:    # less memory fragmentation on small GPUs (must be set before CUDA starts)
    if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 1):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
except ValueError:
    pass

N_BANDS = 20
MAX_FLIGHTS = 20
TS_FORMAT = "%Y%m%dT%H%M%S"
TS_RE = re.compile(r"(\d{8}T\d{6})")
KM_PER_DEG = 111.195
KM_PER_FT = 0.0003048


# ============================================================================ data files

def ts_of(name: str):
    m = TS_RE.search(name)
    return m.group(1) if m else None


def index_pairs(bands_dir: str, adsb_dir: str, ts_offset: int = 0):
    """[(timestamp, bands_path, csv_path)] for every second present in both folders."""
    bands, adsb = {}, {}
    for p in sorted(Path(bands_dir).glob("bands_*.npy")):
        ts = ts_of(p.name)
        if ts and ts not in bands:
            bands[ts] = p
    for p in sorted(Path(adsb_dir).glob("adsb_*.csv")):
        ts = ts_of(p.name)
        if ts and ts not in adsb:
            adsb[ts] = p
    pairs = []
    for ts, bp in sorted(bands.items()):
        key = ts
        if ts_offset:
            key = (datetime.strptime(ts, TS_FORMAT) + timedelta(seconds=ts_offset)).strftime(TS_FORMAT)
        if key in adsb:
            pairs.append((ts, bp, adsb[key]))
    return pairs, len(bands), len(adsb)


def _float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _altitude(row: dict) -> float:
    for key in ("alt", "alt_baro", "alt_geom"):
        v = row.get(key)
        if v is None or str(v).strip() == "":
            continue
        if str(v).strip().lower() == "ground":
            return 0.0
        f = _float(v)
        if f is not None:
            return f
    return 0.0


def read_points(path: Path) -> np.ndarray:
    """First MAX_FLIGHTS rows with a position -> (n, 3) array of lat, lon, alt (ft)."""
    pts = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lat, lon = _float(row.get("lat")), _float(row.get("lon"))
            if lat is None or lon is None:
                continue
            pts.append((lat, lon, _altitude(row)))
            if len(pts) >= MAX_FLIGHTS:
                break
    return np.array(pts, dtype=np.float64).reshape(-1, 3)


def orient(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2 and a.shape[0] == N_BANDS:
        return a
    if a.ndim == 2 and a.shape[1] == N_BANDS:
        return a.T
    raise ValueError(f"expected a 20-band matrix (20 x N or N x 20), got shape {a.shape}")


def compute_band_std(paths, n_files: int = 128, seed: int = 0) -> np.ndarray:
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    acc, cnt = np.zeros(N_BANDS), 0
    for p in paths[:n_files]:
        a = orient(np.load(p, mmap_mode="r"))[:, :200000].astype(np.float64)
        acc += (a ** 2).sum(axis=1)
        cnt += a.shape[1]
    return np.sqrt(acc / max(cnt, 1)).clip(1e-8).astype(np.float32)


# ============================================================================ targets

class Normalizer:
    """(lat, lon, alt_ft) <-> standardised (x_km, y_km, alt_ft); builds the 20 x 3 target."""

    def __init__(self, ref_lat, ref_lon, mean, std, order):
        self.ref_lat, self.ref_lon = float(ref_lat), float(ref_lon)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.order = order

    @classmethod
    def fit(cls, point_lists, order, ref=None):
        arrays = [p for p in point_lists if len(p)]
        if not arrays:
            raise SystemExit("no flight points in the training csv files")
        allp = np.concatenate(arrays)
        ref_lat, ref_lon = ref if ref else (allp[:, 0].mean(), allp[:, 1].mean())
        local = cls(ref_lat, ref_lon, np.zeros(3), np.ones(3), order).to_local(allp)
        return cls(ref_lat, ref_lon, local.mean(0), np.maximum(local.std(0), 1.0), order)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        x = (pts[:, 1] - self.ref_lon) * KM_PER_DEG * math.cos(math.radians(self.ref_lat))
        y = (pts[:, 0] - self.ref_lat) * KM_PER_DEG
        return np.stack([x, y, pts[:, 2]], axis=1)

    def to_geo(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=np.float64).reshape(-1, 3)
        lon = self.ref_lon + local[:, 0] / (KM_PER_DEG * math.cos(math.radians(self.ref_lat)))
        lat = self.ref_lat + local[:, 1] / KM_PER_DEG
        return np.stack([lat, lon, local[:, 2]], axis=1)

    def encode(self, pts: np.ndarray):
        tgt = np.zeros((MAX_FLIGHTS, 3), dtype=np.float32)
        mask = np.zeros(MAX_FLIGHTS, dtype=bool)
        n = min(len(pts), MAX_FLIGHTS)
        if n:
            local = self.to_local(pts[:n])
            key = {"lat": local[:, 1], "lon": local[:, 0], "alt": local[:, 2],
                   "dist": np.hypot(local[:, 0], local[:, 1])}.get(self.order)
            idx = np.argsort(key, kind="stable") if key is not None else np.arange(n)
            tgt[:n] = (local[idx] - self.mean) / self.std
            mask[:n] = True
        return tgt, mask

    def denorm_local(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64).reshape(-1, 3) * self.std + self.mean

    def state(self) -> dict:
        return dict(ref_lat=self.ref_lat, ref_lon=self.ref_lon, mean=self.mean.tolist(),
                    std=self.std.tolist(), order=self.order)

    @classmethod
    def from_state(cls, s: dict):
        return cls(s["ref_lat"], s["ref_lon"], s["mean"], s["std"], s["order"])


class BandsDataset(Dataset):
    def __init__(self, items, norm, band_std, n_samples, train, aug):
        self.items = items                              # [(ts, bands_path, points)]
        self.enc = [norm.encode(p) for _, _, p in items]
        self.band_std = torch.from_numpy(band_std)
        self.n_samples, self.train, self.aug = n_samples, train, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = load_bands(self.items[i][1], self.n_samples, self.train)
        x = torch.from_numpy(x) / self.band_std[:, None]
        if self.train and self.aug:                     # the input is noisy: train on noisier
            x = x * random.uniform(0.7, 1.4)
            if random.random() < 0.5:
                x = -x
            x = x + torch.randn_like(x) * random.uniform(0.0, 0.3)
        tgt, mask = self.enc[i]
        return x, torch.from_numpy(tgt), torch.from_numpy(mask), i

    def targets(self):
        tgt = torch.from_numpy(np.stack([t for t, _ in self.enc]))
        mask = torch.from_numpy(np.stack([m for _, m in self.enc]))
        return tgt, mask


def load_bands(path, n_samples: int, random_crop: bool) -> np.ndarray:
    a = orient(np.load(path, mmap_mode="r"))
    length = a.shape[1]
    if length >= n_samples:
        s = random.randint(0, length - n_samples) if random_crop else 0
        return np.array(a[:, s:s + n_samples], dtype=np.float32)
    x = np.zeros((N_BANDS, n_samples), dtype=np.float32)
    x[:, :length] = a
    return x


def prepare_data(args):
    pairs, n_b, n_a = index_pairs(args.bands_dir, args.adsb_dir, args.ts_offset)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} paired seconds ({n_b} band files, {n_a} csv files)")
    if len(pairs) < 10:
        raise SystemExit("need at least 10 paired files (check --bands-dir, --adsb-dir, --ts-offset)")
    items = [(ts, bp, read_points(ap)) for ts, bp, ap in pairs]
    n = len(items)
    a = int(n * 0.70)
    b = a + int(n * 0.15)
    train, val, test = items[:a], items[a + args.gap:b], items[b + args.gap:]
    if not val or not test:
        raise SystemExit("not enough files for this --gap; lower --gap or add data")
    ref = (args.ref_lat, args.ref_lon) if args.ref_lat is not None and args.ref_lon is not None else None
    norm = Normalizer.fit([p for _, _, p in train], args.order, ref)
    band_std = compute_band_std([bp for _, bp, _ in train])
    aug = not args.no_aug
    print(f"train {len(train)} / val {len(val)} / test {len(test)}  "
          f"(chronological, gap {args.gap}); slot order: {args.order}")
    return SimpleNamespace(
        norm=norm, band_std=band_std,
        train=BandsDataset(train, norm, band_std, args.n_samples, True, aug),
        val=BandsDataset(val, norm, band_std, args.n_samples, False, False),
        test=BandsDataset(test, norm, band_std, args.n_samples, False, False))


# ============================================================================ network

def gn(c: int) -> nn.GroupNorm:
    """GroupNorm with the largest of 8, 4, 2, 1 groups that divides the channel count."""
    return nn.GroupNorm(math.gcd(c, 8), c)


class BandwiseStem(nn.Module):
    """LAYER 1: every band has its own 1D kernel(s) that only see that band.

    Band b (b..b+1 kHz) uses a kernel of base_kernel / (b+1) samples, so the kernel always
    spans about the same number of periods: 100 samples at 1 kHz ... 5 samples at 20 kHz.
    Output: (B, 20 * filters, ceil(L / stride)); channels are grouped band by band.
    """

    def __init__(self, filters=1, base_kernel=100, stride=4):
        super().__init__()
        self.stride = stride
        self.kernel_sizes = [max(stride, int(base_kernel / (b + 1) + 0.5)) for b in range(N_BANDS)]
        self.convs = nn.ModuleList(
            nn.Conv1d(1, filters, k, stride=stride, bias=False) for k in self.kernel_sizes)

    def forward(self, x):                                        # x: (B, 20, L)
        length = x.shape[-1]
        out_len = -(-length // self.stride)
        outs = []
        for b, conv in enumerate(self.convs):
            pad = max((out_len - 1) * self.stride + conv.kernel_size[0] - length, 0)
            outs.append(conv(F.pad(x[:, b:b + 1], (pad // 2, pad - pad // 2))))
        return torch.cat(outs, dim=1)


class ResBlock1D(nn.Module):
    """Two convolutions (pre-activation) plus a shortcut: identity, or pooling + 1x1 conv."""

    def __init__(self, cin, cout, stride=1, k=7):
        super().__init__()
        self.n1, self.n2 = gn(cin), gn(cout)
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        if cin == cout and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.AvgPool1d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv1d(cin, cout, 1, bias=False))

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return h + self.skip(x)


# ---- set losses

def slot_loss(logit, xyz, tgt, mask, w_xyz):
    """Slot k of the output is trained against slot k of the target."""
    bce = F.binary_cross_entropy_with_logits(logit, mask.float())
    reg = F.smooth_l1_loss(xyz[mask], tgt[mask]) if mask.any() else xyz.sum() * 0.0
    return bce + w_xyz * reg, {"bce": bce.item(), "reg": reg.item()}


def matched_loss(logit, xyz, tgt, mask, w_xyz):
    """Hungarian matching: the slot order of the target does not matter."""
    with torch.no_grad():
        cost = torch.cdist(xyz.float(), tgt.float(), p=1) - torch.sigmoid(logit.float())[:, :, None]
        cost = cost.cpu().numpy()
    present = torch.zeros_like(logit)
    bi, pi, gi = [], [], []
    for b, n in enumerate(mask.sum(1).tolist()):
        if n == 0:
            continue
        r, c = linear_sum_assignment(cost[b][:, :n])    # present targets come first
        present[b, torch.as_tensor(r, device=logit.device)] = 1.0
        bi += [b] * len(r)
        pi += r.tolist()
        gi += c.tolist()
    bce = F.binary_cross_entropy_with_logits(logit, present)
    if bi:
        dev = logit.device
        bi, pi, gi = (torch.tensor(v, device=dev) for v in (bi, pi, gi))
        reg = F.smooth_l1_loss(xyz[bi, pi], tgt[bi, gi])
    else:
        reg = xyz.sum() * 0.0
    return bce + w_xyz * reg, {"bce": bce.item(), "reg": reg.item()}


class BandResNet(nn.Module):
    """7 convolution layers: band-wise layer 1 + three residual blocks of two convolutions."""

    def __init__(self, width=64, enc_dim=256, band_filters=1, base_kernel=100, stem_stride=4,
                 loss_mode="matched", w_xyz=5.0, dropout=0.2):
        super().__init__()
        c0 = N_BANDS * band_filters
        self.stem = BandwiseStem(band_filters, base_kernel, stem_stride)          # layer 1
        self.stem_norm = gn(c0)
        blocks, cin = [], c0
        for w in (width, 2 * width, 4 * width):                                   # layers 2-7
            blocks.append(ResBlock1D(cin, w, stride=4))
            cin = w
        self.blocks = nn.ModuleList(blocks)
        self.norm = gn(cin)
        self.proj = nn.Linear(2 * cin, enc_dim)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(enc_dim, 512),
                                  nn.GELU(), nn.Linear(512, MAX_FLIGHTS * 4))
        self.loss_mode, self.w_xyz = loss_mode, w_xyz
        self.ckpt = False                   # gradient checkpointing: less GPU memory, ~30 % slower

    def features(self, x):
        use = self.ckpt and self.training and torch.is_grad_enabled()

        def front(t):
            return F.gelu(self.stem_norm(self.stem(t)))

        h = checkpoint(front, x, use_reentrant=False) if use else front(x)
        for blk in self.blocks:
            h = checkpoint(blk, h, use_reentrant=False) if use else blk(h)
        h = F.gelu(self.norm(h))
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    def forward(self, x):
        out = self.head(self.features(x)).float().view(-1, MAX_FLIGHTS, 4)
        return out[..., 0], out[..., 1:]

    def loss(self, x, tgt, mask):
        logit, xyz = self(x)
        fn = matched_loss if self.loss_mode == "matched" else slot_loss
        return fn(logit, xyz, tgt, mask, self.w_xyz)

    @torch.no_grad()
    def predict(self, x):
        logit, xyz = self(x)
        return torch.sigmoid(logit), xyz


class PriorBaseline(nn.Module):
    """No audio: per-slot presence frequency and mean position of the training set (reference)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("p", torch.zeros(MAX_FLIGHTS))
        self.register_buffer("xyz", torch.zeros(MAX_FLIGHTS, 3))
        self.dummy = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def fit(self, tgt, mask):
        m = mask.float()
        self.p.copy_(m.mean(0))
        self.xyz.copy_((tgt * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None])

    def loss(self, x, tgt, mask):
        return self.dummy.sum() * 0.0, {}

    @torch.no_grad()
    def predict(self, x):
        b = x.shape[0]
        return self.p.expand(b, -1), self.xyz.expand(b, -1, -1)


def model_hp(args) -> dict:
    return dict(width=args.width, enc_dim=args.enc_dim, band_filters=args.band_filters,
                base_kernel=args.base_kernel, stem_stride=args.stem_stride,
                loss_mode=args.loss, w_xyz=args.w_xyz)


# ============================================================================ training / evaluation

class EMA:
    def __init__(self, model, decay):
        self.decay, self.step = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)


AMP_ENABLED = True


def amp_dtype(device):
    """bf16 on Ampere or newer, fp16 (with loss scaling) on older GPUs, none on CPU."""
    if device.type != "cuda" or not AMP_ENABLED:
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def amp_ctx(device):
    dt = amp_dtype(device)
    return torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()


def make_scaler(enabled: bool, device):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError, RuntimeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def match_flights(pred_local: np.ndarray, gt_local: np.ndarray, tol_km: float):
    """Hungarian match in 3D (km). Returns (true positives, horizontal errs km, altitude errs ft)."""
    if len(pred_local) == 0 or len(gt_local) == 0:
        return 0, [], []
    p, g = pred_local.copy(), gt_local.copy()
    p[:, 2] *= KM_PER_FT
    g[:, 2] *= KM_PER_FT
    cost = np.linalg.norm(p[:, None] - g[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    horiz = np.linalg.norm(pred_local[r, :2] - gt_local[c, :2], axis=1)
    alt = np.abs(pred_local[r, 2] - gt_local[c, 2])
    return int((cost[r, c] < tol_km).sum()), horiz.tolist(), alt.tolist()


@torch.no_grad()
def evaluate(model, loader, norm, device, tol_km=10.0, thr=0.5):
    model.eval()
    tot_loss = n = 0
    tp = n_pred = n_gt = count_ok = 0
    count_err, horiz, alt = [], [], []
    for x, tgt, mask, _ in loader:
        x_d = x.to(device, non_blocking=True)
        with amp_ctx(device):
            loss, _ = model.loss(x_d, tgt.to(device), mask.to(device))
            prob, xyz = model.predict(x_d)
        tot_loss += float(loss) * len(x)
        n += len(x)
        prob, xyz = prob.float().cpu().numpy(), xyz.float().cpu().numpy()
        tgt, mask = tgt.numpy(), mask.numpy()
        for b in range(len(prob)):
            sel = prob[b] > thr
            pred = norm.denorm_local(xyz[b][sel])
            gt = norm.denorm_local(tgt[b][mask[b]])
            t, h, a = match_flights(pred, gt, tol_km)
            tp += t
            n_pred += len(pred)
            n_gt += len(gt)
            count_err.append(abs(len(pred) - len(gt)))
            count_ok += int(len(pred) == len(gt))
            horiz += h
            alt += a
    prec, rec = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {
        "loss": tot_loss / max(n, 1), "n": n,
        "count_mae": float(np.mean(count_err)), "count_acc": count_ok / max(n, 1),
        "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "horiz_km_mean": float(np.mean(horiz)) if horiz else float("nan"),
        "horiz_km_median": float(np.median(horiz)) if horiz else float("nan"),
        "alt_ft_mae": float(np.mean(alt)) if alt else float("nan"),
    }


def fmt_metrics(m: dict) -> str:
    return (f"loss {m['loss']:.4f} | count MAE {m['count_mae']:.2f} acc {m['count_acc']:.2f} | "
            f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.3f} | "
            f"horiz median {m['horiz_km_median']:.1f} km, alt MAE {m['alt_ft_mae']:.0f} ft")


def make_loader(ds, batch, train, workers):
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train and len(ds) >= batch,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def probe_peak_mb(model, ds, micro, device) -> float:
    """Peak GPU MB of one forward + backward on `micro` real samples (inf = out of memory)."""
    x, tgt, mask, _ = default_collate([ds[i % len(ds)] for i in range(micro)])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss, peak = None, float("inf")
    try:
        with amp_ctx(device):
            loss, _ = model.loss(x.to(device), tgt.to(device), mask.to(device))
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
    loss = None
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return peak


def largest_micro_batch(peak_fn, start: int, limit_mb: float) -> int:
    """Largest micro-batch (start, start/2, ... 1) whose peak memory fits, 0 if none."""
    m = start
    while m >= 1:
        if peak_fn(m) <= limit_mb:
            return m
        m //= 2
    return 0


def plan_memory(args, model, data, device):
    """Fit one training step into the GPU budget -> (micro_batch, accumulation, grad_ckpt)."""
    batch = args.batch_size
    micro = min(args.micro_batch or batch, batch)
    model.ckpt = args.grad_ckpt
    if device.type != "cuda" or args.gpu_mem_mb <= 0:
        return micro, math.ceil(batch / micro), args.grad_ckpt
    budget = args.gpu_mem_mb - args.ctx_overhead_mb
    param_mb = sum(p.numel() * 4 for p in model.parameters()) / 2 ** 20
    limit = 0.9 * budget - 3 * param_mb              # 10 % headroom; AdamW states (2x) + EMA copy (1x)

    def peak(ckpt):
        def fn(m):
            model.ckpt = ckpt
            return probe_peak_mb(model, data.train, m, device)
        return fn

    use_ckpt = args.grad_ckpt
    m = largest_micro_batch(peak(use_ckpt), micro, limit)
    if not use_ckpt and m < min(8, micro):
        m2 = largest_micro_batch(peak(True), micro, limit)
        if m2 > m:
            m, use_ckpt = m2, True
    if m == 0:
        raise SystemExit(f"even one sample does not fit {args.gpu_mem_mb:.0f} MB; "
                         "lower --n-samples, --width or --band-filters, or raise --gpu-mem-mb")
    model.ckpt = use_ckpt
    return m, math.ceil(batch / m), use_ckpt


def train_model(args, data, device) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    on_cuda = device.type == "cuda"
    if on_cuda and args.gpu_mem_mb > 0:
        budget = args.gpu_mem_mb - args.ctx_overhead_mb
        if budget < 300:
            raise SystemExit("--gpu-mem-mb minus --ctx-overhead-mb is below 300 MB")
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 20
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), device)
        print(f"GPU budget {args.gpu_mem_mb:.0f} MB = {budget:.0f} MB for PyTorch "
              f"+ {args.ctx_overhead_mb:.0f} MB CUDA context")
    hp = model_hp(args)
    model = BandResNet(**hp).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBandResNet: {n_params / 1e6:.2f} M parameters, 7 convolution layers")
    print(f"layer-1 kernel length per band (0-1 kHz ... 19-20 kHz): {model.stem.kernel_sizes}")
    train_tgt, train_mask = data.train.targets()

    micro, accum, ckpt = plan_memory(args, model, data, device)
    print(f"micro-batch {micro} x accumulation {accum} (effective {micro * accum}), "
          f"gradient checkpointing {'on' if ckpt else 'off'}, amp {amp_dtype(device) or 'off'}")
    eval_bs = max(1, min(micro, 16))
    val_loader = make_loader(data.val, eval_bs, False, args.workers)
    test_loader = make_loader(data.test, eval_bs, False, args.workers)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    train_loader = make_loader(data.train, micro, True, args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = max(1, args.epochs * max(1, len(train_loader) // accum))
    warm = max(1, min(200, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = make_scaler(amp_dtype(device) == torch.float16, device)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    best_key, best_state, best_epoch = None, None, -1
    final = model
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, run, cnt = time.time(), 0.0, 0
        for i, (x, tgt, mask, _) in enumerate(train_loader):
            x, tgt, mask = x.to(device, non_blocking=True), tgt.to(device), mask.to(device)
            with amp_ctx(device):
                loss, _ = model.loss(x, tgt, mask)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema:
                    ema.update(model)
            run += float(loss.detach()) * len(x)
            cnt += len(x)
        final = ema.model if ema else model
        val = evaluate(final, val_loader, data.norm, device, args.tol_km)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train loss {run / max(cnt, 1):.4f}  "
              f"val: {fmt_metrics(val)}  ({time.time() - t0:.0f}s)")
        key = (val["f1"], -val["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = copy.deepcopy(final.state_dict())
    final = ema.model if ema else model
    if best_state is not None:
        final.load_state_dict(best_state)
    val = evaluate(final, val_loader, data.norm, device, args.tol_km)
    test = evaluate(final, test_loader, data.norm, device, args.tol_km)

    prior = PriorBaseline().to(device)                  # reference without any audio
    prior.fit(train_tgt.to(device), train_mask.to(device))
    prior_test = evaluate(prior, test_loader, data.norm, device, args.tol_km)
    print(f"\nbest epoch {best_epoch}  ({(time.time() - t_start) / 60:.1f} min)\n"
          f"  val : {fmt_metrics(val)}\n  test: {fmt_metrics(test)}\n"
          f"  test, no-audio prior (reference): {fmt_metrics(prior_test)}")
    if on_cuda:
        print(f"  GPU peak: {torch.cuda.max_memory_allocated(device) / 2 ** 20:.0f} MB allocated, "
              f"{torch.cuda.max_memory_reserved(device) / 2 ** 20:.0f} MB reserved "
              f"(+ about {args.ctx_overhead_mb:.0f} MB CUDA context outside PyTorch)")
    torch.save({"hp": hp, "state": final.state_dict(), "norm": data.norm.state(),
                "band_std": data.band_std, "n_samples": args.n_samples,
                "val": val, "test": test, "test_prior": prior_test}, out_dir / "best.pt")
    print(f"saved {out_dir / 'best.pt'}")
    return {"val": val, "test": test}


# ============================================================================ commands

def pick_device(arg: str) -> torch.device:
    return torch.device(arg if arg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cmd_info(args):
    pairs, n_b, n_a = index_pairs(args.bands_dir, args.adsb_dir, args.ts_offset)
    print(f"band files {n_b}, csv files {n_a}, paired seconds {len(pairs)}")
    if not pairs:
        return
    a = np.load(pairs[0][1], mmap_mode="r")
    print(f"first band file {pairs[0][1].name}: shape {a.shape}, dtype {a.dtype} "
          f"-> used as {orient(a).shape} (bands x samples)")
    counts = np.array([len(read_points(ap)) for _, _, ap in pairs])
    print(f"flights per second: mean {counts.mean():.2f}, max {counts.max()}, "
          f"seconds with 0 flights {np.mean(counts == 0) * 100:.1f} %, "
          f"capped at {MAX_FLIGHTS}: {np.mean(counts >= MAX_FLIGHTS) * 100:.1f} %")
    print("histogram (flights: seconds): " +
          ", ".join(f"{k}: {int((counts == k).sum())}" for k in range(0, min(counts.max(), MAX_FLIGHTS) + 1)))


def cmd_train(args):
    set_seed(args.seed)
    train_model(args, prepare_data(args), pick_device(args.device))


def cmd_predict(args):
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = BandResNet(**ck["hp"])
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    norm = Normalizer.from_state(ck["norm"])
    band_std = torch.as_tensor(ck["band_std"])
    src = Path(args.input)
    paths = sorted(src.glob("bands_*.npy")) if src.is_dir() else [src]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        x = torch.from_numpy(load_bands(p, ck["n_samples"], False)) / band_std[:, None]
        with amp_ctx(device):
            prob, xyz = model.predict(x[None].to(device))
        prob, xyz = prob[0].float().cpu().numpy(), xyz[0].float().cpu().numpy()
        sel = prob > args.threshold
        result = np.zeros((MAX_FLIGHTS, 3), dtype=np.float64)           # unused slots stay 0
        result[sel] = norm.to_geo(norm.denorm_local(xyz[sel]))
        np.save(out_dir / f"{p.stem}.pred.npy", result)
        print(f"{p.name}: {int(sel.sum())} flight(s)")
        for k in np.flatnonzero(sel):
            print(f"   slot {k:2d}  p={prob[k]:.2f}  lat {result[k, 0]:.5f}  lon {result[k, 1]:.5f}  "
                  f"alt {result[k, 2]:.0f} ft")


def add_data_args(sp):
    sp.add_argument("--bands-dir", default="/DATA/BANDS")
    sp.add_argument("--adsb-dir", default="/DATA/ADSB")
    sp.add_argument("--ts-offset", type=int, default=0,
                    help="seconds to add to the bands timestamp to get the csv timestamp (clock offset)")
    sp.add_argument("--limit", type=int, default=0, help="use only the first N paired seconds")


def add_train_args(sp):
    add_data_args(sp)
    sp.add_argument("--out", default="runs")
    sp.add_argument("--n-samples", type=int, default=48000, help="samples per band fed to the network")
    sp.add_argument("--order", choices=["dist", "lat", "lon", "alt", "file"], default="dist",
                    help="slot order of the 20 flights in the target")
    sp.add_argument("--ref-lat", type=float, default=None, help="reference point for --order dist")
    sp.add_argument("--ref-lon", type=float, default=None)
    sp.add_argument("--gap", type=int, default=5, help="files left out between train/val/test")
    sp.add_argument("--tol-km", type=float, default=10.0, help="a flight counts as found within this 3D distance")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    sp.add_argument("--micro-batch", type=int, default=0,
                    help="samples per GPU step (0 = all; the rest is gradient accumulation)")
    sp.add_argument("--gpu-mem-mb", type=float, default=0,
                    help="GPU memory budget incl. CUDA context, e.g. 1700; picks micro-batch / checkpointing "
                         "automatically and caps PyTorch's allocator (0 = no limit)")
    sp.add_argument("--ctx-overhead-mb", type=float, default=450,
                    help="CUDA context + kernels outside PyTorch's allocator, counted in the budget")
    sp.add_argument("--grad-ckpt", action="store_true", help="force gradient checkpointing")
    sp.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--wd", type=float, default=0.02)
    sp.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights (0 = off)")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--no-aug", action="store_true", help="disable gain / sign / noise augmentation")
    sp.add_argument("--loss", choices=["matched", "slot"], default="matched",
                    help="Hungarian-matched set loss or slot-by-slot loss")
    sp.add_argument("--w-xyz", type=float, default=5.0, help="weight of the position loss")
    sp.add_argument("--band-filters", type=int, default=1,
                    help="kernels per band in layer 1 (1 = exactly 20 kernels, one per band)")
    sp.add_argument("--base-kernel", type=int, default=100,
                    help="layer-1 kernel length of band 0 (1 kHz); band b uses base / (b+1)")
    sp.add_argument("--stem-stride", type=int, default=4, help="layer-1 stride (<= shortest kernel)")
    sp.add_argument("--width", type=int, default=64, help="channels of the first residual block (then x2, x4)")
    sp.add_argument("--enc-dim", type=int, default=256)


def main():
    parser = argparse.ArgumentParser(description="bands -> flights, residual CNN in PyTorch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="check pairing and flights per second")
    add_data_args(s)
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("train", help="train the network")
    add_train_args(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("predict", help="band file(s) -> 20 x 3 flight arrays")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--input", required=True, help="a bands .npy file or a folder of them")
    s.add_argument("--out", default="predictions")
    s.add_argument("--threshold", type=float, default=0.5, help="slot is used if probability > threshold")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_predict)

    args = parser.parse_args()
    global AMP_ENABLED
    AMP_ENABLED = not getattr(args, "no_amp", False)
    args.fn(args)


if __name__ == "__main__":
    main()
