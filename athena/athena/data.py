"""Robust CSI recording, resampling and leakage-safe window generation."""
import json
import os
import random
import re
import time

import numpy as np

FEATURES = 64


# --------------------------------------------------------------- saving

def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s).strip()).strip("_") or "x"


def save_take(root, label, person, room, ts, rssi, amp):
    label = safe_name(label).lower()
    ts = np.asarray(ts, dtype=np.uint32).reshape(-1)
    rssi = np.asarray(rssi, dtype=np.int8).reshape(-1)
    amp = np.asarray(amp, dtype=np.uint8)
    if amp.ndim != 2 or amp.shape[1] != FEATURES:
        raise ValueError(f"CSI must have shape (frames, {FEATURES}); got {amp.shape}")
    if not (len(ts) == len(rssi) == len(amp)):
        raise ValueError("timestamp, RSSI and CSI frame counts must match")
    if len(ts) < 2:
        raise ValueError("recording needs at least two CSI frames")
    d = os.path.join(root, label)
    os.makedirs(d, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_name(person)}_{safe_name(room)}"
    path, k = os.path.join(d, stem + ".npz"), 1
    while os.path.exists(path):
        k += 1
        path = os.path.join(d, f"{stem}-{k}.npz")
    np.savez_compressed(path, ts=ts, rssi=rssi, amp=amp, label=label,
                        person=str(person or ""), room=str(room or ""))
    return path


def list_takes(root):
    """Return {label: [recording paths]} in stable order."""
    out = {}
    if not os.path.isdir(root):
        return out
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if os.path.isdir(d):
            files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz"))
            if files:
                out[lab] = files
    return out


# ----------------------------------------------------------- processing

def _unwrap_u32_timestamps(ts):
    """Unwrap ESP millisecond timestamps across uint32 rollover before sorting."""
    raw = np.asarray(ts).reshape(-1)
    if not len(raw):
        return np.zeros(0, dtype=np.float64)
    vals = raw.astype(np.uint64).astype(np.float64)
    offset = 0.0
    out = np.empty(len(vals), dtype=np.float64)
    previous = vals[0]
    for i, value in enumerate(vals):
        if i and value - previous < -(2 ** 31):
            offset += 2 ** 32
        out[i] = value + offset
        previous = value
    return out


def resample(ts_ms, amp, fs, max_gap_ms=None):
    """Resample irregular CSI to fs Hz; long packet gaps stay invalid, not fabricated."""
    arr = np.asarray(amp, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, FEATURES), np.float32)
    ts = _unwrap_u32_timestamps(ts_ms)
    if len(ts) != len(arr) or len(ts) < 2 or fs <= 0:
        return np.zeros((0, arr.shape[1]), np.float32)
    finite_rows = np.isfinite(ts) & np.isfinite(arr).all(axis=1)
    ts, arr = ts[finite_rows], arr[finite_rows]
    if len(ts) < 2:
        return np.zeros((0, arr.shape[1]), np.float32)
    order = np.argsort(ts, kind="stable")
    ts, arr = ts[order], arr[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, arr = ts[keep], arr[keep]
    if len(ts) < 2:
        return np.zeros((0, arr.shape[1]), np.float32)

    step = 1000.0 / float(fs)
    if max_gap_ms is None:
        max_gap_ms = max(120.0, 4.0 * step)
    grid = np.arange(ts[0], ts[-1] + 1e-6, step, dtype=np.float64)
    out = np.empty((len(grid), arr.shape[1]), np.float32)
    for k in range(arr.shape[1]):
        out[:, k] = np.interp(grid, ts, arr[:, k])
    # Interpolation across a brief jitter is fine. Across a long dropout it invents motion.
    gaps = np.diff(ts) > float(max_gap_ms)
    for left, right in zip(ts[:-1][gaps], ts[1:][gaps]):
        mask = (grid > left + step * 0.5) & (grid < right - step * 0.5)
        out[mask] = np.nan
    return out


def normalize_window(w):
    """Remove the per-subcarrier static offset while preserving temporal changes."""
    w = np.asarray(w, dtype=np.float32)
    return w - np.mean(w, axis=0, keepdims=True)


def make_windows(series, win, hop):
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2 or win <= 0 or hop <= 0:
        raise ValueError("series must be 2D and win/hop must be positive")
    n = (len(series) - win) // hop + 1
    if n <= 0:
        return np.zeros((0, series.shape[1]), np.float32).reshape(0, win, series.shape[1])
    windows = []
    for start in range(0, len(series) - win + 1, hop):
        w = series[start:start + win]
        # Never train or infer on windows made mostly from interpolated packet gaps.
        if np.isfinite(w).all():
            windows.append(w)
    if not windows:
        return np.zeros((0, win, series.shape[1]), np.float32)
    return np.stack(windows).astype(np.float32)


# -------------------------------------------------------------- dataset
class DatasetError(Exception):
    pass


def split_takes(takes, seed=7):
    """Split BY RECORDING (never by overlapping windows), per label: ~70/15/15."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for lab, files in takes.items():
        if len(files) < 3:
            raise DatasetError(
                f"'{lab}' has only {len(files)} recording(s). Record at least 3 per activity; "
                "10-20 independent recordings is a better starting point.")
        f = files[:]
        rng.shuffle(f)
        n_test = max(1, round(len(f) * 0.15))
        n_val = max(1, round(len(f) * 0.15))
        split["test"] += [(p, lab) for p in f[:n_test]]
        split["val"] += [(p, lab) for p in f[n_test:n_test + n_val]]
        split["train"] += [(p, lab) for p in f[n_test + n_val:]]
    return split


def build_split(files, labels, fs, win, hop, normalize=True):
    """Build windows from complete recordings. No window can cross a recording boundary."""
    if not files:
        raise DatasetError("this split has no recordings")
    X, y = [], []
    for path, lab in files:
        try:
            with np.load(path, allow_pickle=False) as z:
                series = resample(z["ts"], z["amp"], fs)
        except Exception as exc:
            raise DatasetError(f"cannot read recording {os.path.basename(path)}: {exc}") from exc
        w = make_windows(series, win, hop)
        if len(w):
            X.append(np.stack([normalize_window(x) if normalize else x for x in w]))
            y.extend([labels.index(lab)] * len(w))
    if not X:
        raise DatasetError(f"no complete {win}-frame windows; record longer clips and check packet loss")
    return np.concatenate(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def load_dataset(root, fs=50, win=160, hop=25, normalize=True):
    takes = list_takes(root)
    if len(takes) < 2:
        raise DatasetError("Record at least 2 different activities first.")
    labels = sorted(takes)
    split = split_takes(takes)
    out = {k: build_split(v, labels, fs, win, hop, normalize) for k, v in split.items()}
    return labels, out, {k: len(v) for k, v in split.items()}


# ------------------------------------------- cross-room / cross-person split
GROUPS = ("room", "person")


def take_meta(path, label):
    with np.load(path, allow_pickle=False) as z:
        clean = lambda k: (str(z[k]).strip().lower() if k in z.files else "") or "unknown"
        return {"path": path, "label": label, "room": clean("room"), "person": clean("person")}


def list_takes_meta(root):
    return [take_meta(p, lab) for lab, files in list_takes(root).items() for p in files]


def group_values(metas, group):
    if group not in GROUPS:
        raise DatasetError(f"group must be one of {GROUPS}")
    return sorted({m[group] for m in metas})


def split_leave_one_group_out(metas, group, held, seed=7):
    """Test = every recording from a held-out room/person."""
    if group not in GROUPS:
        raise DatasetError(f"group must be one of {GROUPS}")
    test = [(m["path"], m["label"]) for m in metas if m[group] == held]
    if not test:
        raise DatasetError(f"no recordings with {group} = '{held}'")
    rng = random.Random(seed)
    by_label = {}
    for m in metas:
        if m[group] != held:
            by_label.setdefault(m["label"], []).append(m["path"])
    train, val = [], []
    for lab, files in sorted(by_label.items()):
        files = sorted(files)
        rng.shuffle(files)
        n_val = max(1, round(len(files) * 0.15)) if len(files) >= 2 else 0
        val += [(p, lab) for p in files[:n_val]]
        train += [(p, lab) for p in files[n_val:]]
    if not train or not val:
        raise DatasetError(f"not enough non-held-out {group} recordings for train/validation")
    train_labels = {lab for _, lab in train}
    missing = sorted({lab for _, lab in test} - train_labels)
    if missing:
        raise DatasetError(f"held-out {group} '{held}' has classes absent from training: {missing}")
    return {"train": train, "val": val, "test": test}


def load_fold(metas, group, held, labels, fs=50, win=160, hop=25, normalize=True, seed=7):
    split = split_leave_one_group_out(metas, group, held, seed)
    out = {k: build_split(v, labels, fs, win, hop, normalize) for k, v in split.items()}
    return out, {k: len(v) for k, v in split.items()}


def summary(root):
    """Recording counts/durations plus a simple quality snapshot for the dashboard."""
    res = {}
    for lab, files in list_takes(root).items():
        secs = 0.0
        frames = 0
        for p in files:
            try:
                with np.load(p, allow_pickle=False) as z:
                    t = _unwrap_u32_timestamps(z["ts"])
                    frames += len(t)
                    if len(t) > 1:
                        secs += max(0.0, (t[-1] - t[0]) / 1000.0)
            except Exception:
                continue
        res[lab] = {"takes": len(files), "seconds": round(secs), "frames": frames}
    return res


def delete_label(root, label):
    import shutil
    d = os.path.join(root, safe_name(label))
    if os.path.isdir(d):
        shutil.rmtree(d)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
