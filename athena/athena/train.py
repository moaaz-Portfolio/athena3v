"""Reproducible, class-balanced training and held-out evaluation for Athena."""
import copy
import os
import time

import numpy as np

from .data import FEATURES, load_dataset, write_json


DEFAULT_FALL_THRESHOLD = 0.72


def confusion(y, p, n):
    m = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(np.asarray(y).tolist(), np.asarray(p).tolist()):
        m[int(a), int(b)] += 1
    return m


def scores(m):
    m = np.asarray(m, dtype=np.float64)
    tp = np.diag(m)
    prec = tp / np.maximum(m.sum(0), 1)
    rec = tp / np.maximum(m.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    acc = float(tp.sum() / max(m.sum(), 1))
    return prec, rec, f1, acc


def _macro_f1(y, pred, n):
    return float(scores(confusion(y, pred, n))[2].mean())


def _temperature_scale(logits, y):
    """Small deterministic validation-only temperature search (no test-set tuning)."""
    import torch
    if len(y) < 8:
        return 1.0
    x = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(y, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    best_t, best_loss = 1.0, float("inf")
    for t in np.geomspace(0.5, 3.0, 36):
        loss = float(criterion(x / float(t), target).item())
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def _fall_threshold(probs, y, labels, fallback=DEFAULT_FALL_THRESHOLD):
    """Choose a conservative fall threshold on validation only when enough examples exist."""
    if "fall" not in labels:
        return float(fallback)
    k = labels.index("fall")
    target = (np.asarray(y) == k).astype(np.int32)
    p = np.asarray(probs)[:, k]
    if int(target.sum()) < 4 or int((1 - target).sum()) < 8:
        return float(fallback)
    best = (0.0, float(fallback))
    for threshold in np.arange(0.50, 0.951, 0.01):
        pred = p >= threshold
        tp = int(np.sum(pred & (target == 1)))
        fp = int(np.sum(pred & (target == 0)))
        fn = int(np.sum((~pred) & (target == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        # Prefer thresholds that avoid an unbounded false-alarm rate on validation.
        if precision >= 0.70 and f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def _predict_logits(model, X, device, torch, batch_size=256):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunks.append(model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _augment(x, torch):
    """Signal-preserving augmentation. Avoids circular shifts that create fake transitions."""
    b, t, f = x.shape
    sample_std = x.std(dim=(1, 2), keepdim=True).clamp_min(0.05)
    gain = 0.85 + 0.30 * torch.rand((b, 1, 1), device=x.device)
    x = x * gain
    x = x + torch.randn_like(x) * sample_std * (0.015 + 0.045 * torch.rand((b, 1, 1), device=x.device))

    # Brief time masking simulates a short unreliable CSI interval without shifting the event.
    if t >= 20 and torch.rand(()) < 0.45:
        width = int(torch.randint(2, max(3, min(13, t // 8)), (1,)).item())
        start = int(torch.randint(0, max(1, t - width), (1,)).item())
        x[:, start:start + width, :] *= 0.15
    # A few missing subcarriers simulate frequency-selective fading.
    if f >= 16 and torch.rand(()) < 0.35:
        count = max(1, f // 32)
        idx = torch.randperm(f, device=x.device)[:count]
        x[:, :, idx] *= 0.5
    return x


def _fall_metrics(m, labels):
    if "fall" not in labels:
        return None
    i = labels.index("fall")
    tp = int(m[i, i])
    fn = int(m[i, :].sum() - tp)
    fp = int(m[:, i].sum() - tp)
    tn = int(m.sum() - tp - fn - fp)
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "false_positive_windows": fp,
        "true_negative_windows": tn,
        "false_negative_windows": fn,
        "true_positive_windows": tp,
    }


def train(root, out_dir, epochs=60, fs=50, win=160, hop=25, init=None,
          log=print, progress=None, should_stop=lambda: False,
          dataset=None, normalize=True, save=True, seed=42, patience=12):
    """Train with recording-level splits; test remains untouched until final scoring.

    dataset=(labels, {train,val,test: (X,y)}, counts) is used by crossval.py.
    """
    import torch
    import torch.nn as nn
    from .model import ActivityNet, load_pretrained

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    labels, data, counts = dataset if dataset is not None else load_dataset(root, fs, win, hop, normalize)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    if len(set(ytr.tolist())) < 2:
        raise RuntimeError("training split must contain at least two classes")
    log(f"recordings train/val/test: {counts['train']}/{counts['val']}/{counts['test']} | "
        f"windows: {len(Xtr)}/{len(Xva)}/{len(Xte)} | classes: {labels}")
    if "fall" in labels:
        log("fall class found: using a longer temporal window and a validation-only alarm threshold")

    scale = float(np.std(Xtr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    Xtr = np.ascontiguousarray(Xtr / scale, dtype=np.float32)
    Xva = np.ascontiguousarray(Xva / scale, dtype=np.float32)
    Xte = np.ascontiguousarray(Xte / scale, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActivityNet(FEATURES, len(labels)).to(device)
    if init and os.path.exists(init):
        copied = load_pretrained(model, init)
        log(f"warm-started {copied} compatible encoder tensors from the legacy checkpoint")

    counts_per_class = np.bincount(ytr, minlength=len(labels)).astype(np.float32)
    # Sqrt inverse frequency is less extreme than raw inverse-frequency weighting.
    class_weights = 1.0 / np.sqrt(np.maximum(counts_per_class, 1.0))
    class_weights /= class_weights.mean()
    class_weights = np.clip(class_weights, 0.55, 2.5)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.025,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=2e-5)

    xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    batch_size = min(96, max(16, len(Xtr)))
    best_state, best_f1, best_epoch = None, -1.0, 0
    stale = 0
    history = []

    for ep in range(1, epochs + 1):
        if should_stop():
            log("training stopped by user; restoring the best validation checkpoint")
            break
        model.train()
        perm = torch.randperm(len(xtr_t), device=device)
        total_loss = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xb = _augment(xtr_t[idx].clone(), torch)
            yb = ytr_t[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
        scheduler.step()

        val_logits = _predict_logits(model, Xva, device, torch)
        val_pred = val_logits.argmax(axis=1)
        val_f1 = _macro_f1(yva, val_pred, len(labels))
        val_acc = float(np.mean(val_pred == yva))
        mean_loss = total_loss / max(len(perm), 1)
        history.append({"epoch": ep, "loss": mean_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        log(f"epoch {ep:>3}/{epochs} | loss {mean_loss:.4f} | val acc {val_acc:.3f} | "
            f"val macro-F1 {val_f1:.3f} | lr {optimizer.param_groups[0]['lr']:.2g}")
        if progress:
            progress(ep, epochs)
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_epoch = ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= max(3, int(patience)):
            log(f"early stopping: validation did not improve for {patience} epochs")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    # Calibrate only on validation data. Never inspect test labels for threshold selection.
    val_logits = _predict_logits(model, Xva, device, torch)
    temperature = _temperature_scale(val_logits, yva)
    val_probs = torch.softmax(torch.from_numpy(val_logits) / temperature, dim=1).numpy()
    fall_threshold = _fall_threshold(val_probs, yva, labels)

    test_logits = _predict_logits(model, Xte, device, torch)
    test_probs = torch.softmax(torch.from_numpy(test_logits) / temperature, dim=1).numpy()
    pred = test_probs.argmax(axis=1)
    if "fall" in labels:
        fall_i = labels.index("fall")
        pred[test_probs[:, fall_i] >= fall_threshold] = fall_i
    mt = confusion(yte, pred, len(labels))
    pr, rc, f1, acc = scores(mt)
    macro_f1 = float(f1.mean())

    ckpt = {
        "version": "athena-fall-2.0",
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": labels, "feature_count": FEATURES, "window_length": win,
        "fs": fs, "hop": hop, "scale": scale, "confidence_threshold": 0.50,
        "fall_threshold": fall_threshold, "fall_confirm_windows": 2,
        "temperature": temperature, "normalize": normalize,
    }
    metrics = {
        "model_version": ckpt["version"], "normalize": normalize, "labels": labels,
        "fs": fs, "window_length": win, "hop": hop, "recordings": counts,
        "windows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "temperature": temperature,
        "fall_threshold": fall_threshold,
        "test": {
            "accuracy": acc, "macro_f1": macro_f1,
            "balanced_accuracy": float(np.mean(rc)),
            "per_class": {lab: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i])}
                           for i, lab in enumerate(labels)},
            "fall": _fall_metrics(mt, labels),
            "confusion_matrix": mt.tolist(),
        },
        "history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warning": "Research prototype only. Test recordings are independent recordings, but may share room/person/device unless group-held-out evaluation is run. Never use as a life-safety alarm.",
    }
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "activity_model.pt")
        tmp_path = path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        write_json(os.path.join(out_dir, "metrics.json"), metrics)
        log(f"TEST accuracy {acc:.3f} | macro-F1 {macro_f1:.3f} | saved {path}")
        if metrics["test"]["fall"] is not None:
            fm = metrics["test"]["fall"]
            log(f"FALL test precision {fm['precision']:.3f} | recall {fm['recall']:.3f} | "
                f"F1 {fm['f1']:.3f} | threshold {fall_threshold:.2f}")
    else:
        log(f"TEST accuracy {acc:.3f} | macro-F1 {macro_f1:.3f}")
    return metrics
