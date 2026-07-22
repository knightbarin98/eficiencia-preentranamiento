"""Bucle de entrenamiento por fold: early stopping + elección de umbral en val interno.

Idéntico para TODOS los comparadores (mismas augs/optimizador/épocas/tuning).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import ViewDataset, subset_index


def _make_loader(index_df, full_index, cfg, mode, shuffle, train=False):
    ds = ViewDataset(index_df, cfg, mode=mode, train=train)
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=False,
    )


def _build_optimizer(model, cfg):
    """Adam con LR discriminativo: backbone (todo menos fc) a lr*mult, head a lr."""
    tcfg = cfg["train"]
    lr = float(tcfg["lr"])
    wd = float(tcfg["weight_decay"])
    mult = float(tcfg.get("backbone_lr_mult", 1.0))
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("fc.") else backbone).append(p)
    groups = [
        {"params": backbone, "lr": lr * mult},
        {"params": head, "lr": lr},
    ]
    return torch.optim.Adam(groups, lr=lr, weight_decay=wd)


def _pos_weight(index_df: pd.DataFrame, device) -> torch.Tensor | None:
    n_pos = int((index_df["y_vista"] == 1).sum())
    n_neg = int((index_df["y_vista"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


def _predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, ys, idxs = [], [], []
    with torch.no_grad():
        for img, y, idx in loader:
            img = img.to(device)
            out = model(img).squeeze(-1)  # (B,)
            logits.append(out.detach().cpu().numpy().reshape(-1))
            ys.append(y.numpy().reshape(-1))
            idxs.append(np.asarray(idx).reshape(-1))
    return np.concatenate(logits), np.concatenate(ys), np.concatenate(idxs)


def _choose_threshold(logits, y, metric: str) -> float:
    """Umbral sobre probabilidades del val interno (youden|f1). Solo informativo."""
    probs = 1 / (1 + np.exp(-logits))
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(-probs)
    best_t, best_score = 0.5, -1.0
    for t in np.unique(probs):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        if metric == "f1":
            denom = 2 * tp + fp + fn
            score = (2 * tp / denom) if denom else 0.0
        else:  # youden = sens + spec - 1
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            score = sens + spec - 1
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def train_one_fold(cfg, full_index, fold, weights, model, device, logger):
    """Entrena un comparador en un fold. Devuelve (df_pred_test_por_vista, threshold)."""
    mode = cfg.get("mode", "toy")
    tr_df = subset_index(full_index, fold["train_patients"])
    va_df = subset_index(full_index, fold["val_patients"])
    te_df = subset_index(full_index, fold["test_patients"])

    train_loader = _make_loader(tr_df, full_index, cfg, mode, shuffle=True, train=True)
    val_loader = _make_loader(va_df, full_index, cfg, mode, shuffle=False, train=False)
    test_loader = _make_loader(te_df, full_index, cfg, mode, shuffle=False, train=False)

    model = model.to(device)
    pos_weight = _pos_weight(tr_df, device) if cfg["train"]["class_weighted_loss"] else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = _build_optimizer(model, cfg)

    best_auc, best_state, patience = -1.0, None, 0
    max_patience = int(cfg["train"]["early_stopping_patience"])
    min_epochs = int(cfg["train"].get("min_epochs", 0))

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        running = 0.0
        for img, y, _ in train_loader:
            img, y = img.to(device), y.to(device)
            opt.zero_grad()
            out = model(img).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * img.size(0)
        train_loss = running / max(len(tr_df), 1)

        # val interno
        v_logits, v_y, _ = _predict_logits(model, val_loader, device)
        v_auc = (
            roc_auc_score(v_y, v_logits) if len(np.unique(v_y)) == 2 else float("nan")
        )
        logger.info(
            f"  [{weights}] fold {fold['fold']} epoch {epoch}  "
            f"train_loss={train_loss:.4f}  val_auc={v_auc:.4f}"
        )

        score = v_auc if not np.isnan(v_auc) else -1.0
        if score > best_auc:
            best_auc, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= max_patience and epoch + 1 >= min_epochs:
                logger.info(f"  [{weights}] fold {fold['fold']} early stop @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # umbral en val interno (informativo; AUROC no depende de él)
    v_logits, v_y, _ = _predict_logits(model, val_loader, device)
    threshold = _choose_threshold(v_logits, v_y, cfg["train"]["threshold_metric"])

    # predicción OOF a nivel VISTA sobre el test del fold
    t_logits, t_y, t_idx = _predict_logits(model, test_loader, device)
    pred_df = te_df.iloc[t_idx][
        ["patient_id", "view", "knee_id", "side_class", "y_vista"]
    ].copy()
    pred_df["logit"] = t_logits
    pred_df["fold"] = fold["fold"]
    pred_df["weights"] = weights
    return pred_df.reset_index(drop=True), threshold
