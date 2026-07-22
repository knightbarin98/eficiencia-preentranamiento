"""Evaluación a nivel RODILLA: agregación vista->rodilla, AUROC/AUPRC y ΔAUROC
pareado remuestreando PACIENTES completos.

Agregación CONGELADA: media de logits dentro de cada knee_id (los 4 pacientes
mixtos NO se promedian entre sus dos rodillas: cada knee_id es su propia unidad).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate_views_to_knee(view_pred_df: pd.DataFrame) -> pd.DataFrame:
    """Media de logits por knee_id -> una predicción por rodilla.

    knee_id codifica (patient_id, lado); patient_id se recupera para el bootstrap.
    """
    g = view_pred_df.groupby("knee_id")
    out = g.agg(
        patient_id=("patient_id", "first"),
        side_class=("side_class", "first"),
        y=("y_vista", "first"),
        logit=("logit", "mean"),
        n_views=("logit", "size"),
    ).reset_index()
    # sanity: y constante dentro de cada knee_id
    yvar = g["y_vista"].nunique()
    bad = yvar[yvar > 1]
    assert bad.empty, f"knee_id con etiqueta inconsistente: {list(bad.index)}"
    out["prob"] = 1 / (1 + np.exp(-out["logit"]))
    return out


def knee_metrics(knee_df: pd.DataFrame) -> dict:
    y = knee_df["y"].values.astype(int)
    score = knee_df["logit"].values  # AUROC es invariante a la sigmoide
    prob = knee_df["prob"].values
    return {
        "n_knees": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, prob))
        if len(np.unique(y)) == 2
        else float("nan"),
    }


def paired_bootstrap_delta_auroc(
    knee_a: pd.DataFrame,
    knee_b: pd.DataFrame,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    return_deltas: bool = False,
) -> dict:
    """ΔAUROC = AUROC(A) - AUROC(B), IC por bootstrap remuestreando PACIENTES.

    Cada paciente arrastra sus 1-2 rodillas juntas. Mismo resample para A y B
    (pareado). knee_a/knee_b deben cubrir el mismo conjunto de knee_id.
    """
    a = knee_a.set_index("knee_id").sort_index()
    b = knee_b.set_index("knee_id").sort_index()
    assert list(a.index) == list(b.index), "A y B no comparten el mismo set de rodillas"
    assert np.array_equal(a["y"].values, b["y"].values), "Etiquetas desalineadas A vs B"

    y = a["y"].values.astype(int)
    sa = a["logit"].values
    sb = b["logit"].values
    patient_of_knee = a["patient_id"].values

    # Mapa paciente -> índices de sus rodillas
    patients = np.array(sorted(set(int(p) for p in patient_of_knee)))
    knees_by_pat = {p: np.where(patient_of_knee == p)[0] for p in patients}

    obs = float(roc_auc_score(y, sa)) - float(roc_auc_score(y, sb))

    rng = np.random.default_rng(seed)
    deltas = []
    n_pat = len(patients)
    for _ in range(n_boot):
        sampled = patients[rng.integers(0, n_pat, size=n_pat)]
        idx = np.concatenate([knees_by_pat[p] for p in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        d = roc_auc_score(yy, sa[idx]) - roc_auc_score(yy, sb[idx])
        deltas.append(d)

    deltas = np.asarray(deltas)
    out = {
        "comparison": "A - B",
        "delta_auroc_observed": obs,
        "delta_auroc_boot_mean": float(deltas.mean()) if len(deltas) else float("nan"),
        "ci": ci,
        "ci_low": pooled_ci(deltas, ci)[0],
        "ci_high": pooled_ci(deltas, ci)[1],
        "p_bootstrap": bootstrap_pvalue(deltas),
        "n_boot_effective": int(len(deltas)),
        "n_patients": int(n_pat),
    }
    if return_deltas:
        out["_deltas"] = deltas
    return out


def pooled_ci(deltas: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    """IC percentil sobre una distribución de deltas (una o varias semillas agrupadas)."""
    if len(deltas) == 0:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.quantile(deltas, alpha)), float(np.quantile(deltas, 1 - alpha))


def bootstrap_pvalue(deltas: np.ndarray) -> float:
    """p-valor bootstrap de dos colas para H0: ΔAUROC = 0 (cluster-robusto)."""
    if len(deltas) == 0:
        return float("nan")
    frac_neg = float((deltas < 0).mean())
    frac_pos = float((deltas > 0).mean())
    return float(min(1.0, 2.0 * min(frac_neg, frac_pos)))
