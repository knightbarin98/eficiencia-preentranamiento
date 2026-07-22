"""Particiones por PACIENTE: StratifiedGroupKFold(5) sin fuga.

Invariante: todas las vistas/rodillas de un paciente en el mismo fold. Dentro del
train de cada fold, un split interno (grupo/estratificado) separa val para early
stopping y elección de umbral. Se versiona folds_manifest.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _patient_labels(index_df: pd.DataFrame) -> pd.Series:
    """Etiqueta a nivel paciente para estratificar: positivo si tiene alguna vista fx."""
    return index_df.groupby("patient_id")["y_vista"].max()


def make_folds(index_df: pd.DataFrame, n_folds: int, internal_val_folds: int, seed: int):
    """Devuelve lista de dicts por fold con listas de patient_id train/val/test."""
    df = index_df.reset_index(drop=True)
    X = np.zeros(len(df))
    y = df["y_vista"].values
    groups = df["patient_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        test_patients = sorted(set(int(p) for p in groups[te_idx]))
        train_all = sorted(set(int(p) for p in groups[tr_idx]))

        # Split interno para val (grupo por paciente, estratificado)
        tr_df = df[df["patient_id"].isin(train_all)].reset_index(drop=True)
        inner = StratifiedGroupKFold(
            n_splits=internal_val_folds, shuffle=True, random_state=seed + 1
        )
        i_tr, i_va = next(
            inner.split(np.zeros(len(tr_df)), tr_df["y_vista"].values, tr_df["patient_id"].values)
        )
        val_patients = sorted(set(int(p) for p in tr_df["patient_id"].values[i_va]))
        train_patients = sorted(set(train_all) - set(val_patients))

        folds.append(
            {
                "fold": k,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "test_patients": test_patients,
                "prevalence": _fold_prevalence(df, train_patients, val_patients, test_patients),
            }
        )
    return folds


def _prev(df: pd.DataFrame, patients) -> dict:
    sub = df[df["patient_id"].isin(patients)]
    pat_lab = sub.groupby("patient_id")["y_vista"].max()
    knee = sub.drop_duplicates("knee_id")
    return {
        "n_patients": int(sub["patient_id"].nunique()),
        "n_knees": int(sub["knee_id"].nunique()),
        "n_views": int(len(sub)),
        "patient_pos_rate": float(pat_lab.mean()) if len(pat_lab) else 0.0,
        "knee_pos_rate": float(knee["y_vista"].mean()) if len(knee) else 0.0,
    }


def _fold_prevalence(df, train_p, val_p, test_p) -> dict:
    return {"train": _prev(df, train_p), "val": _prev(df, val_p), "test": _prev(df, test_p)}


def check_no_leakage(folds: list[dict], index_df: pd.DataFrame) -> None:
    """Aserciones de cero fuga. Levanta AssertionError si algo falla."""
    all_patients = set(int(p) for p in index_df["patient_id"].unique())

    # 1) Cada paciente en exactamente un fold de TEST
    test_counts: dict[int, int] = {}
    covered = set()
    for f in folds:
        for p in f["test_patients"]:
            test_counts[p] = test_counts.get(p, 0) + 1
            covered.add(p)
    dupes = [p for p, c in test_counts.items() if c > 1]
    assert not dupes, f"Pacientes en >1 fold de test: {dupes}"
    assert covered == all_patients, (
        f"Cobertura de test incompleta. Faltan: {sorted(all_patients - covered)}"
    )

    # 2) Dentro de un fold, train/val/test disjuntos; val ⊂ complemento de test
    for f in folds:
        tr, va, te = set(f["train_patients"]), set(f["val_patients"]), set(f["test_patients"])
        assert not (tr & te), f"Fold {f['fold']}: solape train/test"
        assert not (va & te), f"Fold {f['fold']}: solape val/test"
        assert not (tr & va), f"Fold {f['fold']}: solape train/val"

    # 3) Ninguna rodilla huérfana: cada knee_id de un paciente cae con su paciente
    #    (garantizado por agrupar por patient_id; se verifica que no haya knee_id
    #     compartido entre pacientes)
    knee_owner = index_df.groupby("knee_id")["patient_id"].nunique()
    shared = knee_owner[knee_owner > 1]
    assert shared.empty, f"knee_id compartido entre pacientes: {list(shared.index)}"


def save_manifest(folds: list[dict], index_df: pd.DataFrame, path: str, seed: int) -> None:
    manifest = {
        "seed": seed,
        "n_folds": len(folds),
        "n_patients": int(index_df["patient_id"].nunique()),
        "n_knees": int(index_df["knee_id"].nunique()),
        "n_views": int(len(index_df)),
        "folds": folds,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------- #
# CLI: genera folds_manifest.json sobre el cohorte real post-QC (Prompt 1.3)
# --------------------------------------------------------------------------- #
def _main():
    import argparse

    from qc import build_qc_cohort
    from utils import load_config, set_seed

    ap = argparse.ArgumentParser(description="Genera splits por paciente (post-QC).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)

    cohort, _ = build_qc_cohort(cfg, verbose=True)
    n_folds = int(cfg["splits"]["n_folds"])
    folds = make_folds(cohort, n_folds, int(cfg["splits"]["internal_val_folds"]), seed)
    check_no_leakage(folds, cohort)  # aborta si hay fuga

    out = args.out or os.path.join(cfg["paths"]["output_root"], "folds_manifest.json")
    save_manifest(folds, cohort, out, seed)

    print("\n=== Prevalencia por fold (test) ===")
    print(f"{'fold':>4} {'pac':>4} {'rodillas':>9} {'fx':>4} {'neg':>4} {'prev':>6}")
    for f in folds:
        t = f["prevalence"]["test"]
        n_pos = round(t["knee_pos_rate"] * t["n_knees"])
        print(f"{f['fold']:>4} {t['n_patients']:>4} {t['n_knees']:>9} "
              f"{n_pos:>4} {t['n_knees']-n_pos:>4} {t['knee_pos_rate']:>6.3f}")
    print("\n✅ cero fuga verificada. folds_manifest.json ->", out)


if __name__ == "__main__":
    _main()
