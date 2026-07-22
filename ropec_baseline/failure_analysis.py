"""Fase 5 (Prompt 5.1): análisis de fallos.

Sobre las predicciones OOF a nivel rodilla del comparador elegido (por defecto
RadImageNet, semilla base), en un umbral de operación (Youden sobre el OOF):
  - matriz de confusión a nivel rodilla,
  - falsos negativos (fracturas perdidas) desglosados por tipo Schatzker (del xlsx),
  - nº de vistas por rodilla en aciertos vs. fallos (proxy de dificultad).

Uso:
    python failure_analysis.py --config config.yaml \
        --oof outputs/multiseed_100/seed_1337/radimagenet/oof_knee_preds.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data import _find_col  # reutiliza el localizador de columnas del xlsx


def youden_threshold(y, prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--oof", default="outputs/multiseed_100/seed_1337/radimagenet/oof_knee_preds.csv")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    xlsx_path = cfg["paths"]["platif_xlsx"]

    oof = pd.read_csv(args.oof)
    y = oof["y"].astype(int).values
    prob = oof["prob"].values
    thr = youden_threshold(y, prob)
    pred = (prob >= thr).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"OOF: {args.oof}")
    print(f"umbral Youden (prob) = {thr:.3f}")
    print("=== Matriz de confusión a nivel rodilla ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  sensibilidad={sens:.3f}  especificidad={spec:.3f}  n_rodillas={len(oof)}")

    # nº de vistas por rodilla: aciertos vs fallos (si la columna existe)
    if "n_views" in oof.columns:
        oof = oof.assign(correct=(pred == y))
        by = oof.groupby("correct")["n_views"].mean()
        print("\n=== nº medio de vistas por rodilla ===")
        for k, v in by.items():
            print(f"  {'acierto' if k else 'fallo'}: {v:.2f} vistas/rodilla")

    # Falsos negativos por tipo Schatzker (join con xlsx a nivel paciente)
    fn_rows = oof[(pred == 0) & (y == 1)].copy()
    if len(fn_rows) == 0:
        print("\nSin falsos negativos en este umbral.")
        return
    try:
        meta = pd.read_excel(xlsx_path, engine="openpyxl")
        id_col = _find_col(meta.columns, "patient", "id") or meta.columns[0]
        sch_col = _find_col(meta.columns, "schatzker") or _find_col(meta.columns, "fracture", "type")
        meta = meta.rename(columns={id_col: "patient_id"})
        meta["patient_id"] = pd.to_numeric(meta["patient_id"], errors="coerce").astype("Int64")
        fn_rows = fn_rows.merge(meta[["patient_id", sch_col]], on="patient_id", how="left")
        print(f"\n=== Falsos negativos (fracturas perdidas): {len(fn_rows)} rodillas ===")
        counts = fn_rows[sch_col].astype(str).value_counts()
        for k, v in counts.items():
            print(f"  {k}: {v}")
        print("\nDetalle (patient_id, knee_id, prob, schatzker):")
        for _, r in fn_rows.iterrows():
            print(f"  P{int(r['patient_id'])}  {r['knee_id']}  prob={r['prob']:.3f}  {r[sch_col]}")
    except Exception as e:  # el xlsx es opcional para el desglose
        print(f"\n(No pude cruzar con xlsx: {type(e).__name__}: {e})")
        print(f"Falsos negativos: {len(fn_rows)} rodillas (sin desglose Schatzker)")


if __name__ == "__main__":
    main()
