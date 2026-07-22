"""Control de calidad (QC) con criterios CONGELADOS antes de mirar métricas.

Criterios (id, criterio) -> qc_exclusions.csv + diagrama de flujo qc_flow.json:

  AUTO-DETECTABLES desde los datos (implementados aquí):
    - resolucion_minima        : min(H,W) de OriginalImage < qc.min_resolution
    - concordancia_mascara     : BW ausente / forma != imagen / máscara vacía o llena
    - duplicado                : near-duplicate por pHash (hamming <= umbral); se
                                 conserva la 1ª aparición, se excluyen las siguientes
    - vista_admitida           : garantizado por construcción (solo imN entran al
                                 índice; Coronal_CT ya excluido en data.py)

  CLÍNICOS / SUBJETIVOS (NO auto-detectables — revisión manual, congelada):
    - platillo_no_visible, lado_no_identificable, hardware_posoperatorio
    Se leen de qc.manual_exclusions_csv si se provee. Por defecto NO hay ninguna
    (no se inventan exclusiones). El criterio queda documentado y aplicable.

Regla: los criterios se congelan ANTES de ver métricas; no se excluyen casos
difíciles porque el modelo falle.

Flujo: conjunto completo (186 pac / 190 rodillas / 421 vistas) -> tras QC.
Una vista excluida no tira su rodilla si quedan otras vistas; una rodilla cae si
TODAS sus vistas se excluyen; un paciente cae si todas sus rodillas caen.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from data import (
    build_real_index,
    load_platif_struct,
    read_view_image,
)
from utils import load_config


# --------------------------------------------------------------------------- #
# pHash (DCT) 64-bit en numpy — sin dependencia extra (imagehash no requerido)
# --------------------------------------------------------------------------- #
def _phash(img: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    from scipy.fftpack import dct

    n = hash_size * highfreq_factor
    # normaliza a [0,255] y redimensiona por muestreo simple
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        hi = lo + 1e-6
    im = np.clip((img - lo) / (hi - lo), 0, 1)
    # resize por interpolación en malla regular
    ys = np.linspace(0, im.shape[0] - 1, n).astype(int)
    xs = np.linspace(0, im.shape[1] - 1, n).astype(int)
    small = im[np.ix_(ys, xs)]
    d = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    dlow = d[:hash_size, :hash_size]
    med = np.median(dlow.flatten()[1:])  # mediana excluyendo el término DC [0,0]
    bits = (dlow > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# QC principal
# --------------------------------------------------------------------------- #
def _read_view_arrays(struct, view):
    v = getattr(struct, view)
    img = read_view_image(struct, view)
    bw = getattr(v, "BW", None)
    bw = np.asarray(bw) if bw is not None else None
    return img, bw


def run_qc(cfg: dict, index_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    qc = cfg["qc"]
    min_res = int(qc["min_resolution"])
    dup_thr = int(qc["dup_hamming_threshold"])
    dup_scope = str(qc.get("dup_scope", "cross_patient"))
    check_mask = bool(qc.get("check_mask_concordance", True))

    exclusions = []          # dicts: patient_id, view, knee_id, criterio, detail
    hashes = []              # (phash, patient_id, view) de las vistas conservadas

    # Recorre por archivo para reusar el struct cargado
    for mat_path, grp in index_df.groupby("mat_path"):
        struct = load_platif_struct(mat_path)
        for _, row in grp.iterrows():
            pid, view, knee = int(row["patient_id"]), row["view"], row["knee_id"]
            img, bw = _read_view_arrays(struct, view)

            # 1) resolución mínima
            if min(img.shape[:2]) < min_res:
                exclusions.append(dict(patient_id=pid, view=view, knee_id=knee,
                                       criterio="resolucion_minima",
                                       detail=f"{img.shape[0]}x{img.shape[1]}"))
                continue

            # 2) concordancia máscara
            if check_mask:
                bad = None
                if bw is None:
                    bad = "BW ausente"
                elif bw.shape[:2] != img.shape[:2]:
                    bad = f"forma BW {bw.shape} != img {img.shape}"
                else:
                    frac = float((bw > 0).mean())
                    if frac <= 0.0 or frac >= 1.0:
                        bad = f"mascara trivial (frac_fg={frac:.3f})"
                if bad is not None:
                    exclusions.append(dict(patient_id=pid, view=view, knee_id=knee,
                                           criterio="concordancia_mascara", detail=bad))
                    continue

            # 3) duplicados near-duplicate por pHash.
            #    Alcance cross_patient: SOLO se excluye si el match es de OTRO paciente
            #    (riesgo de fuga). Intra-paciente no se excluye (contralateral/multi-vista
            #    son datos legítimos y no hay fuga: los folds son por paciente).
            h = _phash(img)
            dup_of = None
            for (ph, dpid, dview) in hashes:
                if _hamming(h, ph) <= dup_thr:
                    if dup_scope == "cross_patient" and dpid == pid:
                        continue  # match intra-paciente -> no es fuga, no excluir
                    dup_of = f"{dpid}:{dview}"
                    break
            if dup_of is not None:
                exclusions.append(dict(patient_id=pid, view=view, knee_id=knee,
                                       criterio="duplicado", detail=f"~{dup_of}"))
                continue

            hashes.append((h, pid, view))

    # 4) exclusiones clínicas manuales (revisor) — congeladas, opcional
    man_csv = qc.get("manual_exclusions_csv")
    if man_csv and os.path.exists(man_csv):
        man = pd.read_csv(man_csv)
        for _, r in man.iterrows():
            pid, view = int(r["patient_id"]), str(r["view"])
            match = index_df[(index_df.patient_id == pid) & (index_df.view == view)]
            knee = match["knee_id"].iloc[0] if len(match) else f"{pid}:?"
            exclusions.append(dict(patient_id=pid, view=view, knee_id=knee,
                                   criterio=str(r.get("criterio", "manual")),
                                   detail="revisor"))

    excl_df = pd.DataFrame(exclusions, columns=[
        "patient_id", "view", "knee_id", "criterio", "detail"])
    flow = _flow_diagram(index_df, excl_df)
    return excl_df, flow


def _flow_diagram(index_df: pd.DataFrame, excl_df: pd.DataFrame) -> dict:
    excluded_views = set(zip(excl_df["patient_id"], excl_df["view"])) if len(excl_df) else set()
    keep = index_df[~index_df.apply(
        lambda r: (r["patient_id"], r["view"]) in excluded_views, axis=1)]

    def counts(df):
        knees = df.drop_duplicates("knee_id")
        return {
            "patients": int(df["patient_id"].nunique()),
            "knees": int(df["knee_id"].nunique()),
            "views": int(len(df)),
            "knee_pos": int((knees["y_vista"] == 1).sum()),
            "knee_neg": int((knees["y_vista"] == 0).sum()),
        }

    by_crit = (excl_df.groupby("criterio").size().to_dict() if len(excl_df) else {})
    return {
        "full": counts(index_df),
        "after_qc": counts(keep),
        "views_excluded": int(len(excl_df)),
        "views_excluded_by_criterio": by_crit,
        "knees_dropped": counts(index_df)["knees"] - counts(keep)["knees"],
        "patients_dropped": counts(index_df)["patients"] - counts(keep)["patients"],
    }


def apply_exclusions(index_df: pd.DataFrame, excl_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el índice sin las vistas excluidas por QC."""
    if len(excl_df) == 0:
        return index_df.reset_index(drop=True)
    excluded = set(zip(excl_df["patient_id"], excl_df["view"]))
    keep = index_df[~index_df.apply(
        lambda r: (r["patient_id"], r["view"]) in excluded, axis=1)]
    return keep.reset_index(drop=True)


def build_qc_cohort(cfg: dict, verbose: bool = False):
    """Índice real -> QC -> índice post-QC (cohorte de análisis). Devuelve (index, excl)."""
    index = build_real_index(cfg, seed=int(cfg["seed"]), run_assertions=True)
    excl_df, flow = run_qc(cfg, index)
    cohort = apply_exclusions(index, excl_df)
    if verbose:
        print(f"[QC] cohorte: {cohort['patient_id'].nunique()} pac / "
              f"{cohort['knee_id'].nunique()} rodillas / {len(cohort)} vistas "
              f"(excluidas {len(excl_df)} vistas)")
    return cohort, excl_df


def main():
    ap = argparse.ArgumentParser(description="QC de PlaTiF (criterios congelados).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg["paths"]["output_root"]
    os.makedirs(out_dir, exist_ok=True)

    index = build_real_index(cfg, seed=int(cfg["seed"]), run_assertions=True)
    excl_df, flow = run_qc(cfg, index)

    excl_path = os.path.join(out_dir, "qc_exclusions.csv")
    flow_path = os.path.join(out_dir, "qc_flow.json")
    excl_df.to_csv(excl_path, index=False)
    with open(flow_path, "w") as f:
        json.dump(flow, f, indent=2)

    print("\n=== Diagrama de flujo QC ===")
    print(f"  completo : {flow['full']['patients']} pac / {flow['full']['knees']} rodillas "
          f"/ {flow['full']['views']} vistas")
    print(f"  excluidas: {flow['views_excluded']} vistas  {flow['views_excluded_by_criterio']}")
    print(f"  tras QC  : {flow['after_qc']['patients']} pac / {flow['after_qc']['knees']} rodillas "
          f"({flow['after_qc']['knee_pos']} fx / {flow['after_qc']['knee_neg']} normal) "
          f"/ {flow['after_qc']['views']} vistas")
    print(f"  rodillas caídas: {flow['knees_dropped']}  pacientes caídos: {flow['patients_dropped']}")
    print(f"\nqc_exclusions.csv -> {excl_path}  ({len(excl_df)} filas)")
    print(f"qc_flow.json      -> {flow_path}")


if __name__ == "__main__":
    main()
