"""Orquestador end-to-end: índice -> folds -> entrena cada comparador -> OOF nivel
rodilla -> AUROC/AUPRC + ΔAUROC pareado. Escribe todos los artefactos en el repo.

Uso:
    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --smoke   # 1 fold, override rápido

`run_once(...)` es reutilizado por run_multiseed.py (barrido de semillas).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import pandas as pd

from data import build_index, find_mixed_patients
from eval import aggregate_views_to_knee, knee_metrics, paired_bootstrap_delta_auroc
from model import build_model
from splits import check_no_leakage, make_folds, save_manifest
from train import train_one_fold
from utils import (
    dump_versions,
    get_logger,
    load_config,
    resolve_device,
    save_config,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 solo fold externo (test rápido)")
    return ap.parse_args()


def _json_safe_deltas(deltas: dict) -> dict:
    """Copia de los deltas sin el array bootstrap crudo (_deltas), para serializar."""
    out = {}
    for k, d in deltas.items():
        out[k] = {kk: vv for kk, vv in d.items() if kk != "_deltas"}
    return out


def run_once(cfg, seed, out_dir, logger, device, smoke=False, collect_boot=False,
             index=None, excl_df=None):
    """Una corrida CV completa (todos los comparadores) para una semilla.

    Devuelve dict con knee_oof (por comparador), metrics, deltas (con _deltas si
    collect_boot=True) y comparators. Escribe artefactos en out_dir. Si se pasa
    `index` (cohorte post-QC precomputado, seed-independiente) se reutiliza en vez
    de reconstruirlo (evita recomputar el QC por semilla).
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    save_config(cfg, os.path.join(out_dir, "config_used.yaml"))
    dump_versions(os.path.join(out_dir, "versions.json"))

    # 1) Índice -----------------------------------------------------------------
    if index is not None:
        if excl_df is not None:
            excl_df.to_csv(os.path.join(out_dir, "qc_exclusions.csv"), index=False)
    elif cfg.get("mode", "toy") == "real":
        from qc import build_qc_cohort

        index, excl_df = build_qc_cohort(cfg, verbose=True)
        excl_df.to_csv(os.path.join(out_dir, "qc_exclusions.csv"), index=False)
    else:
        index = build_index(cfg, seed)
    index.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    mixed = find_mixed_patients(index)
    logger.info(
        f"[seed {seed}] index: {len(index)} vistas | {index['knee_id'].nunique()} rodillas | "
        f"{index['patient_id'].nunique()} pacientes | mixtos={sorted(mixed)}"
    )
    assert len(mixed) >= 1, "Debe haber >=1 paciente mixto (ejercita la derivación knee_id)."

    # 2) Folds por paciente -----------------------------------------------------
    n_folds = int(cfg["splits"]["n_folds"])
    folds = make_folds(index, n_folds, int(cfg["splits"]["internal_val_folds"]), seed)
    check_no_leakage(folds, index)
    save_manifest(folds, index, os.path.join(out_dir, "folds_manifest.json"), seed)
    if smoke:
        folds = folds[:1]
        logger.info("SMOKE: usando solo el fold 0")

    # 3) Entrenar cada comparador ----------------------------------------------
    comparators = list(cfg["comparators"])
    knee_oof: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict] = {}

    for weights in comparators:
        logger.info(f"===== [seed {seed}] Comparador: {weights} =====")
        comp_dir = os.path.join(out_dir, weights)
        os.makedirs(comp_dir, exist_ok=True)
        view_preds_all = []

        for fold in folds:
            set_seed(seed + fold["fold"])  # semilla por fold, misma para todos los comparadores
            model = build_model(cfg, weights, logger=logger)
            pred_df, thr = train_one_fold(cfg, index, fold, weights, model, device, logger)
            view_preds_all.append(pred_df)
            knee_fold = aggregate_views_to_knee(pred_df)
            knee_fold["fold"] = fold["fold"]
            knee_fold["threshold"] = thr
            knee_fold.to_csv(
                os.path.join(comp_dir, f"fold_{fold['fold']}_preds.csv"), index=False
            )

        view_oof = pd.concat(view_preds_all, ignore_index=True)
        knee = aggregate_views_to_knee(view_oof)
        knee.to_csv(os.path.join(comp_dir, "oof_knee_preds.csv"), index=False)
        knee_oof[weights] = knee

        m = knee_metrics(knee)
        metrics[weights] = m
        with open(os.path.join(comp_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        logger.info(
            f"[seed {seed}][{weights}] OOF  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
            f"prev={m['prevalence']:.3f}  n_knees={m['n_knees']}"
        )

    # 4) ΔAUROC pareado entre comparadores -------------------------------------
    deltas = {}
    bcfg = cfg["eval"]["bootstrap"]
    for a, b in itertools.combinations(comparators, 2):
        common = sorted(set(knee_oof[a]["knee_id"]) & set(knee_oof[b]["knee_id"]))
        ka = knee_oof[a][knee_oof[a]["knee_id"].isin(common)]
        kb = knee_oof[b][knee_oof[b]["knee_id"].isin(common)]
        d = paired_bootstrap_delta_auroc(
            ka, kb, n_boot=int(bcfg["n_boot"]), ci=float(bcfg["ci"]), seed=seed,
            return_deltas=collect_boot,
        )
        d["A"], d["B"] = a, b
        deltas[f"{a}_vs_{b}"] = d
        logger.info(
            f"[seed {seed}] ΔAUROC {a}-{b}: obs={d['delta_auroc_observed']:+.4f}  "
            f"IC{int(bcfg['ci']*100)}%=[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p_boot={d['p_bootstrap']:.4f}"
        )

    summary = {
        "run_name": cfg["run_name"],
        "mode": cfg.get("mode"),
        "seed": seed,
        "n_folds": len(folds),
        "comparators": comparators,
        "metrics": metrics,
        "delta_auroc": _json_safe_deltas(deltas),
    }
    with open(os.path.join(out_dir, "delta_auroc.json"), "w") as f:
        json.dump(_json_safe_deltas(deltas), f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "knee_oof": knee_oof,
        "metrics": metrics,
        "deltas": deltas,
        "comparators": comparators,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])

    out_dir = os.path.join(cfg["paths"]["output_root"], cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("ropec", logfile=os.path.join(out_dir, "run.log"))
    device = resolve_device(cfg.get("device", "auto"))
    logger.info(f"device={device}  run={cfg['run_name']}  mode={cfg.get('mode')}")

    import torch  # noqa: F401  (garantiza fallo temprano y claro si falta torch)

    run_once(cfg, seed, out_dir, logger, device, smoke=args.smoke)
    logger.info(f"Artefactos en: {out_dir}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
