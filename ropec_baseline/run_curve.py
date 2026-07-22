"""Fase 3 — Curva de eficiencia de etiquetas.

Subconjuntos ANIDADOS de train por paciente (10⊂25⊂50⊂100%), estratificados, los
MISMOS para los 3 comparadores. >=3 semillas por fracción. Analiza la interacción
procedencia×fracción: ¿se ensancha RadImageNet−ImageNet al bajar etiquetas? y da
equivalencias de datos (p.ej. RadImageNet@F ≈ random@100%).

Uso:
    python run_curve.py --config config.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from eval import (
    aggregate_views_to_knee,
    bootstrap_pvalue,
    knee_metrics,
    paired_bootstrap_delta_auroc,
    pooled_ci,
)
from model import build_model
from splits import (
    check_no_leakage,
    make_folds,
    nested_stratified_subsets,
    patient_positivity,
)
from train import train_one_fold
from utils import get_logger, load_config, resolve_device, set_seed


def _mean_std(xs):
    a = np.asarray([x for x in xs if x == x], dtype=float)
    return (float(a.mean()) if len(a) else float("nan"),
            float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Curva de eficiencia de etiquetas (Fase 3).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 semilla, 1 fold (valida plumbing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fractions = sorted(float(f) for f in cfg["curve"]["fractions"])
    seeds = list(cfg["curve"]["seeds"])
    if args.smoke:
        seeds = seeds[:1]
    comparators = list(cfg["comparators"])
    base_dir = os.path.join(cfg["paths"]["output_root"], "curve")
    os.makedirs(base_dir, exist_ok=True)
    logger = get_logger("ropec-curve", logfile=os.path.join(base_dir, "curve.log"))
    device = resolve_device(cfg.get("device", "auto"))
    bcfg = cfg["eval"]["bootstrap"]
    ci = float(bcfg["ci"])
    logger.info(f"device={device}  fractions={fractions}  seeds={seeds}  comparadores={comparators}")

    import torch  # noqa: F401

    # Cohorte QC compartido (seed-independiente)
    assert cfg.get("mode") == "real", "La curva usa PlaTiF real (mode: real)."
    from qc import build_qc_cohort

    cohort, _ = build_qc_cohort(cfg, verbose=True)
    is_pos = patient_positivity(cohort)

    # Acumuladores: (fraction, comparator) -> lista de AUROC por semilla; y OOF por (seed,frac,comp)
    auroc = defaultdict(list)
    auprc = defaultdict(list)
    knee_store = {}   # (seed, fraction, comparator) -> knee_oof df

    for seed in seeds:
        logger.info(f"########## SEMILLA {seed} ##########")
        set_seed(seed)
        folds = make_folds(cohort, int(cfg["splits"]["n_folds"]),
                           int(cfg["splits"]["internal_val_folds"]), seed)
        check_no_leakage(folds, cohort)
        if args.smoke:
            folds = folds[:1]
            logger.info("SMOKE: 1 fold")

        # Subconjuntos anidados por fold (mismos para todos comparadores y fracciones)
        fold_subsets = {}
        for fold in folds:
            fold_subsets[fold["fold"]] = nested_stratified_subsets(
                fold["train_patients"], is_pos, fractions, seed + fold["fold"])

        for fraction in fractions:
            for weights in comparators:
                view_preds = []
                for fold in folds:
                    set_seed(seed + fold["fold"])  # mismo seeding que multiseed
                    model = build_model(cfg, weights, logger=None)
                    tp = fold_subsets[fold["fold"]][fraction]
                    pred_df, _ = train_one_fold(cfg, cohort, fold, weights, model, device,
                                                logger=logger, train_patients=tp)
                    view_preds.append(pred_df)
                knee = aggregate_views_to_knee(pd.concat(view_preds, ignore_index=True))
                m = knee_metrics(knee)
                auroc[(fraction, weights)].append(m["auroc"])
                auprc[(fraction, weights)].append(m["auprc"])
                knee_store[(seed, fraction, weights)] = knee
                # persistir OOF
                od = os.path.join(base_dir, f"seed_{seed}", f"frac_{int(fraction*100)}")
                os.makedirs(od, exist_ok=True)
                knee.to_csv(os.path.join(od, f"{weights}_oof.csv"), index=False)
                logger.info(f"[seed {seed}][frac {fraction:.2f}][{weights}] "
                            f"n_train_pac={len(tp)}  OOF AUROC={m['auroc']:.4f}")

    # --- Agregación: AUROC por (fracción, comparador) --------------------------
    curve_table = {}
    for fraction in fractions:
        curve_table[fraction] = {}
        for w in comparators:
            au_m, au_s = _mean_std(auroc[(fraction, w)])
            ap_m, ap_s = _mean_std(auprc[(fraction, w)])
            curve_table[fraction][w] = {
                "auroc_mean": au_m, "auroc_std": au_s,
                "auprc_mean": ap_m, "auprc_std": ap_s,
                "auroc_per_seed": auroc[(fraction, w)],
            }

    # --- Interacción procedencia×fracción: ΔAUROC por fracción -----------------
    interaction = {}
    for fraction in fractions:
        interaction[fraction] = {}
        for a, b in itertools.combinations(comparators, 2):
            per_seed_delta, pooled = [], []
            for seed in seeds:
                ka, kb = knee_store[(seed, fraction, a)], knee_store[(seed, fraction, b)]
                common = sorted(set(ka["knee_id"]) & set(kb["knee_id"]))
                d = paired_bootstrap_delta_auroc(
                    ka[ka.knee_id.isin(common)], kb[kb.knee_id.isin(common)],
                    n_boot=int(bcfg["n_boot"]), ci=ci, seed=seed, return_deltas=True)
                per_seed_delta.append(d["delta_auroc_observed"])
                pooled.append(np.asarray(d["_deltas"]))
            dm, ds = _mean_std(per_seed_delta)
            pooled = np.concatenate(pooled) if pooled else np.array([])
            lo, hi = pooled_ci(pooled, ci)
            interaction[fraction][f"{a}_vs_{b}"] = {
                "delta_mean_over_seeds": dm, "delta_std_over_seeds": ds,
                "pooled_ci_low": lo, "pooled_ci_high": hi,
                "pooled_p_bootstrap": bootstrap_pvalue(pooled),
                "delta_per_seed": per_seed_delta,
            }

    # --- Equivalencia de datos: fracción mínima que iguala random@100% ---------
    equivalences = {}
    rand_100 = curve_table[1.0]["random"]["auroc_mean"] if "random" in comparators else None
    if rand_100 is not None:
        for w in comparators:
            if w == "random":
                continue
            hit = None
            for f in fractions:
                if curve_table[f][w]["auroc_mean"] >= rand_100:
                    hit = f
                    break
            equivalences[w] = {
                "reaches_random@100%_at_fraction": hit,
                "random@100%_auroc": rand_100,
            }

    summary = {
        "fractions": fractions, "seeds": seeds, "comparators": comparators,
        "curve": {str(f): curve_table[f] for f in fractions},
        "interaction": {str(f): interaction[f] for f in fractions},
        "data_equivalence": equivalences,
    }
    out_path = os.path.join(base_dir, "curve_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Reporte legible -------------------------------------------------------
    logger.info("\n================ CURVA DE EFICIENCIA ================")
    header = "  frac   " + "  ".join(f"{w:>12}" for w in comparators)
    logger.info(header)
    for fraction in fractions:
        row = f"  {int(fraction*100):>3}%  " + "  ".join(
            f"{curve_table[fraction][w]['auroc_mean']:.3f}±{curve_table[fraction][w]['auroc_std']:.3f}"
            for w in comparators)
        logger.info(row)
    logger.info("\nInteracción procedencia×fracción (ΔAUROC media±desv | IC95% | p_boot):")
    for fraction in fractions:
        logger.info(f"  --- fracción {int(fraction*100)}% ---")
        for pair, d in interaction[fraction].items():
            logger.info(
                f"    {pair:<26} Δ={d['delta_mean_over_seeds']:+.4f}±{d['delta_std_over_seeds']:.4f}  "
                f"IC95%=[{d['pooled_ci_low']:+.4f},{d['pooled_ci_high']:+.4f}]  p={d['pooled_p_bootstrap']:.4f}")
    if equivalences:
        logger.info("\nEquivalencia de datos (random@100% AUROC = "
                    f"{rand_100:.3f}):")
        for w, e in equivalences.items():
            h = e["reaches_random@100%_at_fraction"]
            logger.info(f"    {w}: alcanza random@100% desde la fracción "
                        f"{int(h*100) if h else '—'}%")
    logger.info(f"\nResumen -> {out_path}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
