"""Barrido multi-semilla: N corridas CV independientes -> robustez de OPTIMIZACIÓN
(media±desv del AUROC entre semillas) combinada con robustez de MUESTREO (bootstrap
pareado por pacientes, agrupando las distribuciones de todas las semillas).

Reporta:
  - AUROC/AUPRC por comparador: media ± desv. sobre las semillas.
  - ΔAUROC por par: media ± desv. del efecto entre semillas + IC95% AGRUPADO
    (concatenando los deltas bootstrap de todas las semillas -> muestreo+semilla)
    + p-valor bootstrap agrupado.

Uso:
    python run_multiseed.py --config config.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from eval import bootstrap_pvalue, pooled_ci
from run_experiment import run_once
from utils import get_logger, load_config, resolve_device


def _mean_std(xs):
    a = np.asarray([x for x in xs if x == x], dtype=float)  # descarta NaN
    return float(a.mean()) if len(a) else float("nan"), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Barrido multi-semilla (robustez).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 fold por semilla (rápido)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = list(cfg["multiseed"]["seeds"])
    base_dir = os.path.join(cfg["paths"]["output_root"], cfg["run_name"])
    os.makedirs(base_dir, exist_ok=True)
    logger = get_logger("ropec-multiseed", logfile=os.path.join(base_dir, "multiseed.log"))
    device = resolve_device(cfg.get("device", "auto"))
    ci = float(cfg["eval"]["bootstrap"]["ci"])
    logger.info(f"device={device}  run={cfg['run_name']}  seeds={seeds}")

    import torch  # noqa: F401

    # Cohorte post-QC: seed-independiente -> se construye UNA vez y se reutiliza.
    shared_index, shared_excl = None, None
    if cfg.get("mode", "toy") == "real":
        from qc import build_qc_cohort

        shared_index, shared_excl = build_qc_cohort(cfg, verbose=True)
        logger.info(f"Cohorte QC compartido: {shared_index['knee_id'].nunique()} rodillas "
                    f"(se reutiliza en las {len(seeds)} semillas)")

    # Acumuladores
    auroc_by_comp = defaultdict(list)
    auprc_by_comp = defaultdict(list)
    delta_obs_by_pair = defaultdict(list)     # observados por semilla
    delta_pool_by_pair = defaultdict(list)     # arrays bootstrap a agrupar
    comparators = None

    for s in seeds:
        od = os.path.join(base_dir, f"seed_{s}")
        logger.info(f"########## SEMILLA {s} ##########")
        res = run_once(cfg, s, od, logger, device, smoke=args.smoke, collect_boot=True,
                       index=shared_index, excl_df=shared_excl)
        comparators = res["comparators"]
        for c, m in res["metrics"].items():
            auroc_by_comp[c].append(m["auroc"])
            auprc_by_comp[c].append(m["auprc"])
        for pair, d in res["deltas"].items():
            delta_obs_by_pair[pair].append(d["delta_auroc_observed"])
            if "_deltas" in d:
                delta_pool_by_pair[pair].append(np.asarray(d["_deltas"]))

    # --- Agregación por comparador (robustez de optimización) ------------------
    per_comparator = {}
    for c in comparators:
        au_m, au_s = _mean_std(auroc_by_comp[c])
        ap_m, ap_s = _mean_std(auprc_by_comp[c])
        per_comparator[c] = {
            "auroc_mean": au_m, "auroc_std": au_s, "auroc_per_seed": auroc_by_comp[c],
            "auprc_mean": ap_m, "auprc_std": ap_s, "auprc_per_seed": auprc_by_comp[c],
        }

    # --- Agregación por par (muestreo + semilla) -------------------------------
    per_pair = {}
    for pair in delta_obs_by_pair:
        obs_m, obs_s = _mean_std(delta_obs_by_pair[pair])
        pooled = np.concatenate(delta_pool_by_pair[pair]) if delta_pool_by_pair[pair] else np.array([])
        lo, hi = pooled_ci(pooled, ci)
        per_pair[pair] = {
            "delta_mean_over_seeds": obs_m,
            "delta_std_over_seeds": obs_s,
            "delta_observed_per_seed": delta_obs_by_pair[pair],
            "pooled_ci": ci,
            "pooled_ci_low": lo,
            "pooled_ci_high": hi,
            "pooled_p_bootstrap": bootstrap_pvalue(pooled),
            "n_pooled_boot": int(len(pooled)),
        }

    summary = {
        "run_name": cfg["run_name"],
        "seeds": seeds,
        "n_seeds": len(seeds),
        "comparators": comparators,
        "per_comparator": per_comparator,
        "per_pair": per_pair,
    }
    out_path = os.path.join(base_dir, "multiseed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Reporte legible -------------------------------------------------------
    logger.info("\n================ RESUMEN MULTI-SEMILLA ================")
    logger.info(f"Semillas: {seeds}")
    logger.info("\nAUROC OOF por comparador (media ± desv. entre semillas):")
    for c in comparators:
        pc = per_comparator[c]
        logger.info(f"  {c:<12} AUROC={pc['auroc_mean']:.4f} ± {pc['auroc_std']:.4f}   "
                    f"AUPRC={pc['auprc_mean']:.4f} ± {pc['auprc_std']:.4f}")
    logger.info("\nΔAUROC por par (media±desv entre semillas | IC95% agrupado | p_boot):")
    for pair, pp in per_pair.items():
        logger.info(
            f"  {pair:<26} Δ={pp['delta_mean_over_seeds']:+.4f} ± {pp['delta_std_over_seeds']:.4f}   "
            f"IC95%=[{pp['pooled_ci_low']:+.4f}, {pp['pooled_ci_high']:+.4f}]   "
            f"p_boot={pp['pooled_p_bootstrap']:.4f}"
        )
    logger.info(f"\nResumen -> {out_path}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
