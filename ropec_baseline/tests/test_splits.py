"""Test de CERO FUGA para los splits por paciente (Prompt 1.3).

Usa un índice sintético (sin datos reales) para correr en cualquier entorno.
Verifica: ningún patient_id en dos folds de test, cobertura completa, ninguna
rodilla huérfana (knee_id compartido entre pacientes), y que ambas rodillas de un
paciente mixto caen en el mismo fold. Reporta prevalencia por fold.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import build_index_from_view_records, find_mixed_patients  # noqa: E402
from splits import check_no_leakage, make_folds  # noqa: E402

NORMAL = 7


def _synthetic_index(n_patients=60, n_mixed=4, seed=0):
    rng = np.random.default_rng(seed)
    mixed_ids = set(range(1, n_mixed + 1))
    records = []
    for i in range(n_patients):
        pid = i + 1
        n_views = int(rng.integers(1, 6))
        if pid in mixed_ids:
            labels = [int(rng.integers(1, 7)), NORMAL]
            for _ in range(max(0, n_views - 2)):
                labels.append(int(rng.integers(1, 7)) if rng.random() < 0.5 else NORMAL)
        elif rng.random() < 0.67:
            labels = [int(rng.integers(1, 7)) for _ in range(n_views)]
        else:
            labels = [NORMAL for _ in range(n_views)]
        for v, lab in enumerate(labels):
            records.append({"patient_id": pid, "view": f"im{v}", "label": lab})
    return build_index_from_view_records(records)


def test_no_patient_in_two_test_folds():
    idx = _synthetic_index()
    folds = make_folds(idx, n_folds=5, internal_val_folds=5, seed=123)
    seen = {}
    for f in folds:
        for p in f["test_patients"]:
            seen[p] = seen.get(p, 0) + 1
    assert all(c == 1 for c in seen.values()), "algún paciente en >1 fold de test"
    assert set(seen) == set(int(p) for p in idx["patient_id"].unique())


def test_check_no_leakage_passes():
    idx = _synthetic_index()
    folds = make_folds(idx, n_folds=5, internal_val_folds=5, seed=123)
    check_no_leakage(folds, idx)  # no debe levantar


def test_mixed_patient_knees_same_fold():
    idx = _synthetic_index()
    mixed = find_mixed_patients(idx)
    assert len(mixed) >= 1
    folds = make_folds(idx, n_folds=5, internal_val_folds=5, seed=123)
    # localizar el fold de test de cada paciente mixto y verificar sus 2 knee_id
    for pid in mixed:
        knees = set(idx[idx.patient_id == pid]["knee_id"])
        assert len(knees) == 2, f"P{pid} debería aportar 2 rodillas"
        folds_with = [f["fold"] for f in folds if pid in f["test_patients"]]
        assert len(folds_with) == 1, f"P{pid} no está en exactamente un fold de test"


def test_no_orphan_knee():
    idx = _synthetic_index()
    owners = idx.groupby("knee_id")["patient_id"].nunique()
    assert (owners == 1).all(), "hay knee_id compartido entre pacientes"


def test_train_val_test_disjoint():
    idx = _synthetic_index()
    folds = make_folds(idx, n_folds=5, internal_val_folds=5, seed=123)
    for f in folds:
        tr, va, te = set(f["train_patients"]), set(f["val_patients"]), set(f["test_patients"])
        assert not (tr & te) and not (va & te) and not (tr & va)


if __name__ == "__main__":
    # permite correr sin pytest: python tests/test_splits.py
    idx = _synthetic_index()
    folds = make_folds(idx, n_folds=5, internal_val_folds=5, seed=123)
    check_no_leakage(folds, idx)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("\n=== Prevalencia por fold (test, sintético) ===")
    for f in folds:
        t = f["prevalence"]["test"]
        print(f"  fold {f['fold']}: {t['n_patients']} pac / {t['n_knees']} rodillas "
              f"/ prev={t['knee_pos_rate']:.3f}")
    print("\nTodos los tests de cero fuga pasan.")
