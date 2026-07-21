"""Datos: índice + Dataset PyTorch.

Fase 0 (mode: toy): dataset juguete que REPRODUCE la estructura real de PlaTiF
    - nº de vistas variable por paciente (1..8),
    - >=1 paciente mixto (rodilla fx + rodilla normal),
    - derivación programática de knee_id,
para ejercitar todo el pipeline end-to-end sin tocar datos reales.

Fase 1 (mode: real) conectará PlaTiF (.mat v5) reescribiendo build_real_index().
La regla de unidad rodilla y el esquema del índice se comparten entre ambos modos.

Esquema del índice (una fila por VISTA):
    patient_id : int
    view       : str   -> "im0", "im1", ...
    knee_id    : str   -> f"{patient_id}:{'fx' if label<=6 else 'normal'}"
    side_class : str   -> "fx" | "normal"
    label      : int   -> label crudo por vista (1..6 = Schatzker, 7 = Normal)
    y_vista    : int   -> int(label != 7)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Regla de unidad rodilla CONGELADA (ver references/data_reality.md).
NORMAL_LABEL = 7


def knee_id_of(patient_id: int, label: int) -> str:
    """knee_id = (patient_id, 'fx' if label<=6 else 'normal')."""
    side = "fx" if int(label) <= 6 else "normal"
    return f"{int(patient_id)}:{side}"


def y_from_label(label: int) -> int:
    return int(int(label) != NORMAL_LABEL)


def build_index_from_view_records(records: list[dict]) -> pd.DataFrame:
    """records: lista de dicts {patient_id, view, label}. Deriva knee_id/side/y."""
    rows = []
    for r in records:
        pid, view, label = int(r["patient_id"]), r["view"], int(r["label"])
        rows.append(
            {
                "patient_id": pid,
                "view": view,
                "knee_id": knee_id_of(pid, label),
                "side_class": "fx" if label <= 6 else "normal",
                "label": label,
                "y_vista": y_from_label(label),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["patient_id", "view"]).reset_index(drop=True)


def find_mixed_patients(df: pd.DataFrame) -> set[int]:
    """Pacientes con vistas de ambos lados (fx + normal) => 2 unidades rodilla."""
    per_pat = df.groupby("patient_id")["side_class"].nunique()
    return set(per_pat[per_pat > 1].index.astype(int))


# --------------------------------------------------------------------------- #
# TOY (Fase 0)
# --------------------------------------------------------------------------- #
def build_toy_index(cfg: dict, seed: int) -> pd.DataFrame:
    """Índice juguete con nº de vistas variable y pacientes mixtos garantizados."""
    rng = np.random.default_rng(seed)
    tcfg = cfg["toy"]
    n = int(tcfg["n_patients"])
    views_choices = list(tcfg["views_choices"])
    pos_rate = float(tcfg["positive_rate"])
    n_mixed = int(tcfg["n_mixed_patients"])

    # Etiqueta de paciente (positivo => tendrá vistas fx)
    pat_pos = rng.random(n) < pos_rate
    # Forzar que los primeros n_mixed pacientes sean mixtos (positivos con ambos lados)
    mixed_ids = set(range(1, n_mixed + 1))
    for pid in mixed_ids:
        pat_pos[pid - 1] = True

    records = []
    for i in range(n):
        pid = i + 1
        n_views = int(rng.choice(views_choices))
        if pid in mixed_ids:
            # Al menos 1 vista fx y 1 normal
            n_views = max(n_views, 2)
            labels = [int(rng.integers(1, 7))]  # una fx (1..6)
            labels += [NORMAL_LABEL]  # una normal
            for _ in range(n_views - 2):
                labels.append(int(rng.integers(1, 7)) if rng.random() < 0.5 else NORMAL_LABEL)
        elif pat_pos[i]:
            labels = [int(rng.integers(1, 7)) for _ in range(n_views)]  # todas fx
        else:
            labels = [NORMAL_LABEL for _ in range(n_views)]  # todas normal

        for v, lab in enumerate(labels):
            records.append({"patient_id": pid, "view": f"im{v}", "label": lab})

    return build_index_from_view_records(records)


# --------------------------------------------------------------------------- #
# REAL (Fase 1 — stub; se implementa en Prompt 1.1)
# --------------------------------------------------------------------------- #
def build_real_index(cfg: dict, seed: int) -> pd.DataFrame:  # pragma: no cover
    raise NotImplementedError(
        "PlaTiF real se conecta en Fase 1 (Prompt 1.1). En Fase 0 usa mode: toy."
    )


def build_index(cfg: dict, seed: int) -> pd.DataFrame:
    if cfg.get("mode", "toy") == "toy":
        return build_toy_index(cfg, seed)
    return build_real_index(cfg, seed)


# --------------------------------------------------------------------------- #
# Dataset PyTorch
# --------------------------------------------------------------------------- #
class ViewDataset:
    """Dataset a nivel VISTA. Devuelve (imagen 3xHxW, y, row_index).

    En toy la imagen es ruido determinista por (patient_id, view) para
    reproducibilidad; su señal se correlaciona ligeramente con y_vista para que
    el pipeline pueda aprender algo distinto de azar (smoke útil, no realista).
    En real (Fase 1) se sustituye la carga por lectura de OriginalImage del .mat.
    """

    def __init__(self, index_df: pd.DataFrame, cfg: dict, mode: str = "toy"):
        import torch  # import perezoso para no exigir torch al construir el índice

        self._torch = torch
        self.df = index_df.reset_index(drop=True)
        self.mode = mode
        self.img_size = int(cfg["model"]["img_size"])

    def __len__(self) -> int:
        return len(self.df)

    def _toy_image(self, patient_id: int, view: str, y: int):
        torch = self._torch
        # Semilla determinista por vista
        seed = (hash((int(patient_id), view)) ^ 0x9E3779B9) & 0x7FFFFFFF
        g = torch.Generator().manual_seed(seed)
        img = torch.randn(3, self.img_size, self.img_size, generator=g) * 0.5
        # Señal débil dependiente de la clase (offset en el canal 0)
        img[0] += 0.5 if y == 1 else -0.5
        return img

    def __getitem__(self, i: int):
        torch = self._torch
        row = self.df.iloc[i]
        y = int(row["y_vista"])
        if self.mode == "toy":
            img = self._toy_image(row["patient_id"], row["view"], y)
        else:  # pragma: no cover  (Fase 1)
            raise NotImplementedError("Carga real de .mat se implementa en Fase 1.")
        return img, torch.tensor(float(y)), int(i)


def subset_index(index_df: pd.DataFrame, patient_ids) -> pd.DataFrame:
    ids = set(int(p) for p in patient_ids)
    return index_df[index_df["patient_id"].isin(ids)].reset_index(drop=True)
