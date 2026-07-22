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

import glob
import os
import re

import numpy as np
import pandas as pd

# Regla de unidad rodilla CONGELADA (ver references/data_reality.md).
NORMAL_LABEL = 7

# --- Verdades de datos congeladas (verificadas en disco 20-21 jul 2026) ----------
EXPECTED_VIEWS = 421          # total de radiografías (suma de todas las imN)
EXPECTED_KNEES = 190          # unidades rodilla
EXPECTED_KNEE_POS = 128       # rodillas fractura
EXPECTED_KNEE_NEG = 62        # rodillas normales
EXPECTED_PATIENTS = 186
# Pacientes con vistas de ambos lados (fx + normal) -> 2 unidades rodilla.
MIXED_SIDE_IDS = {92, 112, 133, 147}
# Bilaterales del xlsx ("R and L"); todos consistentes -> 1 unidad c/u.
XLSX_BILATERAL_IDS = {63, 280, 290}
VIEW_RE = re.compile(r"^im\d+$")

# Caché en RAM de imágenes reales ya preprocesadas (3xSxS). Clave: (mat_path, view).
# Se llena en la 1ª época y se reusa en todos los folds/comparadores/épocas
# (preprocesamiento determinista, sin augmentation -> cacheable). Requiere
# num_workers=0 para compartir el caché en un solo proceso.
_REAL_IMAGE_CACHE: dict = {}


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
# REAL (Fase 1) — carga de PlaTiF (.mat v5)
# --------------------------------------------------------------------------- #
def _patient_id_from_path(path: str) -> int:
    m = re.search(r"Patient_ID_(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"No pude extraer patient_id de {path}")
    return int(m.group(1))


def list_platif_files(platif_root: str) -> list[str]:
    """Todos los .mat bajo Patient_Data_Part_*/ (186 esperados)."""
    files = sorted(
        glob.glob(os.path.join(platif_root, "Patient_Data_Part_*", "Patient_ID_*.mat"))
    )
    return files


def load_platif_struct(mat_path: str):
    """Carga el struct de paciente de un .mat v5. Devuelve el mat_struct raíz."""
    from scipy.io import loadmat

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        # tolerante: prioriza el que empieza por Patient_ID
        cand = [k for k in keys if k.startswith("Patient_ID")]
        if not cand:
            raise ValueError(f"{mat_path}: no encuentro el struct de paciente ({keys})")
        return mat[cand[0]]
    return mat[keys[0]]


def _view_names(struct) -> list[str]:
    """Nombres de campo imN (regex im\\d+); IGNORA Coronal_CT (es CT, fuera de protocolo)."""
    fields = list(getattr(struct, "_fieldnames", []))
    views = sorted([f for f in fields if VIEW_RE.match(f)], key=lambda s: int(s[2:]))
    return views


def read_view_label(struct, view: str) -> int:
    v = getattr(struct, view)
    if not hasattr(v, "label"):
        raise ValueError(f"vista {view} sin subcampo 'label' (campos: {getattr(v,'_fieldnames',None)})")
    return int(np.asarray(v.label).reshape(-1)[0])


def read_view_image(struct, view: str) -> np.ndarray:
    """OriginalImage 2D float de una vista."""
    v = getattr(struct, view)
    img = np.asarray(v.OriginalImage, dtype=np.float32)
    return img


def build_real_index(cfg: dict, seed: int = 0, run_assertions: bool = True) -> pd.DataFrame:
    """Escanea PlaTiF, deriva knee_id y valida contra las verdades congeladas."""
    root = cfg["paths"]["platif_root"]
    files = list_platif_files(root)
    if not files:
        raise FileNotFoundError(f"No hay .mat bajo {root}/Patient_Data_Part_*/")

    records = []
    for path in files:
        pid = _patient_id_from_path(path)
        struct = load_platif_struct(path)
        views = _view_names(struct)
        if not views:
            raise ValueError(f"{path}: sin campos imN")
        for view in views:
            label = read_view_label(struct, view)
            records.append(
                {"patient_id": pid, "view": view, "label": label, "mat_path": path}
            )

    df = build_index_from_view_records(records)
    # conservar mat_path (build_index_from_view_records no lo arrastra)
    path_by_pat = {r["patient_id"]: r["mat_path"] for r in records}
    df["mat_path"] = df["patient_id"].map(path_by_pat)

    if run_assertions:
        assert_real_index(df)
    return df


def assert_real_index(df: pd.DataFrame) -> None:
    """Aserciones congeladas. Imprime actual vs esperado y ABORTA si no cuadra."""
    knees = df.drop_duplicates("knee_id")
    n_views = len(df)
    n_pat = df["patient_id"].nunique()
    n_knees = df["knee_id"].nunique()
    n_pos = int((knees["y_vista"] == 1).sum())
    n_neg = int((knees["y_vista"] == 0).sum())
    mixed = find_mixed_patients(df)

    print("=== PlaTiF index — actual vs esperado ===")
    print(f"  vistas          : {n_views:>4}  (esp {EXPECTED_VIEWS})")
    print(f"  pacientes       : {n_pat:>4}  (esp {EXPECTED_PATIENTS})")
    print(f"  rodillas        : {n_knees:>4}  (esp {EXPECTED_KNEES})")
    print(f"  rodillas fx     : {n_pos:>4}  (esp {EXPECTED_KNEE_POS})")
    print(f"  rodillas normal : {n_neg:>4}  (esp {EXPECTED_KNEE_NEG})")
    print(f"  pacientes mixtos: {sorted(mixed)}  (esp {sorted(MIXED_SIDE_IDS)})")

    assert n_views == EXPECTED_VIEWS, f"vistas={n_views} != {EXPECTED_VIEWS}"
    assert n_pat == EXPECTED_PATIENTS, f"pacientes={n_pat} != {EXPECTED_PATIENTS}"
    assert n_knees == EXPECTED_KNEES, f"rodillas={n_knees} != {EXPECTED_KNEES}"
    assert n_pos == EXPECTED_KNEE_POS, f"rodillas fx={n_pos} != {EXPECTED_KNEE_POS}"
    assert n_neg == EXPECTED_KNEE_NEG, f"rodillas normal={n_neg} != {EXPECTED_KNEE_NEG}"
    # Guarda de mixtos: conjunto EXACTO o aborta (datos no esperados).
    assert mixed == MIXED_SIDE_IDS, (
        f"ABORTO: IDs mixtos {sorted(mixed)} != esperados {sorted(MIXED_SIDE_IDS)}. "
        "Los datos no son los esperados; revisa la fuente antes de continuar."
    )
    print("  ✅ todas las aserciones de datos pasan.")


# --------------------------------------------------------------------------- #
# Cruce con metadata clínica (xlsx) — verificación, no filtro
# --------------------------------------------------------------------------- #
def _find_col(cols, *needles) -> str | None:
    for c in cols:
        cl = str(c).lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


def cross_check_xlsx(df: pd.DataFrame, xlsx_path: str) -> pd.DataFrame:
    """Join por Patient ID; verifica Normal(xlsx) <-> todas las vistas label==7.

    Reporta discrepancias como supuesto marcado (no excluye nada).
    """
    meta = pd.read_excel(xlsx_path, engine="openpyxl")
    cols = list(meta.columns)
    id_col = _find_col(cols, "patient", "id") or cols[0]
    schatzker_col = _find_col(cols, "schatzker") or _find_col(cols, "fracture", "type")
    side_col = _find_col(cols, "right") or _find_col(cols, "left")

    meta = meta.rename(columns={id_col: "patient_id"})
    meta["patient_id"] = pd.to_numeric(meta["patient_id"], errors="coerce").astype("Int64")

    # Etiqueta a nivel paciente derivada de nuestras vistas
    pat = df.groupby("patient_id")["y_vista"].max().reset_index()
    pat["derived_all_normal"] = pat["y_vista"] == 0  # todas las vistas normal
    merged = pat.merge(meta, on="patient_id", how="left")

    discrepancies = []
    if schatzker_col:
        def is_normal_xlsx(v):
            return str(v).strip().lower().startswith("normal")

        merged["xlsx_normal"] = merged[schatzker_col].map(is_normal_xlsx)
        # Discrepancia: xlsx dice Normal pero tenemos alguna vista fx, o viceversa
        mism = merged[merged["xlsx_normal"] != merged["derived_all_normal"]]
        for _, r in mism.iterrows():
            discrepancies.append(
                {
                    "patient_id": int(r["patient_id"]),
                    "xlsx_schatzker": r[schatzker_col],
                    "derived_all_normal": bool(r["derived_all_normal"]),
                }
            )

    print("=== Cruce con xlsx ===")
    print(f"  columnas usadas: id='{id_col}' schatzker='{schatzker_col}' side='{side_col}'")
    print(f"  filas xlsx: {len(meta)}")
    if discrepancies:
        print(f"  ⚠️ {len(discrepancies)} discrepancias Normal<->label7 (SUPUESTO MARCADO):")
        for d in discrepancies:
            print(f"     P{d['patient_id']}: xlsx='{d['xlsx_schatzker']}' "
                  f"all_normal_derivado={d['derived_all_normal']}")
    else:
        print("  ✅ Normal(xlsx) concuerda con label==7 en todos los pacientes.")
    return merged


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

    # Normalización compartida por TODOS los comparadores (invariante: misma preproc).
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, index_df: pd.DataFrame, cfg: dict, mode: str = "toy"):
        import torch  # import perezoso para no exigir torch al construir el índice

        self._torch = torch
        self.df = index_df.reset_index(drop=True)
        self.mode = mode
        self.img_size = int(cfg["model"]["img_size"])
        self._struct_cache: dict[str, object] = {}  # cache pequeño por path (mat_struct)
        self._cache_order: list[str] = []

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

    def _get_struct(self, mat_path: str):
        if mat_path not in self._struct_cache:
            self._struct_cache[mat_path] = load_platif_struct(mat_path)
            self._cache_order.append(mat_path)
            if len(self._cache_order) > 8:  # cache LRU muy simple
                old = self._cache_order.pop(0)
                self._struct_cache.pop(old, None)
        return self._struct_cache[mat_path]

    def _real_image(self, mat_path: str, view: str):
        """OriginalImage -> tensor 3xSxS normalizado (ImageNet stats, compartido)."""
        torch = self._torch
        import torch.nn.functional as F

        cache_key = (mat_path, view, self.img_size)
        cached = _REAL_IMAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached.clone()

        struct = self._get_struct(mat_path)
        img = read_view_image(struct, view)  # HxW float32
        # Normalización robusta por imagen a [0,1] (percentiles 1-99)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        if hi <= lo:
            hi = lo + 1e-6
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        t = torch.from_numpy(img)[None, None]  # 1,1,H,W
        t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear",
                          align_corners=False)[0, 0]
        t = t.repeat(3, 1, 1)  # 3,S,S
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        out = (t - mean) / std
        _REAL_IMAGE_CACHE[cache_key] = out
        return out.clone()

    def __getitem__(self, i: int):
        torch = self._torch
        row = self.df.iloc[i]
        y = int(row["y_vista"])
        if self.mode == "toy":
            img = self._toy_image(row["patient_id"], row["view"], y)
        else:
            img = self._real_image(row["mat_path"], row["view"])
        return img, torch.tensor(float(y)), int(i)


def subset_index(index_df: pd.DataFrame, patient_ids) -> pd.DataFrame:
    ids = set(int(p) for p in patient_ids)
    return index_df[index_df["patient_id"].isin(ids)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI: construir y validar el índice real de PlaTiF de forma aislada (Prompt 1.1)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    from utils import load_config

    ap = argparse.ArgumentParser(description="Construye platif_index.csv (índice real).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="ruta de salida del CSV")
    ap.add_argument("--no-xlsx", action="store_true", help="omitir cruce con xlsx")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = build_real_index(cfg, seed=int(cfg["seed"]), run_assertions=True)

    if not args.no_xlsx:
        try:
            cross_check_xlsx(df, cfg["paths"]["platif_xlsx"])
        except Exception as e:  # el cruce es verificación, no bloquea el índice
            print(f"  ⚠️ cruce xlsx falló ({type(e).__name__}): {e}")

    out = args.out or os.path.join(cfg["paths"]["output_root"], "platif_index.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["patient_id", "view", "knee_id", "side_class", "label", "y_vista", "mat_path"]
    df[cols].to_csv(out, index=False)
    print(f"\nplatif_index.csv escrito en: {out}  ({len(df)} filas)")
