"""Modelo ResNet-50 (timm) + carga de pesos con ASERCIÓN DE CONTEO de tensores.

Landmine real: con strict=False un remap mal hecho descarta todo en silencio ->
entrenarías 'random' creyendo que es RadImageNet. Por eso TODA carga cuenta cuántos
tensores casaron (nombre + shape) y ABORTA si no alcanza el mínimo/exacto esperado.
"""
from __future__ import annotations

from collections import OrderedDict

import timm
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Remap RadImageNet (torchvision resnet50 como Sequential -> nombres timm)
# --------------------------------------------------------------------------- #
_RADIMAGENET_REMAP = {
    "backbone.0.": "conv1.",
    "backbone.1.": "bn1.",
    "backbone.4.": "layer1.",
    "backbone.5.": "layer2.",
    "backbone.6.": "layer3.",
    "backbone.7.": "layer4.",
}


def remap_radimagenet_keys(state_dict: "OrderedDict") -> "OrderedDict":
    out = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for pre, rep in _RADIMAGENET_REMAP.items():
            if k.startswith(pre):
                nk = rep + k[len(pre):]
                break
        out[nk] = v
    return out


def _torch_load_state_dict(path: str, logger=None):
    """torch.load robusto ante el cambio de weights_only en torch>=2.6.

    El .pt de RadImageNet es un OrderedDict de tensores (fuente confiable del
    usuario). Intenta weights_only=True; si el pickle lo requiere, cae a False.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch antiguo sin el kwarg weights_only
        return torch.load(path, map_location="cpu")
    except Exception as e:  # pickle no cargable en modo seguro -> fallback confiable
        if logger:
            logger.info(f"[weights] weights_only=True falló ({type(e).__name__}); "
                        "reintentando weights_only=False (archivo confiable).")
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Carga con aserción de conteo
# --------------------------------------------------------------------------- #
def load_matched_weights(
    model: nn.Module,
    source_sd: "OrderedDict",
    name: str,
    expect_exact: int | None = None,
    expect_min: int | None = None,
    logger=None,
) -> int:
    """Copia in-place los tensores de source_sd que casan (nombre+shape) en model.

    Devuelve el nº de tensores cargados. Aborta (AssertionError) si no cumple el
    conteo esperado. El clasificador 'fc' del modelo queda sin inicializar si la
    fuente es solo extractor — eso es esperado.
    """
    model_sd = model.state_dict()
    loaded, skipped_shape, skipped_missing = 0, [], []
    new_sd = OrderedDict()
    for k, v in model_sd.items():
        if k in source_sd:
            if source_sd[k].shape == v.shape:
                new_sd[k] = source_sd[k].clone()
                loaded += 1
            else:
                new_sd[k] = v
                skipped_shape.append(k)
        else:
            new_sd[k] = v
            skipped_missing.append(k)

    # tensores en la fuente que no existen en el modelo (p.ej. fc de otra tarea)
    unused_source = [k for k in source_sd if k not in model_sd]

    msg = (
        f"[weights:{name}] cargados={loaded}  "
        f"omitidos_por_shape={len(skipped_shape)}  "
        f"faltantes_en_fuente={len(skipped_missing)}  "
        f"no_usados_de_fuente={len(unused_source)}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    if expect_exact is not None:
        assert loaded == expect_exact, (
            f"ABORTO [{name}]: se cargaron {loaded} tensores, se exigían EXACTAMENTE "
            f"{expect_exact}. Remap/pesos incorrectos (no entrenar random creyendo {name})."
        )
    if expect_min is not None:
        assert loaded >= expect_min, (
            f"ABORTO [{name}]: se cargaron {loaded} tensores, se exigía un mínimo de "
            f"{expect_min}. Carga de pesos sospechosa."
        )

    model.load_state_dict(new_sd, strict=True)
    return loaded


# --------------------------------------------------------------------------- #
# Constructor de modelo por comparador
# --------------------------------------------------------------------------- #
def build_model(cfg: dict, weights: str, logger=None) -> nn.Module:
    """weights in {'random','imagenet','radimagenet'}. Backbone único ResNet-50."""
    mcfg = cfg["model"]
    backbone = mcfg["backbone"]
    num_classes = int(mcfg["num_classes"])
    drop_rate = float(mcfg.get("drop_rate", 0.0))  # dropout antes del head (no añade tensores)
    assert backbone == "resnet50", "Backbone ÚNICO congelado: resnet50."

    if weights == "random":
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes,
                                  drop_rate=drop_rate)
        if logger:
            logger.info("[weights:random] backbone inicializado aleatoriamente (sin carga).")
        return model

    if weights == "imagenet":
        # Modelo destino sin preentrenar; copiamos el state_dict ImageNet de timm con aserción.
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes,
                                  drop_rate=drop_rate)
        src = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="")
        src_sd = src.state_dict()
        # Esperamos casar todo el extractor (todo menos el head fc del destino).
        expect_min = len([k for k in model.state_dict() if not k.startswith("fc.")])
        load_matched_weights(
            model, src_sd, name="imagenet", expect_min=expect_min, logger=logger
        )
        return model

    if weights == "radimagenet":
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes,
                                  drop_rate=drop_rate)
        path = cfg["paths"]["radimagenet_resnet50"]
        raw = _torch_load_state_dict(path, logger)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        remapped = remap_radimagenet_keys(raw)
        expect = int(mcfg["expected_tensors"]["radimagenet"])
        load_matched_weights(
            model, remapped, name="radimagenet", expect_exact=expect, logger=logger
        )
        return model

    raise ValueError(f"weights desconocido: {weights}")


# --------------------------------------------------------------------------- #
# CLI: probar la carga de pesos por comparador y reportar conteo (Prompt 2.1)
# --------------------------------------------------------------------------- #
def _main():
    import argparse

    from utils import get_logger, load_config

    ap = argparse.ArgumentParser(description="Prueba de carga de pesos por comparador.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", default="radimagenet",
                    help="random | imagenet | radimagenet | all")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("ropec-weights")
    targets = ["random", "imagenet", "radimagenet"] if args.weights == "all" else [args.weights]

    for w in targets:
        logger.info(f"----- construyendo modelo: weights={w} -----")
        model = build_model(cfg, w, logger=logger)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[{w}] modelo OK  ({n_params/1e6:.1f}M parámetros)")


if __name__ == "__main__":
    _main()
