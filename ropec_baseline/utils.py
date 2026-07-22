"""Utilidades transversales: semillas, logging, dump de versiones, IO de config."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fija todas las semillas relevantes y activa determinismo razonable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn determinista sin matar el rendimiento por completo
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "ropec", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # ya configurado
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Dump de versiones (reproducibilidad — se versiona por corrida)
# --------------------------------------------------------------------------- #
def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def dump_versions(path: str) -> dict:
    info: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_hash": _git_hash(),
    }
    for mod in ["numpy", "pandas", "sklearn", "scipy", "yaml", "timm", "h5py", "openpyxl"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "not-installed"
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not-installed"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def resolve_device(pref: str = "auto") -> str:
    try:
        import torch

        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
