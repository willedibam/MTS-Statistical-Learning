# pyspi/helpers/utils.py
import os, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(path: str, fig=None, dpi: int = 300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = fig or plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def meta_write(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def meta_read(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def list_model_runs(root_dir: str):
    """Yield (model_dir, model_name) for results/<profile>/<run_id>_<model>/"""
    if not os.path.isdir(root_dir):
        return
    for child in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, child)
        if not os.path.isdir(p): 
            continue
        yield p, child

def load_numpy_or_none(path: str):
    try:
        return np.load(path)
    except Exception:
        return None
