import os, shutil, random
from pathlib import Path

# Parámetros
DATA_DIR = Path("data/raw")   # ajusta si tu ruta es distinta
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
PCT = 0.15                   # porcentaje de imágenes a mover
SEED = 42
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(SEED)

# Crear carpetas val/<clase> si no existen
classes = [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()]
for cls in classes:
    (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)

# Mover imágenes
for cls in classes:
    src = TRAIN_DIR / cls
    dst = VAL_DIR / cls

    files = [f for f in src.iterdir() if f.suffix.lower() in ALLOWED_EXTS]
    n_take = max(1, int(len(files) * PCT))

    chosen = random.sample(files, n_take)
    for f in chosen:
        shutil.move(str(f), dst / f.name)

    print(f"Clase '{cls}': movidas {n_take}/{len(files)+n_take} imágenes a val/")
