from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# --------- Rutas robustas ---------
APP_DIR = Path(__file__).resolve().parent.parent      # .../melanoma-app
MODELS_DIR = APP_DIR / "models"

# ⬇️ Pon aquí el nombre EXACTO del archivo .pt
MODEL_FILENAME = "deepcnn6res_clean_best.pt"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# --------- Arquitectura ---------
class ResidBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.conv(x) + x)

class DeepCNN6Res(nn.Module):
    """
    6 bloques conv, kernels grandes al inicio, residuales en los 4 últimos.
    224 -> 56 -> 28 -> 14 -> 7 -> 7 -> 7 -> GAP
    """
    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.num_outputs = num_outputs
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ); self.res2 = ResidBlock(64)

        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ); self.res3 = ResidBlock(128)

        self.b4 = nn.Sequential(
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ); self.res4 = ResidBlock(192)

        self.b5 = nn.Sequential(
            nn.Conv2d(192, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        ); self.res5 = ResidBlock(256)

        self.b6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        ); self.res6 = ResidBlock(256)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.45), nn.Linear(256, self.num_outputs))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x); x = self.res2(x)
        x = self.b3(x); x = self.res3(x)
        x = self.b4(x); x = self.res4(x)
        x = self.b5(x); x = self.res5(x)
        x = self.b6(x); x = self.res6(x)
        x = self.pool(x)
        return self.head(x)  # [B, num_outputs] (logits)

# --------- Carga robusta ---------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _infer_num_outputs_from_state(state: dict) -> int:
    """
    Detecta out_features de la capa final buscando el peso Linear de 'head'.
    Soporta variaciones como 'head.2.weight' o con prefijo 'module.'.
    """
    candidates = [k for k in state.keys() if k.endswith("weight") and "head" in k]
    for k in candidates:
        w = state[k]
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])   # out_features
    return 1  # fallback

def load_model(model_path: Path = MODEL_PATH) -> nn.Module:
    """
    - Detecta si el checkpoint es de 1 o 2 salidas.
    - Si hay desajuste, elimina pesos de la cabeza y carga el resto (strict=False).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}\n(CWD: {Path.cwd()})")

    device = get_device()
    state = torch.load(model_path, map_location=device)

    # Algunos checkpoints se guardan envueltos (p.ej. {'model': sd}). Intento destriparlo.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    num_out = _infer_num_outputs_from_state(state)
    model = DeepCNN6Res(num_outputs=num_out).to(device)

    # Intento 1: carga estricta (si coincide la cabeza no habrá error)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Intento 2: si el fallo es por la cabeza, la quitamos y cargamos el resto
        keys_to_drop = [k for k in state.keys() if "head" in k]
        for k in keys_to_drop:
            state.pop(k, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[WARN] Carga non-strict. missing={missing}, unexpected={unexpected}")

    model.eval()
    print(f"✅ Modelo PyTorch cargado ({num_out} salida(s)) en {device}: {model_path.name}")
    return model

# --------- Preprocesado e inferencia ---------
_TFM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),  # [0..1]
])

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    return _TFM(pil_img.convert("RGB")).unsqueeze(0)  # (1,3,H,W)

@torch.inference_mode()
def predict_proba_malignant(model: nn.Module, pil_img: Image.Image, tta: bool = True) -> float:
    """
    Devuelve prob(malignant) en [0,1].
    - 1 salida -> sigmoid(logit)
    - 2 salidas -> softmax[:,1]
    """
    device = next(model.parameters()).device
    x = preprocess_image(pil_img).to(device)

    logits = model(x)  # (1, C)
    if getattr(model, "num_outputs", 1) == 1:
        p = torch.sigmoid(logits.squeeze(1)).item()
    else:
        p = torch.softmax(logits, dim=1)[0, 1].item()

    if tta:
        x_flip = torch.flip(x, dims=[-1])
        logits2 = model(x_flip)
        if getattr(model, "num_outputs", 1) == 1:
            p2 = torch.sigmoid(logits2.squeeze(1)).item()
        else:
            p2 = torch.softmax(logits2, dim=1)[0, 1].item()
        p = (p + p2) / 2.0

    return float(p)

def classify_image(model: nn.Module, pil_img: Image.Image, threshold: float = 0.5, tta: bool = True) -> Tuple[str, float]:
    p = predict_proba_malignant(model, pil_img, tta=tta)
    label = "Malignant" if p >= threshold else "Benign"
    return label, p
