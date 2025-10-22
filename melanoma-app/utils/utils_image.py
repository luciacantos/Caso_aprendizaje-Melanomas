# utils/utils_image.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans

# =========================
# Utilidades básicas
# =========================
def ensure_rgb(img) -> Image.Image:
    """Garantiza objeto PIL RGB a partir de PIL o np.ndarray."""
    return img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB")


# =========================
# Segmentación robusta (del cuaderno)
# =========================
def largest_component(mask_bin: np.ndarray) -> np.ndarray:
    """
    Conserva sólo el componente conexo de mayor área.
    Espera máscara uint8 {0,255} y devuelve lo mismo.
    """
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_bin
    main = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask_bin)
    cv2.drawContours(clean, [main], -1, 255, thickness=-1)
    return clean


def robust_lesion_mask(rgb_pil: Image.Image) -> np.ndarray:
    """
    Segmentación robusta basada en:
      - HSV: CLAHE en V + Otsu sobre V (lesión más oscura)
      - LAB: Otsu en canal 'a' (refuerzo de pigmentación)
      - Morfología (close/open) + componente mayor
    Devuelve máscara binaria {0,1} en np.uint8.
    """
    rgb = np.array(rgb_pil)
    # HSV + CLAHE en V
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v   = hsv[..., 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    v_blur = cv2.GaussianBlur(v_eq, (5, 5), 0)
    thr_v, _ = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_vdark = (v_blur < thr_v).astype(np.uint8) * 255

    # LAB canal 'a'
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a = lab[..., 1]
    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    thr_a, _ = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_a = (a_blur > thr_a).astype(np.uint8) * 255

    # Combina y limpia
    mask = cv2.bitwise_and(mask_vdark, mask_a)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = largest_component(mask)

    return (mask > 0).astype(np.uint8)


# =========================
# Métricas geométricas (área y simetrías)
# =========================
def geo_features_from_mask(mask: np.ndarray) -> dict:
    """
    Calcula métricas geométricas realistas de la lesión:
      - Área (%)
      - Simetrías vertical y horizontal centradas
    """
    m = (mask > 0).astype(np.uint8)
    area_px = int(m.sum())
    total_px = int(m.size)
    area_pct = 100.0 * area_px / max(total_px, 1)

    if area_px == 0:
        return {"area_px": 0, "area_pct": 0.0,
                "sym_vert_pct": 0.0, "sym_horz_pct": 0.0, "sym_global_pct": 0.0}

    # recorte del bounding box de la lesión
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    lesion = m[y0:y1+1, x0:x1+1]

    # centramos en el centroide para evitar desplazamiento
    M = cv2.moments(lesion)
    cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else lesion.shape[1]//2
    cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else lesion.shape[0]//2
    pad_x = abs(cx - lesion.shape[1]//2)
    pad_y = abs(cy - lesion.shape[0]//2)
    lesion = np.pad(lesion, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")

    # simetría vertical
    h, w = lesion.shape
    left = lesion[:, :w//2]
    right = np.fliplr(lesion[:, w-w//2:])
    inter_v = np.logical_and(left, right).sum()
    union_v = np.logical_or(left, right).sum()
    sym_vert = 100.0 * inter_v / max(union_v, 1)

    # simetría horizontal
    top = lesion[:h//2, :]
    bottom = np.flipud(lesion[h-h//2:, :])
    inter_h = np.logical_and(top, bottom).sum()
    union_h = np.logical_or(top, bottom).sum()
    sym_horz = 100.0 * inter_h / max(union_h, 1)

    sym_global = (sym_vert + sym_horz) / 2.0

    return {
        "area_px": area_px,
        "area_pct": area_pct,
        "sym_vert_pct": sym_vert,
        "sym_horz_pct": sym_horz,
        "sym_global_pct": sym_global,
    }


# =========================
# Paleta / segmentación cromática dentro de la lesión
# =========================
def lesion_palette(rgb_arr: np.ndarray, mask: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[List[int]], List[float]]:
    """
    KMeans sobre los píxeles dentro de la lesión (mask==1).
    Devuelve: imagen segmentada, lista de centros RGB, y porcentajes por cluster.
    """
    m = mask.astype(bool)
    pix = rgb_arr[m]
    if len(pix) < k:
        k = max(1, min(3, len(pix))) if len(pix) > 0 else 1

    if len(pix) == 0:
        return rgb_arr.copy(), [[200, 200, 200]], [1.0]

    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pix)
    centers = km.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels, minlength=k).astype(np.float32)
    perc = (counts / (counts.sum() + 1e-6)).tolist()

    seg = rgb_arr.copy()
    idx = 0
    coords = np.argwhere(m)
    for y, x in coords:
        seg[y, x] = centers[labels[idx]]
        idx += 1

    return seg, centers.tolist(), perc


# =========================
# Visualización: contorno y panel
# =========================
def _draw_contour(rgb_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dibuja contorno principal en verde sobre la imagen RGB."""
    arr = rgb_arr.copy()
    cnts, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(arr, cnts, -1, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    return arr


def panel_like(rgb_pil: Image.Image, mask: np.ndarray, seg_rgb: np.ndarray,
               palette_rgb, palette_perc) -> np.ndarray:
    """
    Construye panel 2x2 + paleta:
      [ original | contorno ]
      [ overlay  | segmentada ]
      [ -------- paleta ----- ]
    """
    arr = np.array(rgb_pil)
    h, w = arr.shape[:2]

    # Ajusta tamaño/forma de la segmentada
    if seg_rgb.shape[:2] != (h, w):
        seg_rgb = cv2.resize(seg_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    if seg_rgb.dtype != np.uint8:
        seg_rgb = seg_rgb.astype(np.uint8)

    a = arr
    b = _draw_contour(arr.copy(), mask)

    over = arr.copy()
    over[mask.astype(bool)] = (0.6 * over[mask.astype(bool)] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    c = seg_rgb

    row1 = np.hstack([a, b])
    row2 = np.hstack([over, c])
    panel_w = row1.shape[1]  # 2*w

    pal_h = 60
    palette_img = np.zeros((pal_h, panel_w, 3), dtype=np.uint8)

    x0 = 0
    total = float(sum(palette_perc)) + 1e-6
    for color, p in zip(palette_rgb, palette_perc):
        span = int((p / total) * panel_w)
        x1 = min(panel_w, x0 + span)
        if x1 > x0:
            palette_img[:, x0:x1] = np.array(color, dtype=np.uint8)
        x0 = x1
    # Relleno final si queda hueco por redondeo
    if x0 < panel_w and len(palette_rgb) > 0:
        palette_img[:, x0:panel_w] = np.array(palette_rgb[-1], dtype=np.uint8)

    grid = np.vstack([row1, row2, palette_img])
    return grid
