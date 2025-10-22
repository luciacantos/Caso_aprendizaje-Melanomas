# utils/utils_gradcam.py
import numpy as np
import cv2
import torch
from PIL import Image

def _pick_target_layer(model, layer_name: str):
    """
    Capas recomendadas para tu arquitectura:
      - 'res6' -> model.res6.conv[3] (muy semántica, calor más compacto)
      - 'res5' -> model.res5.conv[3] (más cobertura; RECOMENDADA)
      - 'b6'   -> model.b6[0]
      - 'b5'   -> model.b5[0]
    """
    if layer_name == "res6": return model.res6.conv[3]
    if layer_name == "res5": return model.res5.conv[3]
    if layer_name == "b6":   return model.b6[0]
    if layer_name == "b5":   return model.b5[0]
    raise ValueError("layer_name debe ser 'res6','res5','b6' o 'b5'")

def _normalize01(x: np.ndarray, p=100) -> np.ndarray:
    # normaliza por percentil para evitar saturación por outliers
    hi = np.percentile(x, p)
    x = np.clip(x, 0, hi)
    x = x - x.min()
    return x / (x.max() + 1e-8)

def _prep_focus_mask(focus_mask, size_wh, dilate=25, blur=31):
    """Devuelve una máscara suave (0..1) al tamaño size_wh=(W,H)."""
    W, H = size_wh
    if focus_mask is None:
        return None
    m = (focus_mask.astype(np.uint8) * 255)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k, iterations=1)
    if blur and blur > 0:
        m = cv2.GaussianBlur(m, (blur, blur), 0)
    return (m.astype(np.float32) / 255.0)

def generate_gradcam(
    model,
    pil_img: Image.Image,
    target_class: int = 1,
    layer_name: str = "res5",   # <- capa más “ancha”
    smooth: int = 16,           # más muestras → más suave
    sigma: float = 0.08,        # ruido más bajo
    focus_mask: np.ndarray | None = None,
    alpha_overlay: float = 0.5,
    gamma: float = 0.7,         # <1 expande el calor; >1 lo contrae
    norm_percentile: int = 99,  # normaliza por percentil alto
    mask_dilate: int = 25,      # “ensancha” la máscara
) -> Image.Image:
    """Grad-CAM con Smooth + máscara dilatada + normalización por percentil."""
    model.eval()
    W, H = pil_img.size
    size = (224, 224)

    arr0 = np.array(pil_img.resize(size)).astype(np.float32) / 255.0
    x0 = torch.from_numpy(arr0).permute(2, 0, 1).unsqueeze(0).float()
    x0.requires_grad_(True)

    target_layer = _pick_target_layer(model, layer_name)
    feats_list, grads_list = [], []

    def fwd_hook(_, __, output): feats_list.append(output)
    def bwd_hook(_, grad_input, grad_output): grads_list.append(grad_output[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    cams = []
    n_runs = max(1, int(smooth))
    for _ in range(n_runs):
        feats_list.clear(); grads_list.clear()
        x = (x0 + torch.randn_like(x0)*sigma).clamp(0, 1) if smooth>0 else x0
        logits = model(x)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        fmap = feats_list[0].detach().cpu().numpy()[0]  # [C,Hf,Wf]
        grad = grads_list[0].detach().cpu().numpy()[0]  # [C,Hf,Wf]
        weights = grad.mean(axis=(1, 2))
        cam = np.maximum((weights[:, None, None] * fmap).sum(axis=0), 0.0)
        cam = cv2.resize(cam, size)
        cams.append(cam)

    cam = np.mean(np.stack(cams, 0), 0)
    cam = _normalize01(cam, p=norm_percentile)
    cam = np.power(cam, gamma)  # gamma<1 “abre” el heatmap

    cam_big = cv2.resize(cam, (W, H))
    fm = _prep_focus_mask(focus_mask, (W, H), dilate=mask_dilate, blur=31)
    if fm is not None:
        cam_big = cam_big * (0.85*fm + 0.15)  # atenúa fondo, prioriza lesión

    heat = cv2.applyColorMap(np.uint8(255 * cam_big), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.array(pil_img)
    overlay = cv2.addWeighted(base, 1 - alpha_overlay, heat, alpha_overlay, 0)

    h1.remove(); h2.remove()
    return Image.fromarray(overlay)

# --------- Alternativa: Eigen-CAM (cubre más “la forma” del objeto) ---------
def eigen_cam(
    model,
    pil_img: Image.Image,
    layer_name: str = "res5",
    focus_mask: np.ndarray | None = None,
    alpha_overlay: float = 0.5,
    mask_dilate: int = 25,
) -> Image.Image:
    """Eigen-CAM (sin gradientes). Suele dar cobertura amplia del objeto."""
    model.eval()
    W, H = pil_img.size
    size = (224, 224)
    arr0 = np.array(pil_img.resize(size)).astype(np.float32)/255.0
    x0 = torch.from_numpy(arr0).permute(2,0,1).unsqueeze(0).float()

    feats = []
    layer = _pick_target_layer(model, layer_name)
    h = layer.register_forward_hook(lambda m,i,o: feats.append(o))
    _ = model(x0)
    h.remove()

    fmap = feats[0][0].detach().cpu().numpy()      # [C,Hf,Wf]
    C,Hf,Wf = fmap.shape
    A = fmap.reshape(C, -1).T                      # [Hf*Wf, C]
    # 1ª componente principal sobre canales
    U, S, Vt = np.linalg.svd(A - A.mean(0, keepdims=True), full_matrices=False)
    pc = (A @ Vt[0][:,None]).reshape(Hf, Wf)
    cam = np.maximum(pc, 0)
    cam = cam - cam.min(); cam = cam / (cam.max()+1e-8)

    cam_big = cv2.resize(cam, (W,H))
    fm = _prep_focus_mask(focus_mask, (W,H), dilate=mask_dilate, blur=31)
    if fm is not None:
        cam_big = cam_big * (0.85*fm + 0.15)

    heat = cv2.applyColorMap(np.uint8(255*cam_big), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.array(pil_img)
    return Image.fromarray(cv2.addWeighted(base, 1-alpha_overlay, heat, alpha_overlay, 0))
