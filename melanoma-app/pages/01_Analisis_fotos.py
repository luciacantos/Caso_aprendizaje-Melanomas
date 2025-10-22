# pages/01_Analisis_fotos.py
import io
import time
import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from utils.utils_image import ensure_rgb, robust_lesion_mask, geo_features_from_mask
from utils.utils_model import predict_proba_malignant, classify_image

st.set_page_config(page_title="Análisis de Melanoma", page_icon="🧬", layout="wide")

# ====== Estilos ======
st.markdown("""
<style>
body { background: #0c0f14; color:#e8eef8; }
h1,h2,h3 { color:#00e0ff; font-weight:800; }
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px; padding: 16px;
  box-shadow: 0 0 24px rgba(0,224,255,0.12);
}
.result-card {
  background: rgba(255,255,255,0.08);
  border-radius: 16px; padding: 22px; text-align:center;
}
.prob {
  font-size: 1.6rem; margin: 6px 0; font-weight: 800;
}
.badge {
  display:inline-block; padding:6px 12px; border-radius:999px;
  font-weight:700; color:#0c0f14; background:#22c55e;
}
.badge.red { background:#ef4444; }
.subtle { color:#9fb4c9; font-size:.9rem; }
.gradient-bar {
  height: 14px; border-radius: 8px;
  background: linear-gradient(90deg, #ffd43b, #ff8c00, #ff0000);
  position: relative; margin-top: 8px; margin-bottom: 4px;
}
.marker {
  position: absolute; top: -4px; width: 2px; height: 22px; background: #ffffff;
  box-shadow: 0 0 6px rgba(255,255,255,.8);
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Análisis de Melanoma")

# ====== Sidebar: parámetros de predicción ======
st.sidebar.subheader("⚙️ Parámetros de inferencia")

# 👉 Explicación clara del umbral maligno
st.sidebar.markdown("""
**¿Qué es el Umbral Maligno?**  
El umbral es el valor a partir del cual el modelo considera una lesión **maligna**.  
- Umbral **bajo (0.3–0.5)** → más sensible, detecta más casos sospechosos.  
- Umbral **alto (0.6–0.8)** → más estricto, reduce falsos positivos.  
Ajusta el valor según tu prioridad entre sensibilidad y precisión.
""")

thr = st.sidebar.slider("Umbral maligno", 0.0, 1.0, 0.50, 0.01)

# ====== Entradas: Subir | Cámara | URL ======
tab_up, tab_cam, tab_url = st.tabs(["📁 Subir imagen", "📸 Cámara", "🌐 URL"])

pil_img = None

with tab_up:
    up = st.file_uploader("Selecciona una imagen (JPG/PNG)", type=["jpg","jpeg","png"])
    if up:
        pil_img = Image.open(up).convert("RGB")

with tab_cam:
    cam = st.camera_input("Haz una foto")
    if cam:
        pil_img = Image.open(cam).convert("RGB")

with tab_url:
    u = st.text_input("Pega la URL directa a una imagen")
    if u:
        try:
            resp = requests.get(u, timeout=10)
            resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.image(pil_img, caption="Imagen desde URL", use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo cargar la imagen de esa URL: {e}")

if pil_img is None:
    st.info("Sube una imagen, usa la cámara o indica una URL para comenzar.")
    st.stop()

# ====== Modelo desde app principal ======
model = st.session_state.get("model")
if model is None:
    st.error("❌ No se ha cargado el modelo. Vuelve a la portada para cargarlo.")
    st.stop()

# ====== Procesado: máscara, borde y “heatmap estilo amigo” ======
img_rgb = ensure_rgb(pil_img)
mask = robust_lesion_mask(img_rgb)                  # {0,1}
geo  = geo_features_from_mask(mask)

# ---- Imagen con contorno
bordered = np.array(img_rgb)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(bordered, cnts, -1, (0,255,0), 3)

# ---- “Heatmap estilo amigo” (coloreado JET) sobre imagen
def friend_like_colormap(pil: Image.Image, lesion_mask: np.ndarray, size=(224,224), alpha=0.45):
    """Genera un heatmap simple: invertimos brillo (lesión oscura -> alto),
    normalizamos, aplicamos color JET y lo mezclamos con la imagen."""
    arr = np.array(pil.resize(size))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v = hsv[...,2]
    inv = 255 - v
    inv = cv2.GaussianBlur(inv, (5,5), 0)
    inv_norm = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    heat_bgr = cv2.applyColorMap(inv_norm.astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # reescala máscara al mismo tamaño
    m = cv2.resize((lesion_mask*255).astype(np.uint8), (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    m_bool = m > 0

    # mezcla
    base = arr.copy()
    base[m_bool] = ((1-alpha)*base[m_bool] + alpha*heat[m_bool]).astype(np.uint8)

    # score rojo medio dentro de la lesión
    score = 0.0
    if m_bool.any():
        score = float((heat[...,0][m_bool].astype(np.float32) / 255.0).mean())  # canal R (RGB)
    return base, score  # imagen RGB con overlay, score en [0,1]

heat_img, score = friend_like_colormap(img_rgb, mask, size=(224,224), alpha=0.50)

# ====== Predicción ======
t0 = time.time()
p_malign = predict_proba_malignant(model, img_rgb)
dt_ms = int((time.time() - t0)*1000)
p_benign = 1.0 - p_malign
pred = "Maligno" if p_malign >= thr else "Benigno"

# ====== Layout visual ======
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Imagen con borde")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(bordered, use_container_width=True, caption="Contorno de la lesión")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### Mapa de calor")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(heat_img, use_container_width=True, caption="Colormap JET sobre la lesión")

    pct = int(round(score * 100))
    st.markdown("<div class='gradient-bar'>", unsafe_allow_html=True)
    st.markdown(f"<div class='marker' style='left: {pct}%;'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Intensidad media en zona coloreada: **{pct}%** (0% amarillo tenue → 100% rojo intenso)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
c3, c4 = st.columns(2, gap="large")

with c3:
    st.markdown("## 📊 Resultado IA")
    badge_cls = "red" if pred == "Maligno" else ""
    st.markdown(f"""
    <div class='result-card'>
      <span class='badge {badge_cls}'>{pred}</span>
      <div class='prob'>Maligno: {round(p_malign*100)}% &nbsp;|&nbsp; Benigno: {round(p_benign*100)}%</div>
      <div class='subtle'>Tiempo de inferencia: {dt_ms} ms · Umbral: {thr:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    if pred == "Maligno":
        st.warning("⚠️ Resultado orientativo compatible con lesión sospechosa. Consulta con dermatología.")
    else:
        st.success("✅ No se observan patrones preocupantes. Realiza auto-chequeo y vigila cambios.")

with c4:
    st.markdown("## 🧩 A·B·C·D clínico")
    asim = 100 - int(round(geo["sym_global_pct"]))
    bordes_txt = "Irregulares" if geo["sym_global_pct"] < 60 else "Definidos"
    st.markdown(f"""
    <div class='result-card'>
      <p><b>A — Asimetría:</b> {asim}% irregular</p>
      <p><b>B — Bordes:</b> {bordes_txt}</p>
      <p><b>C — Color:</b> Multitono (por mapa coloreado)</p>
      <p><b>D — Diámetro/Área:</b> {geo["area_pct"]:.1f}% del área de imagen</p>
      <div class='subtle'>Simetría global estimada: {geo["sym_global_pct"]:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("⚠️ Análisis orientativo. No sustituye la valoración médica profesional.")
