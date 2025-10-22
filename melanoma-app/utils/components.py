# utils/components.py
from __future__ import annotations
import io
from typing import Optional
import streamlit as st
import numpy as np
from PIL import Image

def sidebar_settings():
    st.sidebar.title("⚙️ Ajustes")
    do_hair = st.sidebar.checkbox("Eliminar vello (rápido)", value=True)
    do_color = st.sidebar.checkbox("Balance de color (rápido)", value=True)
    privacy = st.sidebar.checkbox("Guardar nada en servidor (solo memoria)", value=True)
    return do_hair, do_color, privacy


def image_inputs(allow_synthetic=True) -> Optional[Image.Image]:
    tabs = ["📤 Subir", "📷 Cámara", "🔗 URL"]
    if allow_synthetic:
        tabs.append("🧪 Ejemplo")
    tab_objs = st.tabs(tabs)
    img = None

    with tab_objs[0]:
        up = st.file_uploader("Suelta una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up).convert("RGB")

    with tab_objs[1]:
        cam = st.camera_input("Haz una foto")
        if cam:
            img = Image.open(cam).convert("RGB")

    with tab_objs[2]:
        url = st.text_input("Pega una URL de imagen")
        if st.button("Cargar desde URL") and url:
            try:
                import requests
                resp = requests.get(url, timeout=5)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                st.warning(f"No se pudo descargar: {e}")

    if allow_synthetic and len(tab_objs) > 3:
        with tab_objs[3]:
            if st.button("Generar ejemplo sintético"):
                from utils.utils_image import synthetic_demo_image
                img = synthetic_demo_image()

    return img


def download_panel_and_json(panel_np: np.ndarray, prob: float, palette_rgb, palette_perc):
    buf = io.BytesIO()
    Image.fromarray(panel_np).save(buf, format="PNG")
    st.download_button("⬇️ Descargar panel (PNG)", data=buf.getvalue(),
                       file_name="panel_resultados.png", mime="image/png")

    report = {
        "prob_malign": round(float(prob), 4),
        "palette_rgb": [list(map(int, c)) for c in palette_rgb],
        "palette_perc": [round(float(x), 4) for x in palette_perc],
    }
    st.download_button("⬇️ Descargar informe (JSON)",
                       data=io.BytesIO((str(report)).encode("utf-8")),
                       file_name="informe_skincheck.json", mime="application/json")
