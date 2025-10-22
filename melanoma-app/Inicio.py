# Inicio.py
import base64
from pathlib import Path
import streamlit as st
from utils.utils_model import load_model, MODEL_PATH

st.set_page_config(page_title="Detector IA de Melanomas — Inicio", page_icon="🩺", layout="wide")

# ===== Ajuste de ruta del modelo =====
# El modelo ahora se guarda dentro de: melanoma-app/modelo/
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "modelo" / "deepcnn6res_clean_best.pt"

# ===== Util =====
def img_as_base64(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

# ===== Estilos =====
st.markdown("""
<style>
html, body, #root, .stApp,[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 50% -10%, #151a22 0%, #0f141a 45%, #0d1218 100%) !important;
}
[data-testid="stSidebar"] { background: #1b2433 !important; border-right: 1px solid rgba(255,255,255,0.06); }
header, [data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; }
body{ color:#e8f2ff !important; text-align:center !important; font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial !important; }

.hero-title{ text-transform: uppercase; font-weight: 900; letter-spacing: 1.3px; text-align: center;
  font-size: clamp(36px, 6.2vw, 64px); margin: 22px auto 12px auto;
  background: linear-gradient(90deg, #1dd6ff, #5c8cff, #b176ff); -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 2px 18px rgba(39,224,255,.22)); }

.hr{ width:120px;height:5px;border-radius:6px;margin:18px auto 26px auto; background: linear-gradient(90deg,#00e0ff,#6aa4ff,#b67fff); }

.stats{ display:flex; flex-wrap:wrap; justify-content:center; gap:18px; margin: 0 auto 36px auto; max-width: 1100px; }
.card{ width: 320px; background: rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:16px; padding:18px 20px; text-align:center; box-shadow: 0 12px 30px rgba(0,0,0,.3); }
.card h3{ margin:0 0 6px 0; font-size:18px; color:#9bd7ff; }
.card p{ margin:0; color:#c7d6e6; font-size:15px; }

.hero-img{ display:block; margin: 0 auto 20px auto; width: min(600px, 90%); border-radius:14px; box-shadow: 0 14px 32px rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.08); }

.cta-wrap{ display:flex; justify-content:center; align-items:center; width:100%; margin: 10px 0 40px 0; }
.stButton>button{
  background: linear-gradient(90deg,#06b6d4,#2563eb) !important; color:#fff !important; font-weight:900 !important;
  border:none !important; border-radius:14px !important; padding: 18px 32px !important; font-size: 20px !important;
  box-shadow: 0 18px 50px rgba(6,182,212,.28) !important; transition: all .15s ease !important;
}
.stButton>button:hover{ transform: translateY(-2px) scale(1.03); }

.block-container { padding-top: 1.2rem !important; }
.disclaimer{ color:#a4b6c9; font-size:13px; margin-top:10px }
#MainMenu, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ===== Sidebar =====
st.sidebar.markdown("### ⚙️ Modelo (PyTorch)")
st.sidebar.caption("Ruta de pesos (.pt):")
st.sidebar.code(str(MODEL_PATH))

@st.cache_resource
def _load():
    return load_model()

try:
    st.sidebar.info("Cargando modelo…")
    st.session_state.model = _load()
    st.sidebar.success("Modelo cargado ✅")
except Exception as e:
    st.sidebar.error(f"No se pudo cargar el modelo:\n{e}")
    st.stop()

st.sidebar.caption("Pesos en uso:")
st.sidebar.code(MODEL_PATH.name if MODEL_PATH.exists() else "⚠️ No encontrado")

# ===== Hero principal =====
st.markdown("<div class='hero-title'>Detector inteligente de melanomas</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# Tarjetas informativas
st.markdown("""
<div class="stats">
  <div class="card">
    <h3>¿Por qué importa?</h3>
    <p>El cáncer de piel es frecuente y el <b>melanoma</b> es su forma más agresiva.
       Detectarlo a tiempo mejora el pronóstico.</p>
  </div>
  <div class="card">
    <h3>¿Qué aporta la IA?</h3>
    <p>Ayuda a <b>priorizar</b> casos y aporta explicaciones visuales (zonas relevantes) para la revisión clínica.</p>
  </div>
  <div class="card">
    <h3>Uso responsable</h3>
    <p>Es una herramienta <b>orientativa</b> y no sustituye la valoración médica profesional.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ===== Imagen + Botón debajo =====
HERO_IMG = Path("assets/hero.jpg")
hero_b64 = img_as_base64(HERO_IMG)
if hero_b64:
    st.markdown(f"<img src='data:image/jpeg;base64,{hero_b64}' class='hero-img'/>", unsafe_allow_html=True)

# Botón centrado debajo
st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
if st.button("🔎  Analizar una imagen"):
    st.switch_page("pages/01_Analisis_fotos.py")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='disclaimer'>*Uso académico. No reemplaza diagnóstico profesional.*</div>", unsafe_allow_html=True)
