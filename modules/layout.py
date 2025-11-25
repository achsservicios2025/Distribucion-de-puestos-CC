# modules/layout.py
import streamlit as st
from modules.database import save_setting, get_all_settings
from pathlib import Path

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def admin_appearance_ui(conn):
    st.subheader("Apariencia y branding")
    settings = get_all_settings(conn)
    col1, col2 = st.columns(2)
    with col1:
        primary = st.color_picker("Color primario", value=settings.get("primary","#00A04A"))
        accent = st.color_picker("Color acento", value=settings.get("accent","#006B32"))
        bg = st.color_picker("Color fondo", value=settings.get("bg","#ffffff"))
    with col2:
        text = st.color_picker("Color texto", value=settings.get("text","#111111"))
        font = st.selectbox("Fuente", ["Poppins","Montserrat","Roboto","Inter","Lato"], index=0)
        site_title = st.text_input("Título del sitio", value=settings.get("site_title","Gestor de Puestos y Salas — ACHS Servicios"))
    logo = st.file_uploader("Subir logo (opcional)", type=["png","jpg","jpeg"])
    if st.button("Guardar apariencia"):
        save_setting(conn,"primary",primary)
        save_setting(conn,"accent",accent)
        save_setting(conn,"bg",bg)
        save_setting(conn,"text",text)
        save_setting(conn,"font",font)
        save_setting(conn,"site_title",site_title)
        if logo is not None:
            with open(STATIC_DIR/"logo.png","wb") as f:
                f.write(logo.getbuffer())
            save_setting(conn,"logo_path",str(STATIC_DIR/"logo.png"))
        st.success("Apariencia guardada. Recarga la página para ver cambios.")

def apply_appearance_styles(conn):
    settings = get_all_settings(conn)
    font = settings.get("font","Poppins")
    primary = settings.get("primary","#00A04A")
    bg = settings.get("bg","#ffffff")
    text = settings.get("text","#111111")
    css = f"""
    @import url('https://fonts.googleapis.com/css2?family={font.replace(' ','+')}');
    :root {{
        --primary:{primary};
        --bg:{bg};
        --text:{text};
    }}
    body {{ background: var(--bg); color: var(--text); font-family: '{font}', sans-serif; }}
    .stButton>button {{ background-color: var(--primary); color: white; border-radius:8px; }}
    .stSelectbox>div, .stTextInput>div {{ border-radius:8px; }}
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

