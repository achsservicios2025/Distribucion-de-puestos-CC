# modules/layout.py
import streamlit as st
from pathlib import Path
from modules.database import save_setting, get_all_settings

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)


# -----------------------------
# Admin UI: Apariencia/branding
# -----------------------------
def admin_appearance_ui(conn):
    st.subheader("Apariencia y branding")

    settings = get_all_settings(conn) or {}

    # Defaults seguros
    default_primary = settings.get("primary", "#00A04A")
    default_accent = settings.get("accent", "#006B32")
    default_bg = settings.get("bg", "#ffffff")
    default_text = settings.get("text", "#111111")
    default_site_title = settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios")

    font_options = ["Poppins", "Montserrat", "Roboto", "Inter", "Lato"]
    saved_font = settings.get("font", "Poppins")
    try:
        font_index = font_options.index(saved_font)
    except ValueError:
        font_index = 0

    col1, col2 = st.columns(2)

    with col1:
        primary = st.color_picker("Color primario", value=str(default_primary))
        accent = st.color_picker("Color acento", value=str(default_accent))
        bg = st.color_picker("Color fondo", value=str(default_bg))

    with col2:
        text = st.color_picker("Color texto", value=str(default_text))
        font = st.selectbox("Fuente", font_options, index=font_index)
        site_title = st.text_input("Título del sitio", value=str(default_site_title))

    st.divider()

    current_logo_path = settings.get("logo_path", "")
    if current_logo_path:
        try:
            st.caption("Logo actual:")
            st.image(current_logo_path, width=220)
        except Exception:
            st.warning("No se pudo cargar el logo actual (ruta inválida).")

    logo = st.file_uploader("Subir logo (opcional)", type=["png", "jpg", "jpeg"])

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Guardar apariencia", type="primary", use_container_width=True):
            ok = True
            ok &= bool(save_setting(conn, "primary", primary))
            ok &= bool(save_setting(conn, "accent", accent))
            ok &= bool(save_setting(conn, "bg", bg))
            ok &= bool(save_setting(conn, "text", text))
            ok &= bool(save_setting(conn, "font", font))
            ok &= bool(save_setting(conn, "site_title", site_title))

            if logo is not None:
                # normalizamos a png
                logo_path = STATIC_DIR / "logo.png"
                try:
                    with open(logo_path, "wb") as f:
                        f.write(logo.getbuffer())
                    ok &= bool(save_setting(conn, "logo_path", str(logo_path)))
                except Exception as e:
                    ok = False
                    st.error(f"No se pudo guardar el archivo del logo: {e}")

            if ok:
                st.success("Apariencia guardada. Recarga la página si no ves cambios.")
            else:
                st.error("Hubo un problema guardando la apariencia.")

    with c2:
        if st.button("Restablecer colores (default)", use_container_width=True):
            save_setting(conn, "primary", "#00A04A")
            save_setting(conn, "accent", "#006B32")
            save_setting(conn, "bg", "#ffffff")
            save_setting(conn, "text", "#111111")
            save_setting(conn, "font", "Poppins")
            st.success("Listo. Recarga la página para ver cambios.")


# -----------------------------
# Apply global styles
# -----------------------------
def apply_appearance_styles(conn):
    settings = get_all_settings(conn) or {}

    font = settings.get("font", "Poppins")
    primary = settings.get("primary", "#00A04A")
    accent = settings.get("accent", "#006B32")
    bg = settings.get("bg", "#ffffff")
    text = settings.get("text", "#111111")

    # Sanitizar fuente para URL google fonts
    font_q = str(font).strip().replace(" ", "+")

    css = f"""
    <style>
    /* Google font */
    @import url('https://fonts.googleapis.com/css2?family={font_q}:wght@300;400;500;600;700&display=swap');

    :root {{
        --primary: {primary};
        --accent: {accent};
        --bg: {bg};
        --text: {text};
        --radius: 14px;
    }}

    html, body, [class*="css"] {{
        font-family: '{font}', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        color: var(--text);
    }}

    /* background (Streamlit app containers) */
    .stApp {{
        background: var(--bg);
    }}

    /* Headings a bit nicer */
    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: var(--primary);
        color: white;
        border-radius: var(--radius);
        border: 0;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        filter: brightness(0.95);
    }}

    /* Secondary buttons look (when Streamlit uses aria) */
    .stButton > button[kind="secondary"] {{
        background-color: transparent;
        color: var(--text);
        border: 1px solid rgba(0,0,0,0.15);
    }}

    /* Inputs */
    .stTextInput input, .stTextArea textarea {{
        border-radius: var(--radius) !important;
    }}

    /* Selectbox container */
    div[data-baseweb="select"] > div {{
        border-radius: var(--radius) !important;
    }}

    /* Tabs a bit rounded */
    button[data-baseweb="tab"] {{
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
    }}

    /* Nice cards (optional helper class) */
    .soft-card {{
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
