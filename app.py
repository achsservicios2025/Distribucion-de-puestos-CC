import streamlit as st
import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Optional
import random
import datetime
import numpy as np
from io import BytesIO
from PIL import Image
from fpdf import FPDF

# ---------------------------------------------------------
# 1) CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gestor de Puestos y Salas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2) IMPORTS MÓDULOS
# ---------------------------------------------------------
from modules.database import (
    get_conn, init_db, insert_distribution, clear_distribution,
    read_distribution_df, save_setting, get_all_settings,
    add_reservation, user_has_reservation, list_reservations_df,
    add_room_reservation, get_room_reservations_df,
    count_monthly_free_spots, delete_reservation_from_db,
    delete_room_reservation_from_db, perform_granular_delete,
    ensure_reset_table, save_reset_token, validate_and_consume_token,
    get_worksheet
)

try:
    from modules.database import delete_distribution_row, delete_distribution_rows_by_indices
except ImportError:
    def delete_distribution_row(conn, piso, equipo, dia):
        return False

    def delete_distribution_rows_by_indices(conn, indices):
        return False

from modules.auth import get_admin_credentials
from modules.layout import admin_appearance_ui, apply_appearance_styles
from modules.seats import compute_distribution_from_excel, compute_distribution_variants
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones

from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 3) CONSTANTES / DIRS
# ---------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

PLANOS_DIR = Path("modules/planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

for d in (PLANOS_DIR, DATA_DIR, COLORED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 4) SESSION STATE UI
# ---------------------------------------------------------
st.session_state.setdefault("ui", {
    "app_title": "Gestor de Puestos y Salas",
    "bg_color": "#ffffff",
    "logo_path": "assets/logo.png",
    "title_font_size": 64,
    "logo_width": 420,
})

st.session_state.setdefault("screen", "Administrador")
st.session_state.setdefault("forgot_mode", False)

# ---------------------------------------------------------
# 4.5) DB + SETTINGS
# ---------------------------------------------------------
conn = get_conn()

if "db_initialized" not in st.session_state:
    with st.spinner("Conectando a Google Sheets..."):
        init_db(conn)
    st.session_state["db_initialized"] = True

apply_appearance_styles(conn)

settings = get_all_settings(conn) or {}
st.session_state["ui"]["app_title"] = settings.get("site_title", st.session_state["ui"]["app_title"])
st.session_state["ui"]["logo_path"] = settings.get("logo_path", st.session_state["ui"]["logo_path"])

# ---------------------------------------------------------
# 5) CSS
#   ✅ "Acceder" mismo ancho que "Olvidaste..." (fijo)
#   ✅ "Acceder" alineado a la derecha
# ---------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
  background: {st.session_state.ui["bg_color"]};
}}
header {{
  visibility: hidden;
  height: 0px;
}}

div[data-testid="stAppViewContainer"] > .main {{
  padding-top: 0rem !important;
}}
section.main > div {{
  padding-top: 0rem !important;
}}

.block-container {{
  max-width: 100% !important;
  padding-top: 0.75rem !important;
  padding-left: 5cm !important;
  padding-right: 5cm !important;
}}

.mk-content {{
  width: 100%;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}}

html, body, [class*="css"] {{
  font-size: 20px !important;
}}
h1 {{ font-size: 48px !important; }}
h2 {{ font-size: 40px !important; }}
h3 {{ font-size: 32px !important; }}
p, li, label, span {{ font-size: 20px !important; }}

div[data-baseweb="input"] input {{
  font-size: 20px !important;
  padding-top: 14px !important;
  padding-bottom: 14px !important;
}}

div[data-baseweb="select"] > div {{
  font-size: 20px !important;
  min-height: 56px !important;
  border-radius: 18px !important;
}}

.stButton button {{
  font-size: 20px !important;
  font-weight: 900 !important;
  padding: 12px 18px !important;
  border-radius: 16px !important;
}}

.mk-title {{
  text-align: center;
  font-weight: 900;
  margin: 0;
  line-height: 1.05;
}}

/* ✅ Alinear la columna derecha al borde derecho */
.mk-right {{
  display: flex;
  justify-content: flex-end;
}}

/*
✅ Forzamos que "Acceder" mida EXACTAMENTE lo mismo que "Olvidaste tu contraseña".
Ajusta este ancho si cambias el texto del botón izquierdo.
Con los tamaños actuales, 320px calza bien para ambos.
*/
button[kind="primary"][data-testid="baseButton-primary"] {{
  width: 320px !important;
}}

/* También dejamos el botón izquierdo con el mismo ancho */
button[data-testid="baseButton-secondary"] {{
  width: 320px !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def clean_pdf_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\r", "")
           .replace("\t", " ")
           .replace("–", "-")
           .replace("—", "-")
           .replace("−", "-")
           .replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("‘", "'")
           .replace("•", "-")
           .replace("\u00a0", " "))
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s

def go(screen: str):
    st.session_state["screen"] = screen

# ---------------------------------------------------------
# TOPBAR
# ---------------------------------------------------------
def render_topbar_and_menu():
    logo_path = Path(st.session_state.ui["logo_path"])
    size = int(st.session_state.ui.get("title_font_size", 64))
    title = st.session_state.ui.get("app_title", "Gestor de Puestos y Salas")

    c1, c2, c3 = st.columns([1.2, 3.6, 1.2], vertical_alignment="center")

    with c1:
        if logo_path.exists():
            st.image(str(logo_path), width=int(st.session_state.ui.get("logo_width", 420)))
        else:
            st.write("🧩 (Logo aquí)")

    with c2:
        st.markdown(f"<div class='mk-title' style='font-size:{size}px;'>{title}</div>", unsafe_allow_html=True)

    with c3:
        menu_choice = st.selectbox(
            "Menú",
            ["—", "Reservas", "Ver Distribución y Planos"],
            index=0,
            key="top_menu_select",
        )
        if menu_choice == "Reservas":
            go("Reservas")
        elif menu_choice == "Ver Distribución y Planos":
            go("Planos")

# ---------------------------------------------------------
# ADMIN
# ---------------------------------------------------------
def screen_admin(conn):
    st.subheader("Administrador")
    st.session_state.setdefault("forgot_mode", False)

    if not st.session_state["forgot_mode"]:
        st.text_input("Ingresar correo", key="admin_login_email")
        st.text_input("Contraseña", type="password", key="admin_login_pass")

        # ✅ Misma fila. Izq botón izquierdo, der botón derecho pegado a borde.
        c1, c2 = st.columns([1, 1], vertical_alignment="center")
        with c1:
            if st.button("Olvidaste tu contraseña", key="btn_admin_forgot"):
                st.session_state["forgot_mode"] = True
                st.rerun()
        with c2:
            st.markdown("<div class='mk-right-wrap'><div class='mk-right-inner'>", unsafe_allow_html=True)
            if st.button("Acceder", type="primary", key="btn_admin_login"):
                e = st.session_state.get("admin_login_email", "").strip()
                p = st.session_state.get("admin_login_pass", "")
                if not e or not p:
                    st.warning("Completa correo y contraseña.")
                else:
                    st.success("Login recibido (validación real pendiente).")
            st.markdown("</div></div>", unsafe_allow_html=True)

    else:
        st.text_input("Correo de acceso", key="admin_reset_email")
        st.caption("Ingresa el código recibido en tu correo.")
        st.text_input("Código", key="admin_reset_code")

        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
        with c1:
            if st.button("Volver a Acceso", key="btn_admin_back"):
                st.session_state["forgot_mode"] = False
                st.rerun()
        with c2:
            if st.button("Enviar código", type="primary", key="btn_admin_send_code"):
                e = st.session_state.get("admin_reset_email", "").strip()
                if not e:
                    st.warning("Ingresa tu correo.")
                else:
                    st.success("Código enviado (simulado).")
        with c3:
            if st.button("Validar código", type="primary", key="btn_admin_validate"):
                c = st.session_state.get("admin_reset_code", "").strip()
                if not c:
                    st.warning("Ingresa el código.")
                else:
                    st.success("Código validado (simulado).")

# ---------------------------------------------------------
# RESERVAS (placeholder)
# ---------------------------------------------------------
def screen_reservas_tabs(conn):
    st.subheader("Reservas")
    tabs = st.tabs(["Reservar Puesto Flex", "Reserva Salas de Reuniones", "Mis Reservas y Listados"])
    with tabs[0]:
        st.info("Pega aquí tu pantalla completa de 'Reservar Puesto Flex'.")
    with tabs[1]:
        st.info("Pega aquí tu pantalla completa de 'Reserva Salas de Reuniones'.")
    with tabs[2]:
        st.info("Pega aquí tu pantalla completa de 'Mis Reservas y Listados'.")

# ---------------------------------------------------------
# DESCARGAS
# ---------------------------------------------------------
def _df_to_xlsx_bytes(df: pd.DataFrame, sheet_name="data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        (df if df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

def screen_descargas_distribucion_planos(conn):
    st.subheader("Ver Distribución y Planos (solo descarga)")
    t1, t2 = st.tabs(["Distribución", "Planos"])

    with t1:
        st.markdown("### Distribución (Descargar)")
        df = read_distribution_df(conn)
        if df is None or df.empty:
            st.warning("No hay distribución cargada para descargar.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            xlsx_bytes = _df_to_xlsx_bytes(df, sheet_name="distribucion")
            st.download_button(
                "⬇️ Descargar Distribución (XLSX)",
                data=xlsx_bytes,
                file_name="distribucion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with t2:
        st.markdown("### Planos (Descargar)")
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"]
        imgs = []
        for pat in patterns:
            imgs.extend(sorted(PLANOS_DIR.glob(pat)))

        if not imgs:
            st.warning("No se encontraron imágenes de planos.")
            st.write(f"Ruta buscada: `{PLANOS_DIR.resolve()}`")
        else:
            selected = st.selectbox("Selecciona un plano", [p.name for p in imgs], key="dl_plano_sel")
            img_path = next(p for p in imgs if p.name == selected)
            st.image(str(img_path), use_container_width=True)
            st.download_button(
                "⬇️ Descargar plano (imagen)",
                data=img_path.read_bytes(),
                file_name=img_path.name,
                mime="image/png" if img_path.suffix.lower() == ".png" else "image/jpeg",
            )

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
st.markdown("<div class='mk-content'>", unsafe_allow_html=True)
render_topbar_and_menu()
st.divider()

screen = st.session_state.get("screen", "Administrador")

if screen == "Administrador":
    screen_admin(conn)
elif screen == "Reservas":
    screen_reservas_tabs(conn)
elif screen == "Planos":
    screen_descargas_distribucion_planos(conn)
else:
    st.session_state["screen"] = "Administrador"
    screen_admin(conn)

st.markdown("</div>", unsafe_allow_html=True)

