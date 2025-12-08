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

# Inicio = Administrador (pantalla principal)
st.session_state.setdefault("screen", "Administrador")
st.session_state.setdefault("forgot_mode", False)

# ✅ sesión admin
st.session_state.setdefault("is_admin", False)

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

/* mismo ancho para ambos botones del login */
button[kind="primary"][data-testid="baseButton-primary"] {{
  width: 320px !important;
}}
button[data-testid="baseButton-secondary"] {{
  width: 320px !important;
}}

/* ✅ Botón-logo invisible pero clickeable */
.mk-logo-btn button {{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}}
.mk-logo-btn button:hover {{
  filter: brightness(0.98);
}}
.mk-logo-btn button:focus {{
  outline: none !important;
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

def _safe_sheet_lookup(sheets: dict, want: list[str]) -> Optional[pd.DataFrame]:
    """Busca una hoja por nombres posibles, case-insensitive, con contains."""
    if not sheets:
        return None
    norm = {str(k).strip().lower(): k for k in sheets.keys()}
    for w in want:
        w0 = w.strip().lower()
        if w0 in norm:
            return sheets[norm[w0]]
    for w in want:
        w0 = w.strip().lower()
        hit = next((orig for low, orig in norm.items() if w0 in low), None)
        if hit:
            return sheets[hit]
    return None

def _piso_to_label(piso_any) -> str:
    """
    Tu seats devuelve piso como string numérico "1".
    Tu app/DB suele usar "Piso 1".
    """
    if piso_any is None:
        return "Piso 1"
    s = str(piso_any).strip()
    if not s:
        return "Piso 1"
    if s.lower().startswith("piso"):
        return s
    nums = re.findall(r"\d+", s)
    return f"Piso {nums[0]}" if nums else f"Piso {s}"

def admin_logout():
    st.session_state["is_admin"] = False
    st.session_state["forgot_mode"] = False
    go("Administrador")
    st.rerun()

# ---------------------------------------------------------
# TOPBAR
# ---------------------------------------------------------
def render_topbar_and_menu():
    logo_path = Path(st.session_state.ui["logo_path"])
    size = int(st.session_state.ui.get("title_font_size", 64))
    title = st.session_state.ui.get("app_title", "Gestor de Puestos y Salas")
    logo_w = int(st.session_state.ui.get("logo_width", 420))

    c1, c2, c3 = st.columns([1.2, 3.6, 1.2], vertical_alignment="center")

    with c1:
        if logo_path.exists():
            st.markdown("<div class='mk-logo-btn'>", unsafe_allow_html=True)
            if st.button(" ", key="tb_logo_home_btn"):
                go("Administrador")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.image(str(logo_path), width=logo_w)
        else:
            if st.button("🧩 Inicio", key="tb_logo_home_fallback"):
                go("Administrador")
                st.rerun()

    with c2:
        st.markdown(
            f"<div class='mk-title' style='font-size:{size}px;'>{title}</div>",
            unsafe_allow_html=True
        )

    with c3:
        menu_choice = st.selectbox(
            "Menú",
            ["—", "Inicio", "Reservas", "Ver Distribución y Planos"],
            index=0,
            key="tb_top_menu_select",
        )
        if menu_choice == "Inicio":
            go("Administrador")
        elif menu_choice == "Reservas":
            go("Reservas")
        elif menu_choice == "Ver Distribución y Planos":
            go("Planos")

# ---------------------------------------------------------
# ADMIN (LOGIN + PANEL)
# ---------------------------------------------------------
def _validate_admin_login(email: str, password: str) -> bool:
    """
    Valida credenciales usando tu módulo modules.auth.
    Soporta que get_admin_credentials(conn) devuelva dict o tupla/lista.
    """
    try:
        creds = get_admin_credentials(conn)
    except Exception:
        creds = None

    if not creds:
        return True

    e0, p0 = "", ""

    if isinstance(creds, dict):
        e0 = (creds.get("email") or creds.get("admin_email") or "").strip().lower()
        p0 = (creds.get("password") or creds.get("admin_password") or "").strip()
    elif isinstance(creds, (tuple, list)) and len(creds) >= 2:
        e0 = str(creds[0] or "").strip().lower()
        p0 = str(creds[1] or "").strip()
    else:
        return True

    if not e0 or not p0:
        return True

    return (email.strip().lower() == e0) and (password == p0)

def admin_panel(conn):
    st.subheader("Administrador")

    top = st.columns([1, 1], vertical_alignment="center")
    with top[0]:
        st.caption("Sesión de administrador activa.")
    with top[1]:
        _, b = st.columns([1, 1])
        with b:
            if st.button("Cerrar sesión", key="ap_btn_admin_logout", use_container_width=True):
                admin_logout()

    tabs = st.tabs(["Cargar Datos"])

    with tabs[0]:
        st.markdown("### Cargar Excel y generar distribución")
        st.caption("Tu motor seats espera hojas tipo: Equipos, Parámetros y Capacidades (nombres pueden variar).")

        up = st.file_uploader("Subir archivo Excel", type=["xlsx", "xls"], key="ap_admin_excel_upload")

        colA, colB = st.columns([1, 1], vertical_alignment="center")
        with colA:
            cupos_reserva = st.number_input(
                "Cupos libres (reserva diaria)",
                min_value=0, max_value=50, value=2, step=1,
                key="ap_cupos_reserva"
            )
        with colB:
            ignore_params = st.toggle(
                "Ignorar parámetros (solo reparto proporcional)",
                value=False,
                key="ap_ignore_params"
            )

        st.session_state.setdefault("variant_seed", 42)

        seed_enabled = st.toggle("Usar seed", value=False, key="ap_seed_enabled")
        if seed_enabled:
            st.session_state["variant_seed"] = st.number_input(
                "Seed",
                min_value=0, max_value=10_000_000,
                value=int(st.session_state.get("variant_seed", 42)),
                step=1,
                key="ap_seed_value"
            )

        # Estado interno: vista previa (no guarda en DB hasta apretar "Guardar Distribución")
        st.session_state.setdefault("pending_distribution_rows", [])
        st.session_state.setdefault("pending_distribution_deficit", [])
        st.session_state.setdefault("pending_distribution_audit", {})
        st.session_state.setdefault("pending_distribution_score", {})

        def _run_generation(df_equipos, df_param, df_cap) -> bool:
            if df_equipos is None or df_equipos.empty:
                st.error("Falta hoja Equipos (o está vacía).")
                return False

            _df_param = df_param if df_param is not None else pd.DataFrame()
            _df_cap = df_cap if df_cap is not None else pd.DataFrame()

            if not bool(ignore_params):
                variants = compute_distribution_variants(
                    equipos_df=df_equipos,
                    parametros_df=_df_param,
                    df_capacidades=_df_cap,
                    cupos_reserva=int(cupos_reserva),
                    ignore_params=False,
                    n_variants=10,
                    variant_seed=int(st.session_state.get("variant_seed", 42)),
                    variant_mode="holgura",
                )
                best = variants[0] if variants else None
                if not best or not best.get("rows"):
                    st.error("No se generaron filas. Revisa que el Excel tenga columnas clave.")
                    return False
                rows = best["rows"]
                deficit_report = best.get("deficit_report", [])
                audit = best.get("audit", {})
                score_obj = best.get("score", {})
            else:
                rows, deficit_report, audit, score_obj = compute_distribution_from_excel(
                    equipos_df=df_equipos,
                    parametros_df=_df_param,
                    df_capacidades=_df_cap,
                    cupos_reserva=int(cupos_reserva),
                    ignore_params=True,
                    variant_seed=int(st.session_state.get("variant_seed", 42)) if seed_enabled else None,
                    variant_mode="holgura",
                )
                if not rows:
                    st.error("No se generaron filas (rows vacías). Revisa que el Excel tenga columnas clave.")
                    return False

            st.session_state["pending_distribution_rows"] = rows
            st.session_state["pending_distribution_deficit"] = deficit_report
            st.session_state["pending_distribution_audit"] = audit
            st.session_state["pending_distribution_score"] = score_obj
            return True

        if up is not None:
            try:
                xls = pd.ExcelFile(up)
                sheets = {name: xls.parse(name) for name in xls.sheet_names}

                st.success(f"✅ Archivo leído. Hojas: {', '.join(list(sheets.keys()))}")

                df_equipos = _safe_sheet_lookup(sheets, ["equipos", "equipo"])
                df_param = _safe_sheet_lookup(sheets, ["parametros", "parámetros", "parametro", "parámetro"])
                df_cap = _safe_sheet_lookup(sheets, ["capacidades", "capacidad", "aforo", "cupos"])

                # --- Botones Generar / Regenerar / Guardar ---
                b1, b2, b3 = st.columns([1, 1, 1], vertical_alignment="center")
                gen = b1.button("Generar distribución", type="primary", key="ap_btn_gen_dist")
                regen = b2.button("Regenerar", key="ap_btn_regen_dist")
                save_btn = b3.button("Guardar Distribución", key="ap_btn_save_dist")

                if gen:
                    ok = _run_generation(df_equipos, df_param, df_cap)
                    if ok:
                        st.rerun()

                if regen:
                    st.session_state["variant_seed"] = int(st.session_state.get("variant_seed", 42)) + 1
                    ok = _run_generation(df_equipos, df_param, df_cap)
                    if ok:
                        st.rerun()

                if save_btn:
                    rows = st.session_state.get("pending_distribution_rows", [])
                    if not rows:
                        st.warning("Primero genera una distribución para poder guardarla.")
                    else:
                        try:
                            clear_distribution(conn)
                            for r in rows:
                                piso_db = _piso_to_label(r.get("piso"))
                                dia_db = str(r.get("dia", "")).strip()
                                equipo_db = str(r.get("equipo", "")).strip()
                                cupos_db = int(float(r.get("cupos", 0) or 0))

                                insert_distribution(
                                    conn,
                                    piso_db,
                                    dia_db,
                                    equipo_db,
                                    cupos_db,
                                    r.get("% uso diario", None)
                                )
                            st.success("✅ Distribución guardada en Google Sheets (DB).")
                            st.session_state["last_distribution_rows"] = rows
                            st.session_state["last_distribution_deficit"] = st.session_state.get("pending_distribution_deficit", [])
                            st.session_state["last_distribution_audit"] = st.session_state.get("pending_distribution_audit", {})
                            st.session_state["last_distribution_score"] = st.session_state.get("pending_distribution_score", {})
                        except Exception as e:
                            st.error(f"No pude guardar en DB: {e}")
                            return

                # VISTA PREVIA (sin mostrar score)
                rows = st.session_state.get("pending_distribution_rows", [])
                deficit_report = st.session_state.get("pending_distribution_deficit", [])

                if rows:
                    df_out = pd.DataFrame(rows)

                    if "equipo" in df_out.columns:
                        df_out = df_out[df_out["equipo"].astype(str).str.strip().str.lower() != "cupos libres"].copy()

                    df_out["cupos"] = pd.to_numeric(df_out.get("cupos"), errors="coerce").fillna(0).astype(int)
                    df_out["dotacion"] = pd.to_numeric(df_out.get("dotacion"), errors="coerce")
                    df_out["piso"] = df_out["piso"].astype(str)
                    df_out["dia"] = df_out["dia"].astype(str)

                    # Deficit diario (viene desde seats ahora vs mínimos)
                    df_def = pd.DataFrame(deficit_report) if deficit_report else pd.DataFrame()
                    if not df_def.empty and {"piso", "equipo", "dia", "deficit"}.issubset(df_def.columns):
                        df_def2 = df_def.groupby(["piso", "equipo", "dia"], as_index=False)["deficit"].sum()
                        df_def2.rename(columns={"deficit": "Deficit"}, inplace=True)
                        df_out = df_out.merge(df_def2, on=["piso", "equipo", "dia"], how="left")

                    df_out.rename(columns={
                        "piso": "Piso",
                        "equipo": "Equipo",
                        "dotacion": "Personas",
                        "dia": "Días",
                        "cupos": "Cupos Diarios",
                        "% uso diario": "%Uso Diario",
                        "% uso semanal": "%Uso semanal",
                    }, inplace=True)

                    order_map = {d: i for i, d in enumerate(ORDER_DIAS)}
                    df_out["_ord_dia"] = df_out["Días"].map(order_map).fillna(999).astype(int)
                    df_out["_ord_piso"] = df_out["Piso"].str.extract(r"(\d+)")[0].fillna("9999").astype(int)

                    base_cols = ["Piso", "Equipo", "Personas", "Días", "Cupos Diarios", "%Uso Diario", "%Uso semanal"]
                    show_def = (
                        "Deficit" in df_out.columns
                        and not df_out["Deficit"].isna().all()
                        and (pd.to_numeric(df_out["Deficit"], errors="coerce").fillna(0) != 0).any()
                    )
                    if show_def:
                        df_out["Deficit"] = pd.to_numeric(df_out["Deficit"], errors="coerce").fillna(0).astype(int)
                        base_cols.append("Deficit")

                    st.markdown("### Vista previa (Saint-Laguë semanal → diario)")
                    with st.expander("Ver detalle de la distribución (por piso y día)", expanded=False):
                        st.dataframe(
                            df_out.sort_values(["_ord_piso", "_ord_dia", "Equipo"])[base_cols],
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as e:
                st.error(f"No se pudo leer el Excel: {e}")

def screen_admin(conn):
    if st.session_state.get("is_admin"):
        admin_panel(conn)
        return

    st.subheader("Administrador")
    st.session_state.setdefault("forgot_mode", False)

    if not st.session_state["forgot_mode"]:
        st.text_input("Ingresar correo", key="adm_login_email")
        st.text_input("Contraseña", type="password", key="adm_login_pass")

        c1, c2 = st.columns([1, 1], vertical_alignment="center")

        with c1:
            if st.button("Olvidaste tu contraseña", key="adm_btn_forgot"):
                st.session_state["forgot_mode"] = True
                st.rerun()

        with c2:
            _, btn_col = st.columns([1, 1], vertical_alignment="center")
            with btn_col:
                if st.button("Acceder", type="primary", key="adm_btn_login", use_container_width=True):
                    e = st.session_state.get("adm_login_email", "").strip()
                    p = st.session_state.get("adm_login_pass", "")
                    if not e or not p:
                        st.warning("Completa correo y contraseña.")
                    else:
                        ok = _validate_admin_login(e, p)
                        if ok:
                            st.session_state["is_admin"] = True
                            st.success("✅ Acceso concedido.")
                            st.rerun()
                        else:
                            st.error("❌ Credenciales incorrectas.")

    else:
        st.text_input("Correo de acceso", key="adm_reset_email")
        st.caption("Ingresa el código recibido en tu correo.")
        st.text_input("Código", key="adm_reset_code")

        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
        with c1:
            if st.button("Volver a Acceso", key="adm_btn_back"):
                st.session_state["forgot_mode"] = False
                st.rerun()
        with c2:
            if st.button("Enviar código", type="primary", key="adm_btn_send_code"):
                e = st.session_state.get("adm_reset_email", "").strip()
                if not e:
                    st.warning("Ingresa tu correo.")
                else:
                    st.success("Código enviado (simulado).")
        with c3:
            if st.button("Validar código", type="primary", key="adm_btn_validate"):
                c = st.session_state.get("adm_reset_code", "").strip()
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
