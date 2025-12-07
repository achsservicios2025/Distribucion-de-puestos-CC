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
# 1) CONFIGURACIÓN STREAMLIT (una sola vez y arriba)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gestor de puestos y reservas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2) IMPORTS DE MÓDULOS
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
# 3) CONSTANTES / DIRECTORIOS
# ---------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

PLANOS_DIR = Path("modules/planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

for d in (PLANOS_DIR, DATA_DIR, COLORED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 4) ESTADO UI (editable a futuro desde Admin)
# ---------------------------------------------------------
st.session_state.setdefault("ui", {
    "app_title": "Gestor de puestos y reservas",
    "bg_color": "#ffffff",
    "logo_path": "assets/logo.png",
    "title_font_size": 24,  # requerido
})

# menú tipo drawer a la derecha
st.session_state.setdefault("menu_open", False)
st.session_state.setdefault("screen", "Inicio")
st.session_state.setdefault("screen_group", None)   # "Reservas" | "Planos" | None
st.session_state.setdefault("forgot_mode", False)

# tabs index state
st.session_state.setdefault("reservas_tab_idx", 0)
st.session_state.setdefault("planos_tab_idx", 0)

# ---------------------------------------------------------
# 4.5) DB + SETTINGS (ANTES DE DIBUJAR UI)
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
# (opcional) tamaño título configurable desde admin si existe en settings
try:
    st.session_state["ui"]["title_font_size"] = int(settings.get("title_font_size", st.session_state["ui"]["title_font_size"]))
except Exception:
    pass

# ---------------------------------------------------------
# 5) CSS (quita bloque blanco gigante + top spacing + menú derecha)
# ---------------------------------------------------------
st.markdown(f"""
<style>
/* Fondo y “cintillo” superior */
.stApp {{ background: {st.session_state.ui["bg_color"]}; }}
header {{ visibility: hidden; height:0px; }}
/* quitar espacios top típicos que generan el “bloque blanco” */
div[data-testid="stAppViewContainer"] > .main {{
  padding-top: 0rem !important;
}}
section.main > div {{
  padding-top: 0rem !important;
}}
/* Ajuste contenedor */
.block-container {{
  max-width: 100% !important;
  padding-top: 0.5rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}}

/* Topbar */
.mk-topbar {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 16px;
  margin-top: 6px;
}}
.mk-logo img {{
  border-radius: 14px;
}}
.mk-title {{
  flex: 1;
  text-align:center;
  font-weight: 800;
  margin: 0;
  line-height: 1.1;
}}
.mk-menu-area {{
  display:flex;
  justify-content:flex-end;
  align-items:center;
  gap: 10px;
}}
.mk-menu-label {{
  font-size: 14px;
  font-weight: 800;
  color: rgba(0,0,0,0.65);
}}
.mk-btn-compact button {{
  border-radius: 14px !important;
  padding: 6px 12px !important;
  font-weight: 800 !important;
}}
/* Drawer */
.mk-drawer {{
  background: #f3f5f7;
  border-radius: 18px;
  padding: 12px 12px 14px 12px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.12);
  margin-top: 10px;
}}
.mk-drawer .stButton button {{
  width: 100% !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}}
.mk-drawer hr {{
  margin: 10px 0;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helpers utilitarios (de tu código)
# ---------------------------------------------------------
import uuid

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

def sort_floors(floor_list):
    def extract_num(text):
        text = str(text)
        num = re.findall(r"\d+", text)
        return int(num[0]) if num else 0
    return sorted(list(floor_list), key=extract_num)

def apply_sorting_to_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    col_dia = cols_lower.get("dia") or cols_lower.get("día")
    col_piso = cols_lower.get("piso")

    if col_dia:
        df[col_dia] = pd.Categorical(df[col_dia], categories=ORDER_DIAS, ordered=True)

    if col_piso:
        unique_floors = [str(x) for x in df[col_piso].dropna().unique()]
        sorted_floors = sort_floors(unique_floors)
        df[col_piso] = pd.Categorical(df[col_piso], categories=sorted_floors, ordered=True)

    sort_cols = [c for c in (col_piso, col_dia) if c]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df

# ---------------------------------------------------------
# NAV HELPERS (nuevo)
# ---------------------------------------------------------
def go_inicio():
    st.session_state["screen"] = "Inicio"
    st.session_state["screen_group"] = None

def go_admin():
    st.session_state["screen"] = "Administrador"
    st.session_state["screen_group"] = None

def go_reservas():
    st.session_state["screen"] = "Reservas"
    st.session_state["screen_group"] = "Reservas"

def go_planos():
    st.session_state["screen"] = "Planos"
    st.session_state["screen_group"] = "Planos"

def toggle_menu():
    st.session_state["menu_open"] = not st.session_state["menu_open"]

# ---------------------------------------------------------
# UI: TOPBAR (logo izq, título centro 24px, menú derecha)
# ---------------------------------------------------------
def render_topbar():
    # usamos columns para poder poner “Menú + <<” a la derecha
    c1, c2, c3 = st.columns([1.2, 3.6, 1.2], vertical_alignment="center")

    with c1:
        logo_path = Path(st.session_state.ui["logo_path"])
        if logo_path.exists():
            # "3*5 aprox": en Streamlit manejamos en px; dejo width ~150 (aprox 3-5cm visual en pantalla)
            st.image(str(logo_path), width=150)
        else:
            st.write("🧩 (Logo aquí)")

    with c2:
        size = int(st.session_state.ui.get("title_font_size", 24))
        title = st.session_state.ui.get("app_title", "Gestor de puestos y reservas")
        st.markdown(
            f"<div class='mk-title' style='font-size:{size}px;'>{title}</div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            "<div class='mk-menu-area'>"
            "<div class='mk-menu-label'>Menú</div>"
            "</div>",
            unsafe_allow_html=True
        )
        # botón << tal cual (y cuando esté abierto mostramos >> para cerrar)
        btn_label = "<<" if not st.session_state["menu_open"] else ">>"
        st.markdown("<div class='mk-btn-compact'>", unsafe_allow_html=True)
        if st.button(btn_label, key="btn_menu_toggle"):
            toggle_menu()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DRAWER MENÚ (aparece bajo “Menú” a la derecha)
# ---------------------------------------------------------
def render_menu_drawer():
    if not st.session_state["menu_open"]:
        return

    # lo alineamos a la derecha usando columns
    _, _, c = st.columns([2.2, 2.2, 1.6])
    with c:
        st.markdown("<div class='mk-drawer'>", unsafe_allow_html=True)

        st.markdown("**Opciones**")
        if st.button("Reservas", key="drawer_reservas"):
            go_reservas()
            st.session_state["menu_open"] = False
            st.rerun()

        if st.button("Ver Distribución y Planos", key="drawer_planos"):
            go_planos()
            st.session_state["menu_open"] = False
            st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Mantener acceso Administrador en inicio (NO en selectbox)
        if st.button("Administrador", key="drawer_admin"):
            go_admin()
            st.session_state["menu_open"] = False
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PANTALLAS / VISTAS
# ---------------------------------------------------------
def screen_inicio():
    st.info("Usa **Menú >>** (a la derecha) para acceder a Reservas o Ver Distribución y Planos.")
    st.write("")

def screen_admin(conn):
    st.subheader("Administrador")
    st.session_state.setdefault("forgot_mode", False)

    if not st.session_state["forgot_mode"]:
        email = st.text_input("Ingresar correo", key="admin_login_email")
        password = st.text_input("Contraseña", type="password", key="admin_login_pass")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Olvidaste tu contraseña", key="btn_admin_forgot"):
                st.session_state["forgot_mode"] = True
                st.rerun()

        with c2:
            if st.button("Acceder", type="primary", key="btn_admin_login"):
                if not email or not password:
                    st.warning("Completa correo y contraseña.")
                else:
                    st.success("Login recibido (validación real pendiente).")
    else:
        reset_email = st.text_input("Correo de acceso", key="admin_reset_email")
        if st.button("Enviar código", key="btn_admin_send_code"):
            if not reset_email:
                st.warning("Ingresa tu correo.")
            else:
                st.success("Código enviado (simulado).")

        st.caption("Ingresa el código recibido en tu correo.")
        code = st.text_input("Código", key="admin_reset_code")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Validar código", type="primary", key="btn_admin_validate"):
                if not code:
                    st.warning("Ingresa el código.")
                else:
                    st.success("Código validado (simulado).")

        with c2:
            if st.button("Volver a Acceso", key="btn_admin_back"):
                st.session_state["forgot_mode"] = False
                st.rerun()

# ---- (Tus pantallas existentes, tal como pediste mantener lógica) ----
def screen_reservar_puesto_flex(conn):
    import datetime as _dt

    st.header("Reservar Puesto Flex")
    st.info("Reserva de 'Cupos libres' (Máximo 2 días por mes).")

    df = read_distribution_df(conn)

    if df.empty:
        st.warning("⚠️ No hay configuración de distribución cargada en el sistema.")
        return

    c1, c2 = st.columns(2)
    fe = c1.date_input("Selecciona Fecha", min_value=_dt.date.today(), key="fp")
    pisos_disp = sort_floors(df["piso"].unique())
    pi = c2.selectbox("Selecciona Piso", pisos_disp, key="pp")

    dn = ORDER_DIAS[fe.weekday()] if fe.weekday() < 5 else "FinDeSemana"

    if dn == "FinDeSemana":
        st.error("🔒 Es fin de semana. No se pueden realizar reservas.")
        return

    sub_libres = df[
        (df["piso"].astype(str) == str(pi)) &
        (df["dia"].astype(str) == str(dn)) &
        (df["equipo"].astype(str).str.strip().str.lower() == "cupos libres")
    ]

    total_cupos = int(pd.to_numeric(sub_libres["cupos"], errors="coerce").fillna(0).sum()) if not sub_libres.empty else 0

    all_res = list_reservations_df(conn)
    if all_res is None or all_res.empty:
        ocupados = 0
    else:
        def _norm_piso_local(x):
            s = str(x).strip()
            if s.lower().startswith("piso piso"):
                s = s[5:].strip()
            if not s.lower().startswith("piso"):
                s = f"Piso {s}"
            return s

        mask = (
            all_res["reservation_date"].astype(str) == str(fe)
        ) & (
            all_res["piso"].astype(str).map(_norm_piso_local) == _norm_piso_local(pi)
        )
        ocupados = int(mask.sum())

    disponibles = max(0, total_cupos - ocupados)

    if disponibles > 0:
        st.success(f"✅ **Hay cupo: Quedan {disponibles} puestos disponibles**")
    else:
        st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")

    st.markdown("### Datos del Solicitante")

    equipos_disponibles = sorted(df[df["piso"] == pi]["equipo"].unique().tolist())
    equipos_disponibles = [e for e in equipos_disponibles if e != "Cupos libres"]

    with st.form("form_puesto"):
        cf1, cf2 = st.columns(2)
        nombre = cf1.text_input("Nombre")
        area_equipo = cf1.selectbox("Área / Equipo", equipos_disponibles if equipos_disponibles else ["General"])
        em = cf2.text_input("Correo Electrónico")

        submitted = st.form_submit_button(
            "Confirmar Reserva",
            type="primary",
            disabled=(disponibles <= 0)
        )

        if submitted:
            if not nombre.strip():
                st.error("Por favor ingresa tu nombre.")
            if not em:
                st.error("Por favor ingresa tu correo electrónico.")
            elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', em):
                st.error("Por favor ingresa un correo electrónico válido (ejemplo: usuario@ejemplo.com).")
            elif user_has_reservation(conn, em, str(fe)):
                st.error("Ya tienes una reserva registrada para esta fecha.")
            elif count_monthly_free_spots(conn, em, fe) >= 2:
                st.error("Has alcanzado el límite de 2 reservas mensuales.")
            elif disponibles <= 0:
                st.error("Lo sentimos, el cupo se acaba de agotar.")
            else:
                add_reservation(
                    conn,
                    area_equipo,
                    em,
                    pi,
                    str(fe),
                    "Cupos libres",
                    _dt.datetime.now(_dt.timezone.utc).isoformat()
                )
                msg = (
                    f"✅ Reserva Confirmada:\n\n"
                    f"- Área/Equipo: {area_equipo}\n"
                    f"- Fecha: {fe}\n"
                    f"- Piso: {pi}\n"
                    f"- Tipo: Puesto Flex"
                )
                st.success(msg)

                email_sent = send_reservation_email(em, "Confirmación Puesto", msg.replace("\n", "<br>"))
                if email_sent:
                    st.info("📧 Correo de confirmación enviado")
                else:
                    st.warning("⚠️ No se pudo enviar el correo. Verifica la configuración SMTP.")

                st.rerun()

def screen_reservar_sala(conn):
    import datetime as _dt

    st.header("Reservar Sala de Reuniones")
    st.info("Selecciona tu equipo/área y luego elige la sala y horario disponible")

    df_dist = read_distribution_df(conn)
    equipos_lista = []
    if not df_dist.empty:
        equipos_lista = sorted([e for e in df_dist["equipo"].unique() if e != "Cupos libres"])

    if not equipos_lista:
        st.warning("⚠️ No hay equipos configurados. Contacta al administrador.")
        return

    st.markdown("### Selecciona tu Equipo/Área")
    equipo_seleccionado = st.selectbox("Equipo/Área", equipos_lista, key="equipo_sala_sel")

    st.markdown("---")
    st.markdown("### Selecciona Sala y Horario")

    salas_disponibles = [
        "Sala Reuniones Pequeña Piso 1",
        "Sala Reuniones Grande Piso 1",
        "Sala Reuniones Piso 2",
        "Sala Reuniones Piso 3",
    ]

    c_sala, c_fecha = st.columns(2)
    sl = c_sala.selectbox("Selecciona Sala", salas_disponibles, key="sala_sel")

    if "Piso 1" in sl:
        pi_s = "Piso 1"
    elif "Piso 2" in sl:
        pi_s = "Piso 2"
    elif "Piso 3" in sl:
        pi_s = "Piso 3"
    else:
        pi_s = "Piso 1"

    fe_s = c_fecha.date_input("Fecha", min_value=_dt.date.today(), key="fs_sala")

    df_reservas_sala = get_room_reservations_df(conn)
    reservas_hoy = []
    if df_reservas_sala is not None and not df_reservas_sala.empty:
        mask = (df_reservas_sala["reservation_date"].astype(str) == str(fe_s)) & (df_reservas_sala["room_name"] == sl)
        reservas_hoy = df_reservas_sala[mask].to_dict("records")

    st.markdown("#### Horarios Disponibles")

    if reservas_hoy:
        horarios_ocupados = ", ".join([f"{r.get('start_time','')} - {r.get('end_time','')}" for r in reservas_hoy])
        st.warning(f"⚠️ Horarios ocupados: {horarios_ocupados}")

    slots_completos = generate_time_slots("08:00", "20:00", 15)
    if len(slots_completos) < 2:
        st.error("No hay horarios configurados para esta sala.")
        return

    opciones_inicio = slots_completos[:-1]
    inicio_key = f"inicio_sala_{sl}"
    fin_key = f"fin_sala_{sl}"

    hora_inicio_sel = st.selectbox("Hora de inicio", opciones_inicio, index=0, key=inicio_key)

    pos_inicio = slots_completos.index(hora_inicio_sel)
    opciones_fin = slots_completos[pos_inicio + 1:]

    if not opciones_fin:
        st.warning("Selecciona una hora de inicio anterior a las 20:00 hrs.")
        return

    idx_fin = min(4, len(opciones_fin) - 1)
    hora_fin_sel = st.selectbox("Hora de término", opciones_fin, index=idx_fin, key=fin_key)

    inicio_dt = _dt.datetime.strptime(hora_inicio_sel, "%H:%M")
    fin_dt = _dt.datetime.strptime(hora_fin_sel, "%H:%M")
    duracion_min = int((fin_dt - inicio_dt).total_seconds() / 60)

    if duracion_min < 15:
        st.error("El intervalo debe ser de al menos 15 minutos.")
        return

    conflicto_actual = check_room_conflict(reservas_hoy, str(fe_s), sl, hora_inicio_sel, hora_fin_sel)
    if conflicto_actual:
        st.error("❌ Ese intervalo ya está reservado. Elige otro horario.")

    st.markdown("---")
    st.markdown(f"### Confirmar Reserva: {hora_inicio_sel} - {hora_fin_sel}")

    with st.form("form_sala"):
        st.info(
            f"**Equipo/Área:** {equipo_seleccionado}\n\n"
            f"**Sala:** {sl}\n\n"
            f"**Fecha:** {fe_s}\n\n"
            f"**Horario:** {hora_inicio_sel} - {hora_fin_sel}"
        )
        e_s = st.text_input("Correo Electrónico", key="email_sala")
        sub_sala = st.form_submit_button("✅ Confirmar Reserva", type="primary", disabled=conflicto_actual)

    if sub_sala:
        if not e_s:
            st.error("Por favor ingresa tu correo electrónico.")
        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', e_s):
            st.error("Por favor ingresa un correo electrónico válido (ejemplo: usuario@ejemplo.com).")
        elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, hora_inicio_sel, hora_fin_sel):
            st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
        else:
            add_room_reservation(
                conn,
                equipo_seleccionado,
                e_s,
                pi_s,
                sl,
                str(fe_s),
                hora_inicio_sel,
                hora_fin_sel,
                _dt.datetime.now(_dt.timezone.utc).isoformat(),
            )

            msg = (
                f"✅ Sala Confirmada:\n\n"
                f"- Equipo/Área: {equipo_seleccionado}\n"
                f"- Sala: {sl}\n"
                f"- Fecha: {fe_s}\n"
                f"- Horario: {hora_inicio_sel} - {hora_fin_sel}"
            )
            st.success(msg)

            try:
                email_sent = send_reservation_email(e_s, "Reserva Sala", msg.replace("\n", "<br>"))
                if email_sent:
                    st.info("📧 Correo de confirmación enviado")
                else:
                    st.warning("⚠️ No se pudo enviar el correo. Verifica la configuración SMTP.")
            except Exception as email_error:
                st.warning(f"⚠️ Error al enviar correo: {email_error}")

            st.rerun()

def _fmt_item(item: dict) -> str:
    if item["kind"] == "puesto":
        return (f"📌 Puesto | {item['reservation_date']} | Piso: {item.get('piso','')} | "
                f"Área: {item.get('team_area','')} | Nombre: {item.get('user_name','')}")
    return (f"🏢 Sala | {item['reservation_date']} | {item.get('room_name','')} | "
            f"{item.get('start_time','')}–{item.get('end_time','')} | Equipo: {item.get('user_name','')}")

def open_confirm_delete_multi(selected_items: list[dict]):
    st.session_state["confirm_delete_multi"] = {"items": selected_items}
    st.rerun()

def render_confirm_delete_multi_dialog(conn):
    payload = st.session_state.get("confirm_delete_multi")
    if not payload:
        return

    items = payload.get("items") or []
    if not items:
        st.session_state.pop("confirm_delete_multi", None)
        return

    if not hasattr(st, "dialog"):
        st.error("Tu Streamlit no soporta popups centrados (st.dialog). Actualiza Streamlit.")
        return

    @st.dialog("Confirmar anulación")
    def _dlg():
        st.markdown("### ¿Anular las reservas seleccionadas?")
        st.caption("Se eliminarán únicamente las reservas marcadas. Las no marcadas se mantienen.")

        with st.expander("Ver detalle", expanded=True):
            for it in items:
                st.write("• " + _fmt_item(it))

        c1, c2 = st.columns(2)

        if c1.button("🔴 Sí, anular", type="primary", use_container_width=True, key="confirm_multi_yes"):
            deleted = 0
            for it in items:
                if it["kind"] == "puesto":
                    ok = delete_reservation_from_db(conn, it["user_email"], it["reservation_date"], it["team_area"])
                else:
                    inicio = str(it.get("start_time", "")).strip()
                    inicio = inicio[:5] if len(inicio) >= 5 else inicio
                    ok = delete_room_reservation_from_db(conn, it["user_email"], it["reservation_date"], it["room_name"], inicio)
                if ok:
                    deleted += 1

            st.session_state.pop("confirm_delete_multi", None)
            st.cache_data.clear()
            st.success(f"✅ Se anularon {deleted} reserva(s).")
            st.rerun()

        if c2.button("Cancelar", use_container_width=True, key="confirm_multi_no"):
            st.session_state.pop("confirm_delete_multi", None)
            st.rerun()

    _dlg()

def screen_mis_reservas(conn):
    st.header("Mis Reservas y Listados")

    ver_tipo = st.selectbox("Ver:", ["Reserva de Puestos", "Reserva de Salas"], key="mis_reservas_ver_tipo")

    df_puestos = list_reservations_df(conn) or pd.DataFrame()
    df_salas = get_room_reservations_df(conn) or pd.DataFrame()

    if ver_tipo == "Reserva de Puestos":
        if df_puestos.empty:
            st.info("No hay reservas de puestos registradas.")
        else:
            df_view = df_puestos.copy().rename(columns={
                "piso": "Piso",
                "user_name": "Nombre",
                "team_area": "Equipo",
                "user_email": "Correo",
                "reservation_date": "Fecha de Reserva",
            })
            cols = ["Piso", "Nombre", "Equipo", "Correo", "Fecha de Reserva"]
            st.dataframe(df_view[[c for c in cols if c in df_view.columns]], hide_index=True, use_container_width=True)
    else:
        if df_salas.empty:
            st.info("No hay reservas de salas registradas.")
        else:
            df_view = df_salas.copy().rename(columns={
                "room_name": "Sala",
                "user_name": "Equipo",
                "user_email": "Correo",
                "reservation_date": "Fecha de Reserva",
                "start_time": "Hora Inicio",
                "end_time": "Hora Fin",
            })
            cols = ["Sala", "Equipo", "Correo", "Fecha de Reserva", "Hora Inicio", "Hora Fin"]
            st.dataframe(df_view[[c for c in cols if c in df_view.columns]], hide_index=True, use_container_width=True)

    st.markdown("---")

    with st.expander("Anular Reservas", expanded=False):
        correo_buscar = st.text_input(
            "Correo asociado a la(s) reserva(s)",
            key="anular_correo",
            placeholder="usuario@ejemplo.com"
        ).strip()

        buscar = st.button("Buscar", type="primary", key="btn_buscar_reservas")

        st.session_state.setdefault("anular_candidates", [])
        st.session_state.setdefault("anular_checks", {})

        if buscar:
            st.session_state["anular_candidates"] = []
            st.session_state["anular_checks"] = {}

            if not correo_buscar:
                st.error("Ingresa un correo para buscar reservas.")
            elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', correo_buscar):
                st.error("Ingresa un correo válido.")
            else:
                correo_norm = correo_buscar.lower()
                candidates = []

                if not df_puestos.empty and "user_email" in df_puestos.columns:
                    hits = df_puestos[df_puestos["user_email"].astype(str).str.lower().str.strip() == correo_norm].copy()
                    for _, r in hits.iterrows():
                        candidates.append({
                            "kind": "puesto",
                            "user_email": str(r.get("user_email", "")).strip(),
                            "reservation_date": str(r.get("reservation_date", "")).strip(),
                            "team_area": str(r.get("team_area", "")).strip(),
                            "piso": str(r.get("piso", "")).strip(),
                            "user_name": str(r.get("user_name", "")).strip(),
                        })

                if not df_salas.empty and "user_email" in df_salas.columns:
                    hits = df_salas[df_salas["user_email"].astype(str).str.lower().str.strip() == correo_norm].copy()
                    for _, r in hits.iterrows():
                        inicio_raw = str(r.get("start_time", "")).strip()
                        inicio = inicio_raw[:5] if len(inicio_raw) >= 5 else inicio_raw
                        candidates.append({
                            "kind": "sala",
                            "user_email": str(r.get("user_email", "")).strip(),
                            "reservation_date": str(r.get("reservation_date", "")).strip(),
                            "room_name": str(r.get("room_name", "")).strip(),
                            "start_time": inicio,
                            "end_time": str(r.get("end_time", "")).strip(),
                            "piso": str(r.get("piso", "")).strip(),
                            "user_name": str(r.get("user_name", "")).strip(),
                        })

                st.session_state["anular_candidates"] = candidates
                st.session_state["anular_checks"] = {i: False for i in range(len(candidates))}

        candidates = st.session_state.get("anular_candidates") or []

        if candidates:
            st.markdown("### Marca las reservas que quieres anular")
            st.caption("Las que NO marques se mantienen.")

            b1, b2, _ = st.columns([1, 1, 2])
            with b1:
                if st.button("Marcar todas", key="btn_mark_all"):
                    st.session_state["anular_checks"] = {i: True for i in range(len(candidates))}
                    st.rerun()
            with b2:
                if st.button("Desmarcar todas", key="btn_unmark_all"):
                    st.session_state["anular_checks"] = {i: False for i in range(len(candidates))}
                    st.rerun()

            selected_items = []
            for i, item in enumerate(candidates):
                label = _fmt_item(item)
                current = bool(st.session_state["anular_checks"].get(i, False))
                new_val = st.checkbox(label, value=current, key=f"anular_ck_{i}")
                st.session_state["anular_checks"][i] = new_val
                if new_val:
                    selected_items.append(item)

            if st.button("Anular seleccionadas", type="primary", key="btn_anular_multi"):
                if not selected_items:
                    st.warning("Marca al menos una reserva.")
                else:
                    open_confirm_delete_multi(selected_items)

        elif correo_buscar:
            st.info("No se encontraron reservas para ese correo.")

    render_confirm_delete_multi_dialog(conn)

# ---------------------------------------------------------
# NUEVO: VISTA "Reservas" con 3 pestañas (sin selectbox)
# ---------------------------------------------------------
def screen_reservas_tabs(conn):
    st.subheader("Reservas")

    tabs = st.tabs(["Reservar Puesto Flex", "Reserva Salas de Reuniones", "Mis Reservas y Listados"])

    with tabs[0]:
        screen_reservar_puesto_flex(conn)

    with tabs[1]:
        screen_reservar_sala(conn)

    with tabs[2]:
        screen_mis_reservas(conn)

# ---------------------------------------------------------
# NUEVO: "Ver Distribución y Planos" (descarga-only)
#   - 2 pestañas: Distribución / Planos
#   - sin opciones admin (solo descarga xlsx/pdf/imagen)
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
            df_show = apply_sorting_to_df(df)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            xlsx_bytes = _df_to_xlsx_bytes(df_show, sheet_name="distribucion")
            st.download_button(
                "⬇️ Descargar Distribución (XLSX)",
                data=xlsx_bytes,
                file_name="distribucion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # PDF simple con tabla (si ya tienes un reporte mejor, puedes enchufarlo aquí)
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(True, 15)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, clean_pdf_text("Distribución"), ln=True, align="C")
                pdf.ln(4)

                pdf.set_font("Arial", "B", 8)
                cols = list(df_show.columns)
                widths = [max(18, min(45, int(190 / max(1, len(cols))))) for _ in cols]

                for w, c in zip(widths, cols):
                    pdf.cell(w, 6, clean_pdf_text(str(c))[:18], border=1)
                pdf.ln()

                pdf.set_font("Arial", "", 8)
                for _, r in df_show.iterrows():
                    for w, c in zip(widths, cols):
                        pdf.cell(w, 6, clean_pdf_text(str(r.get(c, "")))[:18], border=1)
                    pdf.ln()
                    if pdf.get_y() > 265:
                        pdf.add_page()

                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    "⬇️ Descargar Distribución (PDF)",
                    data=pdf_bytes,
                    file_name="distribucion.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.info(f"No se pudo generar PDF aquí (puedes mantener tu generador actual): {e}")

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

            # PDF wrapper del plano
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(True, 15)
                # ajustar al ancho
                pdf.image(str(img_path), x=10, y=15, w=190)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    "⬇️ Descargar plano (PDF)",
                    data=pdf_bytes,
                    file_name=img_path.stem + ".pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
render_topbar()
render_menu_drawer()
st.divider()

screen = st.session_state.get("screen", "Inicio")

if screen == "Inicio":
    screen_inicio()
elif screen == "Administrador":
    screen_admin(conn)
elif screen == "Reservas":
    screen_reservas_tabs(conn)
elif screen == "Planos":
    screen_descargas_distribucion_planos(conn)
else:
    screen_inicio()

# ---------------------------------------------------------
# (Todo lo demás de tu app original: helpers grandes/funciones de distribución/PDF avanzados)
#   - Lo dejo intacto abajo para que no pierdas nada.
#   - Si ya lo tienes en el archivo, puedes mantenerlo tal cual.
# ---------------------------------------------------------

def _get_team_and_dotacion_cols(df_eq: pd.DataFrame):
    if df_eq is None or df_eq.empty:
        return None, None
    cols = list(df_eq.columns)
    col_team = next((c for c in cols if "equipo" in c.lower()), None)
    col_dot = next((c for c in cols if "personas" in c.lower()), None)
    if not col_dot:
        col_dot = next((c for c in cols if "dot" in c.lower()), None)
    return col_team, col_dot

def _equity_score(rows, deficit, dot_map: dict, days_per_week=5):
    assigned = {}
    for r in rows or []:
        eq = str(r.get("equipo", "")).strip()
        if not eq or eq.lower() == "cupos libres":
            continue
        try:
            cup = int(float(str(r.get("cupos", 0)).replace(",", ".")))
        except Exception:
            cup = 0
        assigned[eq] = assigned.get(eq, 0) + cup

    coverages = []
    for eq, people in dot_map.items():
        needed = float(people) * float(days_per_week)
        if needed <= 0:
            continue
        coverages.append(assigned.get(eq, 0) / needed)

    if not coverages:
        return 9999.0 + (len(deficit) if deficit else 0)

    coverages = np.array(coverages, dtype=float)
    std = float(np.std(coverages))
    rng = float(np.max(coverages) - np.min(coverages))
    conflicts = float(len(deficit) if deficit else 0)
    return (1.0 * std) + (0.6 * rng) + (0.2 * conflicts)

def _dot_map_from_equipos(df_eq: pd.DataFrame) -> dict:
    if df_eq is None or df_eq.empty:
        return {}
    col_team = next((c for c in df_eq.columns if "equipo" in c.lower()), None)
    col_dot = None
    for key in ["personas", "dotacion", "dotación", "dot"]:
        col_dot = next((c for c in df_eq.columns if key in c.lower()), None)
        if col_dot:
            break
    if not col_team or not col_dot:
        return {}

    out = {}
    for _, r in df_eq.iterrows():
        team = str(r.get(col_team, "")).strip()
        if not team or team.lower() == "cupos libres":
            continue
        try:
            val = int(float(str(r.get(col_dot, 0)).replace(",", ".")))
        except Exception:
            continue
        if val > 0:
            out[team] = val
    return out

def _cap_map_from_capacidades(df_cap: pd.DataFrame) -> dict:
    if df_cap is None or df_cap.empty:
        return {}

    col_piso = next((c for c in df_cap.columns if "piso" in c.lower()), None)
    col_dia = next((c for c in df_cap.columns if "dia" in c.lower() or "día" in c.lower()), None)
    col_cap = next((c for c in df_cap.columns if "cap" in c.lower() or "cupo" in c.lower()), None)

    if not col_piso or not col_cap:
        return {}

    out = {}
    for _, r in df_cap.iterrows():
        piso_raw = str(r.get(col_piso, "")).strip()
        if not piso_raw or piso_raw.lower() == "nan":
            continue
        piso = piso_raw if piso_raw.lower().startswith("piso") else f"Piso {piso_raw}"

        dia = str(r.get(col_dia, "")).strip() if col_dia else ""
        if dia.lower() == "nan":
            dia = ""

        try:
            cap = int(float(str(r.get(col_cap, 0)).replace(",", ".")))
        except Exception:
            continue

        if cap <= 0:
            continue

        out[(piso, dia)] = cap

    return out

def _min_daily_for_team(dotacion: int, factor: float = 1.0) -> int:
    if dotacion >= 13: base = 6
    elif dotacion >= 8: base = 4
    elif dotacion >= 5: base = 3
    elif dotacion >= 3: base = 2
    else: base = 0
    return int(round(base * factor))

def _largest_remainder_allocation(weights: dict, total: int) -> dict:
    if total <= 0 or not weights:
        return {k: 0 for k in weights.keys()}
    s = sum(max(0, int(v)) for v in weights.values())
    if s <= 0:
        return {k: 0 for k in weights.keys()}

    quotas = {k: (max(0, int(w)) / s) * total for k, w in weights.items()}
    base = {k: int(q) for k, q in quotas.items()}
    used = sum(base.values())
    remain = total - used

    frac = sorted(((k, quotas[k] - base[k]) for k in quotas), key=lambda x: x[1], reverse=True)
    i = 0
    while remain > 0 and i < len(frac):
        k = frac[i][0]
        base[k] += 1
        remain -= 1
        i += 1
        if i >= len(frac) and remain > 0:
            i = 0
    return base

def generate_distribution_math_correct(
    df_eq: pd.DataFrame,
    df_cap: pd.DataFrame,
    cupos_libres_diarios: int = 2,
    min_dotacion_para_garantia: int = 3,
    min_factor: float = 1.0
):
    dot = _dot_map_from_equipos(df_eq)
    cap_map = _cap_map_from_capacidades(df_cap)
    pisos = sorted({p for (p, _) in cap_map.keys()} or {"Piso 1"})

    rows = []
    deficits = []

    for piso in pisos:
        for dia in ORDER_DIAS:
            cap = cap_map.get((piso, dia)) or cap_map.get((piso, "")) or 0
            if cap <= 0:
                deficits.append({"piso": piso, "dia": dia, "causa": "Sin capacidad definida en hoja Capacidades"})
                continue

            libres = min(cupos_libres_diarios, cap)
            cap_rest = cap - libres

            teams = {k: v for k, v in dot.items() if int(v) >= min_dotacion_para_garantia}
            mins = {t: _min_daily_for_team(int(v), factor=min_factor) for t, v in teams.items()}
            sum_mins = sum(mins.values())

            if sum_mins > cap_rest:
                deficits.append({
                    "piso": piso, "dia": dia,
                    "causa": f"Capacidad insuficiente: mínimos({sum_mins}) > cap_restante({cap_rest})"
                })
                alloc = _largest_remainder_allocation(mins, cap_rest)
            else:
                alloc = mins.copy()
                extra = cap_rest - sum_mins
                extra_alloc = _largest_remainder_allocation(teams, extra)
                for t, v in extra_alloc.items():
                    alloc[t] = alloc.get(t, 0) + v

            total_asignado = 0
            for t, c in alloc.items():
                if c <= 0:
                    continue
                total_asignado += c
                rows.append({"piso": piso, "dia": dia, "equipo": t, "cupos": int(c)})

            rows.append({"piso": piso, "dia": dia, "equipo": "Cupos libres", "cupos": int(libres)})

            if total_asignado + libres != cap:
                deficits.append({
                    "piso": piso, "dia": dia,
                    "causa": f"Mismatch: asignado({total_asignado + libres}) != capacidad({cap})"
                })

    return rows, deficits

def filter_minimum_deficits(deficit_list):
    filtered = []
    for item in deficit_list or []:
        try:
            minimo = int(float(str(item.get("minimo", 0)).strip()))
            asignado = int(float(str(item.get("asignado", 0)).strip()))
        except (TypeError, ValueError):
            continue
        deficit_val = max(0, minimo - asignado)
        if deficit_val > 0:
            fixed = dict(item)
            fixed["minimo"] = minimo
            fixed["asignado"] = asignado
            fixed["deficit"] = deficit_val
            fixed["causa"] = f"Faltan {deficit_val} puestos (capacidad insuficiente)"
            filtered.append(fixed)
    return filtered

def infer_team_dotacion_map(df):
    if df is None or df.empty:
        return {}

    cols_lower = {c.lower(): c for c in df.columns}
    col_equipo = None
    for key, col in cols_lower.items():
        if "equipo" in key:
            col_equipo = col
            break
    if not col_equipo:
        return {}

    col_dot = next((col for key, col in cols_lower.items() if "dotacion" in key or "dotación" in key), None)
    cupos_col = next((col for key, col in cols_lower.items() if "cupo" in key), None)

    pct_col = None
    for key, col in cols_lower.items():
        if "%distrib" in key or "pct" in key or "porcentaje" in key:
            pct_col = col
            break

    dot_map = {}

    if col_dot:
        series = df[[col_equipo, col_dot]].dropna()
        for _, row in series.iterrows():
            eq = str(row[col_equipo]).strip()
            if not eq or eq.lower().startswith("cupos libres"):
                continue
            try:
                dot = int(float(row[col_dot]))
            except (TypeError, ValueError):
                continue
            if dot > 0:
                dot_map.setdefault(eq, dot)

    if not dot_map and cupos_col and pct_col:
        temp = df[[col_equipo, cupos_col, pct_col]].dropna()
        for _, row in temp.iterrows():
            eq = str(row[col_equipo]).strip()
            if not eq or eq.lower().startswith("cupos libres"):
                continue
            try:
                cupos_val = float(str(row[cupos_col]).replace(",", "."))
                pct_val = float(str(row[pct_col]).replace("%", "").replace(",", "."))
            except (TypeError, ValueError):
                continue
            if pct_val <= 0:
                continue
            dot_est = int(round(cupos_val * 100 / pct_val))
            if dot_est > 0 and eq not in dot_map:
                dot_map[eq] = dot_est

    return dot_map

def hex_to_rgba(hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(0, 160, 74, {alpha})"

def create_merged_pdf(piso_sel, conn, global_logo_path):
    if piso_sel is None or (isinstance(piso_sel, float) and pd.isna(piso_sel)):
        piso_sel = "Piso 1"
    piso_sel = str(piso_sel)

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    found_any = False

    df = read_distribution_df(conn)
    base_config = st.session_state.get('last_style_config', {})

    for dia in ORDER_DIAS:
        subset = df[(df['piso'] == piso_sel) & (df['dia'] == dia)]
        current_seats = dict(zip(subset['equipo'], subset['cupos']))

        day_config = base_config.copy()
        if not day_config.get("subtitle_text"):
            day_config["subtitle_text"] = f"Día: {dia}"
        else:
            if "Día:" not in str(day_config.get("subtitle_text","")):
                day_config["subtitle_text"] = f"Día: {dia}"

        img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)

        if img_path and Path(img_path).exists():
            found_any = True
            pdf.add_page()
            try:
                pdf.image(str(img_path), x=10, y=10, w=190)
            except Exception:
                pass

    if not found_any:
        return None
    return pdf.output(dest='S').encode('latin-1')
