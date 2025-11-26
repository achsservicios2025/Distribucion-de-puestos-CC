import streamlit as st
import pandas as pd
import datetime
import os
import uuid
import json
import shutil
import re
from pathlib import Path
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image as PILImage
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
import base64 # Requerido para la solución robusta del st_canvas

# ---------------------------------------------------------
# 1. PARCHE (ELIMINADO Y REEMPLAZADO POR SOLUCIÓN DIRECTA)
# ---------------------------------------------------------
# Se elimina el parche complejo que falla por una solución directa de Base64 en la línea 702.
# ---------------------------------------------------------

# ---------------------------------------------------------
# 2. IMPORTACIONES DE MÓDULOS (Consolidadas)
# ---------------------------------------------------------
from modules.database import (
    get_conn, init_db, insert_distribution, clear_distribution,
    read_distribution_df, save_setting, get_all_settings,
    add_reservation, user_has_reservation, list_reservations_df,
    add_room_reservation, get_room_reservations_df,
    count_monthly_free_spots, delete_reservation_from_db, 
    delete_room_reservation_from_db, perform_granular_delete,
    ensure_reset_table, save_reset_token, validate_and_consume_token
)
from modules.auth import get_admin_credentials
from modules.layout import admin_appearance_ui, apply_appearance_styles
from modules.seats import compute_distribution_from_excel
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones, create_header_image 
from modules.pdfgen import create_merged_pdf, generate_full_pdf, sort_floors, apply_sorting_to_df, clean_pdf_text
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# 3. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

# (Verificación de secretos y conexión, sin cambios)
if "gcp_service_account" not in st.secrets:
    st.error("🚨 ERROR CRÍTICO: No se encuentran los secretos [gcp_service_account]. Revisa el formato TOML en Streamlit Cloud.")
    st.stop()

# Intento de conexión
try:
    creds_dict = dict(st.secrets["gcp_service_account"])
    pk = creds_dict.get("private_key", "")
    if "-----BEGIN PRIVATE KEY-----" not in pk:
        st.error("🚨 ERROR EN PRIVATE KEY: No parece una llave válida.")
        st.stop()
        
    from google.oauth2.service_account import Credentials
    import gspread
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet_name = st.secrets["sheets"]["sheet_name"]
    sh = client.open(sheet_name)
except Exception as e:
    st.error(f"🔥 LA CONEXIÓN FALLÓ AQUÍ: {str(e)}")
    st.stop()

# ----------------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
PLANOS_DIR = Path("planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

DATA_DIR.mkdir(exist_ok=True)
PLANOS_DIR.mkdir(exist_ok=True)
COLORED_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 4. FUNCIONES HELPER & LÓGICA
# ---------------------------------------------------------

def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    """
    Genera una propuesta basada en una estrategia de ordenamiento.
    """
    eq_proc = df_equipos.copy()
    pa_proc = df_parametros.copy()
    
    col_sort = next((c for c in eq_proc.columns if c.lower().strip() == "dotacion"), None)
    
    if strategy == "random":
        eq_proc = eq_proc.sample(frac=1).reset_index(drop=True)
    
    elif strategy == "size_desc" and col_sort:
        eq_proc = eq_proc.sort_values(by=col_sort, ascending=False).reset_index(drop=True)
        
    elif strategy == "size_asc" and col_sort:
        eq_proc = eq_proc.sort_values(by=col_sort, ascending=True).reset_index(drop=True)

    rows, deficit_report = compute_distribution_from_excel(eq_proc, pa_proc, 2)
    
    return rows, deficit_report

def clean_reservation_df(df, tipo="puesto"):
    if df.empty: return df
    cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at', 'registro', 'id.1']]
    df = df.drop(columns=cols_drop, errors='ignore')
    
    if tipo == "puesto":
        df = df.rename(columns={'user_name': 'Nombre', 'user_email': 'Correo', 'piso': 'Piso', 'reservation_date': 'Fecha Reserva', 'team_area': 'Ubicación'})
        cols = ['Fecha Reserva', 'Piso', 'Ubicación', 'Nombre', 'Correo']
        return df[[c for c in cols if c in df.columns]]
    elif tipo == "sala":
        df = df.rename(columns={'user_name': 'Nombre', 'user_email': 'Correo', 'piso': 'Piso', 'room_name': 'Sala', 'reservation_date': 'Fecha', 'start_time': 'Inicio', 'end_time': 'Fin'})
        cols = ['Fecha', 'Inicio', 'Fin', 'Sala', 'Piso', 'Nombre', 'Correo']
        return df[[c for c in cols if c in df.columns]]
    return df

# --- DIALOGOS MODALES ---
@st.dialog("Confirmar Anulación de Puesto")
def confirm_delete_dialog(conn, usuario, fecha_str, area, piso):
    st.warning(f"¿Anular reserva de puesto?\n\n👤 {usuario} | 📅 {fecha_str}\n📍 {piso} - {area}")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", width="stretch", key="yes_p"):
        if delete_reservation_from_db(conn, usuario, fecha_str, area): st.success("Eliminada"); st.rerun()
    if c2.button("Cancelar", width="stretch", key="no_p"): st.rerun()

@st.dialog("Confirmar Anulación de Sala")
def confirm_delete_room_dialog(conn, usuario, fecha_str, sala, inicio):
    st.warning(f"¿Anular reserva de sala?\n\n👤 {usuario} | 📅 {fecha_str}\n🏢 {sala} ({inicio})")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", width="stretch", key="yes_s"):
        if delete_room_reservation_from_db(conn, usuario, fecha_str, sala, inicio): st.success("Eliminada"); st.rerun()
    if c2.button("Cancelar", width="stretch", key="no_p"): st.rerun()

# --- UTILS TOKENS ---
def generate_token(): return uuid.uuid4().hex[:8].upper()

# ---------------------------------------------------------
# INICIO APP
# ---------------------------------------------------------
conn = get_conn()

# MODIFICADO: Protección para no inicializar DB mil veces (Error 429)
if "db_initialized" not in st.session_state:
    with st.spinner('Conectando a Google Sheets...'):
        init_db(conn)
    st.session_state["db_initialized"] = True

apply_appearance_styles(conn)

# MODIFICADO: Cargar Settings una sola vez
if "app_settings" not in st.session_state:
    st.session_state["app_settings"] = get_all_settings(conn)

settings = st.session_state["app_settings"]

# Definir variables
site_title = settings.get("site_title", "Distribución de Puestos")
global_logo_path = settings.get("logo_path", "static/logo.png")

if os.path.exists(global_logo_path):
    c1, c2 = st.columns([1, 5])
    c1.image(global_logo_path, width=150)
    c2.title(site_title)
else:
    st.title(site_title)

# ---------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------
menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

# ==========================================
# A. VISTA PÚBLICA
# ==========================================
if menu == "Vista pública":
    st.header("Cupos y Planos")
    
    df = read_distribution_df(conn)
    
    if not df.empty:
        cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at']]
        df_view = df.drop(columns=cols_drop, errors='ignore')
        df_view = apply_sorting_to_df(df_view, ORDER_DIAS)
        pisos_disponibles = sort_floors(df["piso"].unique())
    else:
        df_view = df
        pisos_disponibles = ["Piso 1"]

    if df.empty: st.info("Sin datos.")
    else:
        t1, t2 = st.tabs(["Estadísticas", "Ver Planos"])
        with t1:
            st.markdown("""
                <style>
                [data-testid="stElementToolbar"] {
                    display: none;
                }
                </style>
                """, unsafe_allow_html=True)
            
            lib = df_view[df_view["equipo"]=="Cupos libres"].groupby(["piso","dia"], as_index=True, observed=False).agg({"cupos":"sum"}).reset_index()
            lib = apply_sorting_to_df(lib, ORDER_DIAS)
            
            st.subheader("Distribución completa")
            st.dataframe(df_view, hide_index=True, width=None, use_container_width=True)
            
            st.subheader("Cupos libres por piso y día")
            st.dataframe(lib, hide_index=True, width=None, use_container_width=True)
        
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            p_sel = c1.selectbox("Selecciona Piso", pisos_disponibles)
            ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS)
            pn = p_sel.replace("Piso ", "").strip()
            st.write("---")
            
            if ds == "Todos (Lunes a Viernes)":
                m = create_merged_pdf(p_sel, ORDER_DIAS, conn, read_distribution_df, global_logo_path, st.session_state.get('last_style_config', {}))
                if m: 
                    st.success("✅ Dossier disponible.")
                    st.download_button("📥 Descargar Semana (PDF)", m, f"Planos_{p_sel}_Semana.pdf", "application/pdf", use_container_width=True)
                else: st.warning("Sin planos generados.")
            else:
                dsf = ds.lower().replace("é","e").replace("á","a")
                fpng = COLORED_DIR / f"piso_{pn}_{dsf}_combined.png"
                fpdf = COLORED_DIR / f"piso_{pn}_{dsf}_combined.pdf"
                
                opts = []
                if fpng.exists(): opts.append("Imagen (PNG)")
                if fpdf.exists(): opts.append("Documento (PDF)")
                
                if opts:
                    if fpng.exists(): st.image(str(fpng), width=550, caption=f"{p_sel} - {ds}")
                    sf = st.selectbox("Formato:", opts, key="dl_pub")
                    tf = fpng if "PNG" in sf else fpdf
                    mim = "image/png" if "PNG" in sf else "application/pdf"
                    with open(tf,"rb") as f: st.download_button(f"📥 Descargar {sf}", f, tf.name, mim, use_container_width=True)
                else: st.warning("No generado.")

# ==========================================
# B. RESERVAS 
# ==========================================
elif menu == "Reservas":
    
    st.header("Gestión de Reservas")
    
    opcion_reserva = st.selectbox(
        "¿Qué deseas gestionar hoy?",
        ["🪑 Reservar Puesto Flex", "🏢 Reservar Sala de Reuniones", "📋 Mis Reservas y Listados"],
        index=0
    )
    st.markdown("---")

    # ---------------------------------------------------------
    # OPCIÓN 1: RESERVAR PUESTO 
    # ---------------------------------------------------------
    if opcion_reserva == "🪑 Reservar Puesto Flex":
        st.subheader("Disponibilidad de Puestos")
        st.info("Reserva de 'Cupos libres' (Máximo 2 días por mes).")
        
        df = read_distribution_df(conn)
        
        if df.empty:
            st.warning("⚠️ No hay configuración de distribución cargada en el sistema.")
        else:
            c1, c2 = st.columns(2)
            fe = c1.date_input("Selecciona Fecha", min_value=datetime.date.today(), key="fp")
            pisos_disp = sort_floors(df["piso"].unique())
            pi = c2.selectbox("Selecciona Piso", pisos_disp, key="pp")
            
            dn = ORDER_DIAS[fe.weekday()] if fe.weekday() < 5 else "FinDeSemana"
            
            if dn == "FinDeSemana":
                st.error("🔒 Es fin de semana. No se pueden realizar reservas.")
            else:
                rg = df[(df["piso"] == pi) & (df["dia"] == dn) & (df["equipo"] == "Cupos libres")]
                
                hay_config = False
                total_cupos = 0
                disponibles = 0
                
                if not rg.empty:
                    hay_config = True
                    total_cupos = int(rg.iloc[0]["cupos"])
                    
                    all_res = list_reservations_df(conn)
                    ocupados = 0
                    if not all_res.empty:
                        mask = (all_res["reservation_date"].astype(str) == str(fe)) & \
                               (all_res["piso"] == pi) & \
                               (all_res["team_area"] == "Cupos libres")
                        ocupados = len(all_res[mask])
                    
                    disponibles = total_cupos - ocupados
                
                if not hay_config:
                    st.warning(f"⚠️ El {pi} no tiene habilitados 'Cupos libres' para los días {dn}.")
                else:
                    if disponibles > 0:
                        st.success(f"✅ **HAY CUPO: Quedan {disponibles} puestos disponibles** (Total: {total_cupos}).")
                    else:
                        st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")
                    
                    st.markdown("### Datos del Solicitante")
                    
                    with st.form("form_puesto"):
                        cf1, cf2 = st.columns(2)
                        nm = cf1.text_input("Nombre Completo")
                        em = cf2.text_input("Correo Electrónico")
                        
                        submitted = st.form_submit_button("Confirmar Reserva", type="primary", disabled=(disponibles <= 0))
                        
                        if submitted:
                            if not nm or not em:
                                st.error("Por favor completa nombre y correo.")
                            elif user_has_reservation(conn, em, str(fe)):
                                st.error("Ya tienes una reserva registrada para esta fecha.")
                            elif count_monthly_free_spots(conn, em, fe) >= 2:
                                st.error("Has alcanzado tu límite de 2 reservas mensuales.")
                            elif disponibles <= 0:
                                st.error("Lo sentimos, el cupo se acaba de agotar.")
                            else:
                                add_reservation(conn, nm, em, pi, str(fe), "Cupos libres", datetime.datetime.now(datetime.timezone.utc).isoformat())
                                msg = f"✅ Reserva Confirmada:\n\n- Usuario: {nm}\n- Fecha: {fe}\n- Piso: {pi}\n- Tipo: Puesto Flex"
                                st.success(msg)
                                send_reservation_email(em, "Confirmación Puesto", msg.replace("\n","<br>"))
                                st.rerun()

    # ---------------------------------------------------------
    # OPCIÓN 2: RESERVAR SALA
    # ---------------------------------------------------------
    elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
        st.subheader("Agendar Sala")
        
        c_sala, c_fecha = st.columns(2)
        sl = c_sala.selectbox("Selecciona Sala", ["Sala Grande Piso 1", "Sala Pequeña Piso 1", "Sala Piso 2", "Sala Piso 3"])
        pi_s = "Piso " + sl.split("Piso ")[1].replace(")", "")
        fe_s = c_fecha.date_input("Fecha", min_value=datetime.date.today(), key="fs")
        
        tm = generate_time_slots("08:00", "20:00", 15)
        
        st.write("Horario:")
        ch1, ch2 = st.columns(2)
        i = ch1.selectbox("Inicio", tm)
        f = ch2.selectbox("Fin", tm, index=min(4, len(tm)-1))
        
        st.markdown("### Datos del Responsable")
        with st.form("form_sala"):
            cf1, cf2 = st.columns(2)
            n_s = cf1.text_input("Nombre Solicitante")
            e_s = cf2.text_input("Correo Solicitante")
            
            sub_sala = st.form_submit_button("Confirmar Sala", type="primary")
            
            if sub_sala:
                if not n_s:
                    st.error("Falta el nombre.")
                elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, i, f):
                    st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
                else:
                    add_room_reservation(conn, n_s, e_s, pi_s, sl, str(fe_s), i, f, datetime.datetime.now(datetime.timezone.utc).isoformat())
                    msg = f"✅ Sala Confirmada:\n\n- Sala: {sl}\n- Fecha: {fe_s}\n- Horario: {i} - {f}"
                    st.success(msg)
                    if e_s: send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))

    # ---------------------------------------------------------
    # OPCIÓN 3: GESTIONAR (ANULAR Y VER TODO) - CORRECCIÓN DE ROBUSTEZ
    # ---------------------------------------------------------
    elif opcion_reserva == "📋 Mis Reservas y Listados":
        
        # --- SECCION 1: BUSCADOR PARA ANULAR ---
        st.subheader("Buscar y Cancelar mis reservas")
        q = st.text_input("Ingresa tu Correo o Nombre para buscar:")
        
        if q:
            dp = list_reservations_df(conn)
            ds = get_room_reservations_df(conn)

            # --- CORRECCIÓN DE ROBUSTEZ DE DATAFRAME ---
            mp = pd.DataFrame()
            if not dp.empty and 'user_name' in dp.columns and 'user_email' in dp.columns:
                mp = dp[(dp['user_name'].astype(str).str.lower().str.contains(q.lower())) | (dp['user_email'].astype(str).str.lower().str.contains(q.lower()))]

            ms = pd.DataFrame()
            if not ds.empty and 'user_name' in ds.columns and 'user_email' in ds.columns:
                ms = ds[(ds['user_name'].astype(str).str.lower().str.contains(q.lower())) | (ds['user_email'].astype(str).str.lower().str.contains(q.lower()))]
            # -------------------------------------------
            
            if mp.empty and ms.empty:
                st.warning("No encontré reservas con esos datos.")
            else:
                if not mp.empty:
                    st.markdown("#### 🪑 Tus Puestos")
                    for idx, r in mp.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([5, 1])
                            c1.markdown(f"**{r['reservation_date']}** | {r['piso']} (Cupo Libre)")
                            if c2.button("Anular", key=f"del_p_{idx}", type="primary"):
                                confirm_delete_dialog(conn, r['user_name'], r['reservation_date'], r['team_area'], r['piso'])

                if not ms.empty:
                    st.markdown("#### 🏢 Tus Salas")
                    for idx, r in ms.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([5, 1])
                            c1.markdown(f"**{r['reservation_date']}** | {r['room_name']} | {r['start_time']} - {r['end_time']}")
                            if c2.button("Anular", key=f"del_s_{idx}", type="primary"):
                                confirm_delete_room_dialog(conn, r['user_name'], r['reservation_date'], r['room_name'], r['start_time'])

        st.markdown("---")
        
        # --- SECCION 2: VER TODO (TABLAS CORREGIDAS) ---
        with st.expander("Ver Listado General de Reservas", expanded=True):
            
            st.subheader("Reserva de puestos") 
            st.dataframe(clean_reservation_df(list_reservations_df(conn)), hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True) 

            st.subheader("Reserva de salas") 
            st.dataframe(clean_reservation_df(get_room_reservations_df(conn), "sala"), hide_index=True, use_container_width=True)

# ==========================================
# E. ADMINISTRADOR
# ==========================================
elif menu == "Administrador":
    st.header("Admin")
    admin_user, admin_pass = get_admin_credentials(conn)
    if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
    
    if not st.session_state["is_admin"]:
        u = st.text_input("Usuario"); p = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            if u==admin_user and p==admin_pass: st.session_state["is_admin"]=True; st.rerun()
            else: st.error("Credenciales incorrectas")
        with st.expander("Recuperar Contraseña"):
            em_chk = st.text_input("Email Registrado")
            if st.button("Solicitar"):
                re = settings.get("admin_email","")
                if re and em_chk.lower()==re.lower():
                    t = generate_token()
                    save_reset_token(conn, t, (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat())
                    send_reservation_email(re, "Token", f"Token: {t}"); st.success("Enviado.")
                else: st.error("Email no coincide.")
            tk = st.text_input("Token"); nu = st.text_input("Nuevo User"); np = st.text_input("Nueva Pass", type="password")
            if st.button("Cambiar"):
                ok, m = validate_and_consume_token(conn, tk)
                if ok: save_setting(conn, "admin_user", nu); save_setting(conn, "admin_pass", np); st.success("OK")
                else: st.error(m)
        st.stop()

    if st.button("Cerrar Sesión"): st.session_state["is_admin"]=False; st.rerun()

    t1, t2, t3, t4, t5, t6 = st.tabs(["Excel", "Editor Visual", "Informes", "Config", "Apariencia", "Mantenimiento"])
    
    # -----------------------------------------------------------
    # T1: GENERADOR DE DISTRIBUCIÓN (CON AUTO-OPTIMIZACIÓN JUSTA)
    # -----------------------------------------------------------
    with t1:
        st.subheader("Generador de Distribución Inteligente")
        st.markdown("Sube el archivo Excel y elige una estrategia. Usa **Auto-Optimizar** para buscar la distribución más equitativa.")
        
        c_up, c_strat = st.columns([2, 1])
        up = c_up.file_uploader("Subir archivo Excel (Hojas: 'Equipos', 'Parámetros')", type=["xlsx"])
        
        # SELECTOR DE ESTRATEGIA
        estrategia = c_strat.radio(
            "Estrategia Base:",
            ["🎲 Aleatorio (Recomendado para Optimizar)", "🧩 Tetris (Grandes primero)", "🐜 Relleno (Pequeños primero)"],
            help="Aleatorio da mejores resultados al usar Auto-Optimizar porque prueba más combinaciones distintas."
        )
        
        strat_map = {
            "🧩 Tetris (Grandes primero)": "size_desc",
            "🎲 Aleatorio (Recomendado para Optimizar)": "random",
            "🐜 Relleno (Pequeños primero)": "size_asc"
        }
        sel_strat_code = strat_map[estrategia]

        # Inicializar variables de sesión
        if 'excel_equipos' not in st.session_state: st.session_state['excel_equipos'] = None
        if 'excel_params' not in st.session_state: st.session_state['excel_params'] = None
        if 'proposal_rows' not in st.session_state: st.session_state['proposal_rows'] = None
        if 'proposal_deficit' not in st.session_state: st.session_state['proposal_deficit'] = None
        if 'last_optimization_stats' not in st.session_state: st.session_state['last_optimization_stats'] = None

        # 1. CARGA DEL ARCHIVO
        if up:
            try:
                if st.button("📂 Procesar", type="primary"):
                    df_eq = pd.read_excel(up, "Equipos")
                    df_pa = pd.read_excel(up, "Parámetros")
                    
                    st.session_state['excel_equipos'] = df_eq
                    st.session_state['excel_params'] = df_pa
                    
                    rows, deficit = get_distribution_proposal(df_eq, df_pa, strategy=sel_strat_code)
                    st.session_state['proposal_rows'] = rows
                    st.session_state['proposal_deficit'] = deficit
                    st.session_state['last_optimization_stats'] = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error al leer el Excel: {e}")

        # 2. VISUALIZACIÓN Y ACCIONES
        if st.session_state['proposal_rows'] is not None:
            st.divider()
            
            # --- SECCIÓN DE RESULTADOS ---
            n_def = len(st.session_state['proposal_deficit']) if st.session_state['proposal_deficit'] else 0
            
            if st.session_state['last_optimization_stats']:
                stats = st.session_state['last_optimization_stats']
                st.info(f"✨ **Resultado Optimizado:** Se probaron {stats['iterations']} combinaciones. Se eligió la que menos castiga repetidamente al mismo equipo.")

            if n_def == 0:
                st.success("✅ **¡Distribución Perfecta!** 0 conflictos detectados.")
            else:
                st.warning(f"⚠️ **Distribución Actual:** {n_def} cupos faltantes en total.")

            t_view, t_def = st.tabs(["📊 Distribución Visual", "🚨 Reporte de Conflictos"])
            
            with t_view:
                df_preview = pd.DataFrame(st.session_state['proposal_rows'])
                if not df_preview.empty:
                    df_sorted = apply_sorting_to_df(df_preview, ORDER_DIAS)
                    st.dataframe(df_sorted, hide_index=True, width=None, use_container_width=True)
                else:
                    st.warning("No se generaron asignaciones.")
            
            with t_def:
                if st.session_state['proposal_deficit']:
                    def_df = pd.DataFrame(st.session_state['proposal_deficit'])
                    
                    # Contamos cuántas veces aparece cada equipo en el reporte de déficit
                    conteo_injusticia = def_df['equipo'].value_counts().reset_index()
                    conteo_injusticia.columns = ['Equipo', 'Veces Perjudicado']
                    
                    c1, c2 = st.columns(2)
                    c1.markdown("**Detalle de Conflictos:**")
                    c1.dataframe(def_df, use_container_width=True)
                    
                    c2.markdown("**⚠️ Equipos más afectados (Repetición):**")
                    c2.dataframe(conteo_injusticia, use_container_width=True)
                    
                    if conteo_injusticia['Veces Perjudicado'].max() > 1:
                        c2.error("Hay equipos sufriendo déficit múltiples días. Se recomienda usar 'Auto-Optimizar'.")
                else:
                    st.info("Sin conflictos. Todos los equipos caben perfectamente.")

            st.markdown("---")
            st.markdown("### 🔧 Herramientas de Justicia")
            
            c_actions = st.columns([1, 1, 1])
            
            # Botón 1: Regenerar simple
            if c_actions[0].button("🔄 Probar otra suerte"):
                with st.spinner("Generando nueva variación..."):
                    rows, deficit = get_distribution_proposal(
                        st.session_state['excel_equipos'], 
                        st.session_state['excel_params'], 
                        strategy=sel_strat_code
                    )
                    st.session_state['proposal_rows'] = rows
                    st.session_state['proposal_deficit'] = deficit
                    st.session_state['last_optimization_stats'] = None
                st.rerun()

            # Botón 2: AUTO-OPTIMIZAR JUSTICIA (LOGICA NUEVA)
            if c_actions[1].button("✨ Auto-Optimizar (Buscar Equidad)"):
                
                NUM_INTENTOS = 20 
                
                progress_text = "Analizando múltiples escenarios para repartir la carga..."
                my_bar = st.progress(0, text=progress_text)
                
                best_rows = None
                best_deficit = None
                
                # Puntuación inicial (mientras más baja mejor)
                min_unfairness_score = 999999 
                min_total_conflicts = 999999
                
                for i in range(NUM_INTENTOS):
                    r, d = get_distribution_proposal(st.session_state['excel_equipos'], st.session_state['excel_params'], strategy="random")
                    
                    current_conflicts = len(d) if d else 0
                    
                    # Calcular Score de Injusticia
                    if d:
                        equipos_afectados = [x['equipo'] for x in d]
                        freqs = {x:equipos_afectados.count(x) for x in set(equipos_afectados)}
                        unfairness_score = sum([val**2 for val in freqs.values()])
                    else:
                        unfairness_score = 0
                    
                    if unfairness_score < min_unfairness_score:
                        min_unfairness_score = unfairness_score
                        min_total_conflicts = current_conflicts
                        best_rows = r
                        best_deficit = d
                    elif unfairness_score == min_unfairness_score:
                        if current_conflicts < min_total_conflicts:
                            min_total_conflicts = current_conflicts
                            best_rows = r
                            best_deficit = d
                    
                    my_bar.progress(int((i + 1) / NUM_INTENTOS * 100), text=f"Simulando escenario {i+1}/{NUM_INTENTOS}...")
                
                st.session_state['proposal_rows'] = best_rows
                st.session_state['proposal_deficit'] = best_deficit
                st.session_state['last_optimization_stats'] = {'iterations': NUM_INTENTOS, 'score': min_unfairness_score}
                
                my_bar.empty()
                st.toast("¡Optimización finalizada! Se aplicó el criterio de equidad.", icon="⚖️")
                st.rerun()

            # Botón 3: Guardar
            if c_actions[2].button("💾 Guardar Definitivo", type="primary"):
                try:
                    clear_distribution(conn)
                    insert_distribution(conn, st.session_state['proposal_rows'])
                    
                    if st.session_state['proposal_deficit']:
                        st.session_state['deficit_report'] = st.session_state['proposal_deficit']
                    elif 'deficit_report' in st.session_state:
                        del st.session_state['deficit_report']
                        
                    st.success("✅ Distribución guardada exitosamente.")
                    st.balloons()
                    st.session_state['proposal_rows'] = None
                    st.session_state['excel_equipos'] = None
                    st.session_state['last_optimization_stats'] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

    with t2:
        st.info("Editor de Zonas")
        zonas = load_zones()
        c1, c2 = st.columns(2)
        
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        
        p_sel = c1.selectbox("Piso", pisos_list); d_sel = c2.selectbox("Día Ref.", ORDER_DIAS)
        p_num = p_sel.replace("Piso ", "").strip()
        
        # --- CÓDIGO CORREGIDO: CARGA DEL PLANO (SOPORTE ESPACIO/MAYÚSCULAS) ---
        file_base = f"piso{p_num}"
        pim = PLANOS_DIR / f"{file_base}.png"
        if not pim.exists():  
            pim = PLANOS_DIR / f"{file_base}.jpg"
        if not pim.exists(): # Búsqueda con espacio
            pim = PLANOS_DIR / f"piso {p_num}.png"
        if not pim.exists(): # Búsqueda con espacio .jpg
            pim = PLANOS_DIR / f"piso {p_num}.jpg"
        if not pim.exists(): # Fallback a P mayúscula
            pim = PLANOS_DIR / f"Piso{p_num}.png"
            
        
        if pim.exists():
            # 1. Carga de imagen
            img = PILImage.open(pim)
            
            cw = 800; w, h = img.size
            ch = int(h * (cw/w)) if w>cw else h
            cw = w if w<=cw else cw

            # 2. SOLUCIÓN ROBUSTA: Codificación a URL Base64 para eludir la función rota
            # Esto es necesario porque el parche falló.
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            background_image_url = f"data:image/png;base64,{img_str}"
            
            # 3. Llamada al Canvas usando la URL de datos
            canvas = st_canvas(fill_color="rgba(0, 160, 74, 0.3)", stroke_width=2, stroke_color="#00A04A", background_image=background_image_url, update_streamlit=True, width=cw, height=ch, drawing_mode="rect", key=f"cv_{p_sel}")
        
            current_seats_dict = {}
            eqs = [""]
            if not df_d.empty:
                subset = df_d[(df_d['piso'] == p_sel) & (df_d['dia'] == d_sel)]
                current_seats_dict = dict(zip(subset['equipo'], subset['cupos']))
                eqs += sorted(subset['equipo'].unique().tolist())
            
            salas_piso = []
            if "1" in p_sel: salas_piso = ["Sala Grande - Piso 1", "Sala Pequeña - Piso 1"]
            elif "2" in p_sel: salas_piso = ["Sala Reuniones - Piso 2"]
            elif "3" in p_sel: salas_piso = ["Sala Reuniones - Piso 3"]
            eqs = eqs + salas_piso

            c1, c2, c3 = st.columns([2,1,1])
            tn = c1.selectbox("Equipo / Sala", eqs); tc = c2.color_picker("Color", "#00A04A")
            if tn and tn in current_seats_dict: st.info(f"Cupos: {current_seats_dict[tn]}")
            
            if c3.button("Guardar", key="sz"):
                if tn and canvas.json_data and canvas.json_data["objects"]: # Check robusto
                    o = canvas.json_data["objects"][-1]
                    zonas.setdefault(p_sel, []).append({"team": tn, "x": int(o["left"]), "y": int(o["top"]), "w": int(o["width"]*o.get("scaleX",1)), "h": int(o["height"]*o.get("scaleY",1)), "color": tc})
                    save_zones(zonas); st.success("OK")
                elif not canvas.json_data or not canvas.json_data.get("objects"):
                    st.error("Dibuja un área primero en el plano.")
            
            st.divider()
            if p_sel in zonas:
                for i, z in enumerate(zonas[p_sel]):
                    c1, c2 = st.columns([4,1])
                    c1.markdown(f"<span style='color:{z['color']}'>■</span> {z['team']}</span>", unsafe_allow_html=True)
                    if c2.button("X", key=f"d{i}"): zonas[p_sel].pop(i); save_zones(zonas); st.rerun()

            st.divider()
            st.subheader("Personalización Título y Leyenda")
            with st.expander("🎨 Editar Estilos", expanded=True):
                tm = st.text_input("Título Principal", f"Distribución {p_sel}")
                ts = st.text_input("Subtítulo (Opcional)", f"Día: {d_sel}")
                
                align_options = ["Izquierda", "Centro", "Derecha"]

                st.markdown("##### Estilos del Título Principal")
                cf1, cf2, cf3 = st.columns(3)
                ff_t = cf1.selectbox("Tipografía (Título)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key="font_t")
                fs_t = cf2.selectbox("Tamaño Letra (Título)", [10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 64, 72, 80], index=9, key="size_t")
                align = cf3.selectbox("Alineación (Título)", align_options, index=1)

                st.markdown("---")
                st.markdown("##### Estilos del Subtítulo")
                cs1, cs2, cs3 = st.columns(3)
                ff_s = cs1.selectbox("Tipografía (Subtítulo)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key="font_s")
                fs_s = cs2.selectbox("Tamaño Letra (Subtítulo)", [10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 64, 72, 80], index=5, key="size_s")
                align_s = cs3.selectbox("Alineación (Subtítulo)", align_options, index=1)

                st.markdown("---")
                st.markdown("##### Estilos de la Leyenda")
                cl1, cl2, cl3 = st.columns(3)
                ff_l = cl1.selectbox("Tipografía (Leyenda)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key="font_l", index=0)
                fs_l = cl2.selectbox("Tamaño Letra (Leyenda)", [8, 10, 12, 14, 16, 18, 20, 24, 28, 32], index=3, key="size_l")
                align_l = cl3.selectbox("Alineación (Leyenda)", align_options, index=0)
                
                st.markdown("---")
                cg1, cg2, cg3, cg4 = st.columns(4) 
                lg = cg1.checkbox("Logo", True, key="chk_logo"); 
                ln = cg2.checkbox("Mostrar Leyenda", True, key="chk_legend");
                align_logo = cg3.selectbox("Alineación Logo", align_options, index=0)
                lw = cg4.slider("Ancho Logo", 50, 300, 150)
                
                cc1, cc2 = st.columns(2)
                bg = cc1.color_picker("Fondo Header", "#FFFFFF"); tx = cc2.color_picker("Color Texto", "#000000")

            fmt_sel = st.selectbox("Formato:", ["Imagen (PNG)", "Documento (PDF)"])
            f_code = "PNG" if "PNG" in fmt_sel else "PDF"
            
            if st.button("🎨 Actualizar Vista Previa"):
                conf = {
                    "title_text": tm,
                    "subtitle_text": ts,
                    "title_font": ff_t,
                    "title_size": fs_t,
                    "subtitle_font": ff_s,
                    "subtitle_size": fs_s,
                    "legend_font": ff_l,
                    "legend_size": fs_l,
                    "alignment": align, 
                    "subtitle_align": align_s, 
                    "legend_align": align_l, 
                    "bg_color": bg, 
                    "title_color": tx, 
                    "subtitle_color": "#666666", 
                    "use_logo": lg, 
                    "use_legend": ln, 
                    "logo_width": lw,
                    "logo_align": align_logo
                }
                st.session_state['last_style_config'] = conf
                
                out = generate_colored_plan(p_sel, d_sel, current_seats_dict, f_code, conf, global_logo_path)
                if out: st.success("Generado.")
            
            ds = d_sel.lower().replace("é","e").replace("á","a")
            fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
            fpdf = COLORED_DIR / f"piso_{p_num}_{ds}_combined.pdf"
            
            if fpng.exists(): st.image(str(fpng), width=550, caption="Vista Previa")
            elif fpdf.exists(): st.info("PDF generado (sin vista previa)")
            
            tf = fpng if "PNG" in fmt_sel else fpdf
            mm = "image/png" if "PNG" in fmt_sel else "application/pdf"
            if tf.exists():
                with open(tf,"rb") as f: st.download_button(f"📥 Descargar {fmt_sel}", f, tf.name, mm, use_container_width=True)

    with t3:
        st.subheader("Generar Reportes de Distribución")
        
        if 'deficit_report' in st.session_state and st.session_state['deficit_report']:
            st.markdown("---")
            st.error("🚨 INFORME DE DÉFICIT DE CUPOS")
            
            df_deficit = pd.DataFrame(st.session_state['deficit_report'])
            df_deficit = df_deficit.rename(columns={
                'piso': 'Piso', 
                'dia': 'Día', 
                'equipo': 'Equipo', 
                'deficit': 'Cupos Faltantes',
                'causa': 'Observación'
            })
            st.dataframe(df_deficit, hide_index=True, width=None, use_container_width=True)
            st.markdown("---")

        rf = st.selectbox("Formato Reporte", ["Excel", "PDF"])
        if st.button("Generar Reporte"):
            df_raw = read_distribution_df(conn); df_raw = apply_sorting_to_df(df_raw, ORDER_DIAS)
            if "Excel" in rf:
                b = BytesIO()
                with pd.ExcelWriter(b) as w: df_raw.to_excel(w, index=False)
                st.session_state['rd'] = b.getvalue(); st.session_state['rn'] = "d.xlsx"; st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                df = df_raw.rename(columns={c: c.lower() for c in df_raw.columns})
                d_data = st.session_state.get('deficit_report', [])
                st.session_state['rd'] = generate_full_pdf(df, logo_path=Path(global_logo_path), deficit_data=d_data, order_dias=ORDER_DIAS)
                st.session_state['rn'] = "reporte_distribucion.pdf"; st.session_state['rm'] = "application/pdf"
            st.success("OK")
        if 'rd' in st.session_state: st.download_button("Descargar", st.session_state['rd'], st.session_state['rn'], mime=st.session_state['rm'])
        
        st.markdown("---")
        cp, cd = st.columns(2)
        pi = cp.selectbox("Piso", pisos_list, key="pi2"); di = cd.selectbox("Día", ["Todos"]+ORDER_DIAS, key="di2")
        if di=="Todos":
            if st.button("Generar Dossier"):
                m = create_merged_pdf(pi, ORDER_DIAS, conn, read_distribution_df, global_logo_path, st.session_state.get('last_style_config', {}))
                if m: st.session_state['dos'] = m; st.success("OK")
            if 'dos' in st.session_state: st.download_button("Descargar Dossier", st.session_state['dos'], "S.pdf", "application/pdf")
        else:
            ds = di.lower().replace("é","e").replace("á","a")
            pn = pi.split()[-1]
            fpng = COLORED_DIR / f"piso_{pn}_{ds}_combined.png"
            fd = COLORED_DIR / f"piso_{pn}_{ds}_combined.pdf"
            ops = []
            if fpng.exists(): ops.append("Imagen (PNG)")
            if fd.exists(): ops.append("Documento (PDF)")
            if ops:
                if fpng.exists(): st.image(str(fpng), width=300)
                sf = st.selectbox("Fmt", ops, key="sf2")
                tf = fpng if "PNG" in sf else fd
                mm = "image/png" if "PNG" in sf else "application/pdf"
                with open(tf,"rb") as f: st.download_button("Descargar", f, tf.name, mm)
            else: st.warning("No existe.")

    with t4:
        nu = st.text_input("User"); np = st.text_input("Pass", type="password"); ne = st.text_input("Email")
        if st.button("Guardar", key="sc"): save_setting(conn, "admin_user", nu); save_setting(conn, "admin_pass", np); save_setting(conn, "admin_email", ne); st.success("OK")

    with t5: admin_appearance_ui(conn)
    
    with t6:
        opt = st.radio("Borrar:", ["Reservas", "Distribución", "Planos/Zonas", "TODO"])
        if st.button("BORRAR", type="primary"): msg = perform_granular_delete(conn, opt); st.success(msg)
