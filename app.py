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
import base64
import numpy as np

# ---------------------------------------------------------
# 1. PARCHE DE COMPATIBILIDAD (Reforzado para Streamlit 1.51+)
# ---------------------------------------------------------
import streamlit.elements.image as st_image

# Intentamos "secuestrar" la función image_to_url de donde vive ahora
try:
    from streamlit.elements.lib.image_utils import image_to_url as _internal_image_to_url
except ImportError:
    try:
        from streamlit.runtime.media_file_storage import image_to_url as _internal_image_to_url
    except ImportError:
        def _internal_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
            return ""

# INYECCIÓN: Si st_image no tiene la función, se la ponemos manualmente
if not hasattr(st_image, 'image_to_url'):
    st_image.image_to_url = _internal_image_to_url

# Parche adicional para el objeto WidthConfig que también suele faltar
if not hasattr(st_image, 'WidthConfig'):
    @dataclass
    class WidthConfig:
        width: int
    st_image.WidthConfig = WidthConfig

# ---------------------------------------------------------
# 2. IMPORTACIÓN SEGURA DE HERRAMIENTAS VISUALES
# ---------------------------------------------------------
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

try:
    from streamlit_image_annotation import image_annotation
except ImportError:
    image_annotation = None

# ---------------------------------------------------------
# 3. IMPORTACIONES DE MÓDULOS PROPIOS
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
from modules.seats import compute_distribution_from_excel, compute_ideal_distribution
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones
from modules.pdfgen import generate_full_pdf, create_merged_pdf

# ---------------------------------------------------------
# 4. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

# 1. Verificar si existen los secretos
if "gcp_service_account" not in st.secrets:
    st.error("🚨 ERROR CRÍTICO: No se encuentran los secretos [gcp_service_account].")
    st.stop()

# 2. Intentar conectar
try:
    creds_dict = dict(st.secrets["gcp_service_account"])
    pk = creds_dict.get("private_key", "")
    if "-----BEGIN PRIVATE KEY-----" not in pk:
        st.error("🚨 ERROR EN PRIVATE KEY: Formato incorrecto en secrets.toml")
        st.stop()
        
    from google.oauth2.service_account import Credentials
    import gspread
    
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet_name = st.secrets["sheets"]["sheet_name"]
    sh = client.open(sheet_name)

except Exception as e:
    st.error(f"🔥 LA CONEXIÓN FALLÓ: {str(e)}")
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
# 5. FUNCIONES HELPER LOCALES (VISUALIZACIÓN APP)
# ---------------------------------------------------------

def sort_floors(floor_list):
    def extract_num(text):
        text = str(text)
        num = re.findall(r'\d+', text)
        return int(num[0]) if num else 0
    return sorted(list(floor_list), key=extract_num)

def apply_sorting_to_df(df):
    if df.empty: return df
    df = df.copy()
    
    cols_lower = {c.lower(): c for c in df.columns}
    col_dia = cols_lower.get('dia') or cols_lower.get('día')
    col_piso = cols_lower.get('piso')
    
    if col_dia:
        df[col_dia] = pd.Categorical(df[col_dia], categories=ORDER_DIAS, ordered=True)
    
    if col_piso:
        unique_floors = [str(x) for x in df[col_piso].dropna().unique()]
        sorted_floors = sort_floors(unique_floors)
        df[col_piso] = pd.Categorical(df[col_piso], categories=sorted_floors, ordered=True)

    sort_cols = []
    if col_piso: sort_cols.append(col_piso)
    if col_dia: sort_cols.append(col_dia)
    
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df

def calculate_weekly_usage_summary(distrib_df):
    if distrib_df.empty: return pd.DataFrame()
    
    # Buscar columnas dinamicamente
    def norm(c): return str(c).lower()
    equipo_col = next((c for c in distrib_df.columns if 'equipo' in norm(c)), None)
    cupos_col = next((c for c in distrib_df.columns if 'cupos' in norm(c)), None)
    
    if not equipo_col or not cupos_col: return pd.DataFrame()
    
    equipos_df = distrib_df[distrib_df[equipo_col] != "Cupos libres"]
    if equipos_df.empty: return pd.DataFrame()
    
    weekly = equipos_df.groupby(equipo_col).agg({cupos_col: 'sum'}).reset_index()
    weekly.columns = ['Equipo', 'Total Cupos Semanales']
    
    weekly['Promedio Diario'] = weekly['Total Cupos Semanales'] / 5
    
    # Integrar % semanal si existe info de dotación
    if 'dotacion_total' in distrib_df.columns:
        dot_df = distrib_df[[equipo_col, 'dotacion_total']].drop_duplicates()
        dot_df[equipo_col] = dot_df[equipo_col].astype(str)
        weekly['Equipo'] = weekly['Equipo'].astype(str)
        
        weekly = weekly.merge(dot_df, left_on='Equipo', right_on=equipo_col, how='left')
        
        def calc_pct(row):
            dot = row['dotacion_total']
            if pd.isna(dot) or dot == 0: return 0
            return round((row['Total Cupos Semanales'] / (dot * 5)) * 100, 1)
            
        weekly['% Distr Semanal'] = weekly.apply(calc_pct, axis=1)
        
        # Limpiar
        cols_to_drop = ['dotacion_total']
        if equipo_col != 'Equipo': cols_to_drop.append(equipo_col)
        weekly = weekly.drop(columns=[c for c in cols_to_drop if c in weekly.columns], errors='ignore')

    return weekly

def clean_reservation_df(df, tipo="puesto"):
    if df.empty: return df
    cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at', 'registro', 'id.1']]
    df = df.drop(columns=cols_drop, errors='ignore')
    
    if tipo == "puesto":
        rename_map = {'user_name': 'Nombre', 'user_email': 'Correo', 'piso': 'Piso', 'reservation_date': 'Fecha Reserva', 'team_area': 'Ubicación'}
        df = df.rename(columns=rename_map)
        desired_cols = ['Fecha Reserva', 'Piso', 'Ubicación', 'Nombre', 'Correo']
        existing_cols = [c for c in desired_cols if c in df.columns]
        return df[existing_cols]
        
    elif tipo == "sala":
        rename_map = {'user_name': 'Nombre', 'user_email': 'Correo', 'piso': 'Piso', 'room_name': 'Sala', 'reservation_date': 'Fecha', 'start_time': 'Inicio', 'end_time': 'Fin'}
        df = df.rename(columns=rename_map)
        desired_cols = ['Fecha', 'Inicio', 'Fin', 'Sala', 'Piso', 'Nombre', 'Correo']
        existing_cols = [c for c in desired_cols if c in df.columns]
        return df[existing_cols]
    return df

def show_distribution_insights(rows, deficit_data):
    df = pd.DataFrame(rows)
    st.subheader("📊 Métricas de la Distribución")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Cupos Asignados", df['cupos'].sum())
    with c2: st.metric("Cupos Libres", df[df['equipo'] == 'Cupos libres']['cupos'].sum())
    with c3: st.metric("Equipos Asignados", df[df['equipo'] != 'Cupos libres']['equipo'].nunique())
    
    uni = df.groupby('dia')['cupos'].sum().std() if not df.empty else 0
    with c4: st.metric("Uniformidad (σ)", f"{uni:.1f}")
    
    st.subheader("📈 Distribución por Día")
    try:
        if not df.empty:
            cupos_por_dia = df.groupby('dia')['cupos'].sum().reindex(ORDER_DIAS).fillna(0)
            fig, ax = plt.subplots(figsize=(10, 4))
            cupos_por_dia.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel('Total Cupos')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    except: pass

# --- MODALES ---
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
    if c2.button("Cancelar", width="stretch", key="no_s"): st.rerun()

def generate_token(): return uuid.uuid4().hex[:8].upper()

# --- EDITOR MANUAL ARREGLADO (FEEDBACK INMEDIATO) ---
def fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, img_width, img_height, alignment_config=None):
    """Editor manual con imagen ajustada, keys únicos y feedback de cupos"""
    
    p_num = p_sel.replace("Piso ", "").strip()

    st.subheader("🎯 Modo de Dibujo Manual")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Plano del {p_sel} (Referencia)", fontsize=10)
    
    if p_sel in zonas:
        for i, zona in enumerate(zonas[p_sel]):
            rect = plt.Rectangle((zona['x'], zona['y']), zona['w'], zona['h'],
                linewidth=2, edgecolor=zona['color'], facecolor=zona['color'] + '40')
            ax.add_patch(rect)
            ax.text(zona['x'], zona['y'], zona['team'], fontsize=8, color='white', 
                    bbox=dict(facecolor='black', alpha=0.5))
    
    st.pyplot(fig, use_container_width=False)
    
    st.subheader("🖊️ Agregar Nueva Zona")
    
    with st.form(f"zona_form_advanced_{p_sel}_{d_sel}"):
        col1, col2 = st.columns(2)
        with col1:
            current_seats_dict = {}
            eqs = [""]
            if not df_d.empty:
                subset = df_d[(df_d['piso'] == p_sel) & (df_d['dia'] == d_sel)]
                current_seats_dict = dict(zip(subset['equipo'], subset['cupos']))
                eqs += sorted(subset['equipo'].unique().tolist())
            
            salas_piso = []
            if "1" in p_sel: salas_piso = ["Sala Reuniones Pequeña Piso 1", "Sala Reuniones Grande Piso 1"]
            elif "2" in p_sel: salas_piso = ["Sala Reuniones Piso 2"]
            elif "3" in p_sel: salas_piso = ["Sala Reuniones Piso 3"]
            eqs = eqs + salas_piso
            
            equipo = st.selectbox("Equipo / Sala", eqs, key=f"team_sel_adv_{p_sel}")
            color = st.color_picker("Color", "#00A04A", key=f"col_pick_adv_{p_sel}")
            
            st.caption("ℹ️ Selecciona un equipo y haz click en guardar para ver la zona.")
        
        with col2:
            st.info("📍 Coordenadas")
            c_x, c_y = st.columns(2)
            x = c_x.slider("X", 0, img_width, 100, key=f"x_man_{p_sel}")
            y = c_y.slider("Y", 0, img_height, 100, key=f"y_man_{p_sel}")
            
            c_w, c_h = st.columns(2)
            w = c_w.slider("Ancho", 10, 600, 100, key=f"w_man_{p_sel}")
            h = c_h.slider("Alto", 10, 600, 80, key=f"h_man_{p_sel}")
        
        if st.form_submit_button("💾 Guardar Zona"):
            if equipo:
                zonas.setdefault(p_sel, []).append({
                    "team": equipo, "x": x, "y": y, "w": w, "h": h, "color": color
                })
                save_zones(zonas)
                st.success(f"Zona guardada para {equipo}")
                st.rerun()
            else:
                st.warning("Selecciona un equipo")

    st.divider()
    st.subheader("🎨 Generar Imagen Final")
    with st.expander("Configuración"):
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            tit = st.text_input("Título", f"Distribución {p_sel}", key=f"tit_final_{p_sel}")
            sub = st.text_input("Subtítulo", f"Día: {d_sel}", key=f"sub_final_{p_sel}")
        with col_style2:
            bg = st.color_picker("Fondo", "#FFFFFF", key=f"bg_final_{p_sel}")
            tx = st.color_picker("Texto", "#000000", key=f"tx_final_{p_sel}")
        
        lg = st.checkbox("Logo", True, key=f"lg_final_{p_sel}")
        
    if st.button("Generar Vista Previa", key=f"btn_gen_{p_sel}"):
        conf = {
            "title_text": tit, "subtitle_text": sub, "bg_color": bg, "title_color": tx, "use_logo": lg
        }
        if alignment_config:
            conf.update(alignment_config)

        current_seats = current_seats_dict if 'current_seats_dict' in locals() else {}
        generate_colored_plan(p_sel, d_sel, current_seats, "PNG", conf, global_logo_path)
        
        ds = d_sel.lower().replace("é", "e").replace("á", "a")
        fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
        if fpng.exists():
            st.image(str(fpng), caption="Vista Previa Generada", width=700)

def enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path, alignment_config=None):
    p_num = p_sel.replace("Piso ", "").strip()
    file_base = f"piso{p_num}"
    pim = PLANOS_DIR / f"{file_base}.png"
    if not pim.exists(): pim = PLANOS_DIR / f"{file_base}.jpg"
    if not pim.exists(): pim = PLANOS_DIR / f"Piso{p_num}.png"

    if not pim.exists():
        st.error(f"❌ No se encontró el plano base para {p_sel} en la carpeta 'planos/'.")
        return

    img = PILImage.open(pim)
    w, h = img.size
    
    # 1. Feedback de Cupos ANTES del editor (Solicitud del cliente)
    if not df_d.empty:
        subset = df_d[(df_d['piso'] == p_sel) & (df_d['dia'] == d_sel)]
        if not subset.empty:
            with st.expander("📋 Ver listado de cupos asignados para este día (Referencia)", expanded=False):
                st.dataframe(subset[['equipo', 'cupos']].sort_values('equipo'), hide_index=True, use_container_width=True)

    # 2. Llamada al editor manual
    fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, w, h, alignment_config)

# ---------------------------------------------------------
# INICIO APP
# ---------------------------------------------------------
conn = get_conn()
if "db_initialized" not in st.session_state:
    with st.spinner('Conectando a Google Sheets...'):
        init_db(conn)
    st.session_state["db_initialized"] = True

apply_appearance_styles(conn)
if "app_settings" not in st.session_state:
    st.session_state["app_settings"] = get_all_settings(conn)
settings = st.session_state["app_settings"]

site_title = settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios")
global_logo_path = settings.get("logo_path", "static/logo.png")

if os.path.exists(global_logo_path):
    c1, c2 = st.columns([1, 5])
    c1.image(global_logo_path, width=150)
    c2.title(site_title)
else:
    st.title(site_title)

# ---------------------------------------------------------
# MENÚ
# ---------------------------------------------------------
menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

# ==========================================
# A. VISTA PÚBLICA (VISUAL RESTAURADA Y MEJORADA)
# ==========================================
if menu == "Vista pública":
    st.header("Cupos y Planos")
    df = read_distribution_df(conn)
    
    if not df.empty:
        cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at']]
        df_view = df.drop(columns=cols_drop, errors='ignore')
        df_view = apply_sorting_to_df(df_view)
        pisos_disponibles = sort_floors(df["piso"].unique())
        
        # --- NUEVO: DASHBOARD DE MÉTRICAS (Para que no se vea vacío/feo) ---
        st.markdown("### 📊 Estado General de Distribución")
        km1, km2, km3 = st.columns(3)
        total_asignados = df_view['cupos'].sum()
        cupos_libres = df_view[df_view['equipo']=='Cupos libres']['cupos'].sum()
        with km1: st.metric("Total Cupos Asignados", int(total_asignados))
        with km2: st.metric("Total Cupos Libres (Semanal)", int(cupos_libres))
        with km3: st.metric("Pisos Gestionados", len(pisos_disponibles))
        st.markdown("---")
        
    else:
        df_view = df
        pisos_disponibles = ["Piso 1"]

    if df.empty: st.info("Sin datos cargados en el sistema.")
    else:
        t1, t2, t3 = st.tabs(["Estadísticas Generales", "Ver Planos y Descargas", "Resumen Semanal"])
        
        # TAB 1: ESTADÍSTICAS Y TABLAS (VISUALIZACIÓN ENRIQUECIDA)
        with t1:
            st.markdown("""<style>[data-testid="stElementToolbar"] {display: none;}</style>""", unsafe_allow_html=True)
            
            c_chart, c_table = st.columns([1, 1])
            
            lib = df_view[df_view["equipo"]=="Cupos libres"].groupby(["piso","dia"], as_index=True, observed=False).agg({"cupos":"sum"}).reset_index()
            lib = apply_sorting_to_df(lib)
            
            with c_chart:
                st.subheader("📈 Ocupación por Piso")
                # Gráfico simple de barras
                if not df_view.empty:
                    chart_data = df_view.groupby('piso')['cupos'].sum()
                    st.bar_chart(chart_data)
            
            with c_table:
                st.subheader("🟢 Disponibilidad Cupos Libres")
                st.dataframe(lib, hide_index=True, use_container_width=True)

            st.subheader("📋 Distribución Detallada")
            st.dataframe(df_view, hide_index=True, use_container_width=True)
        
        # TAB 2: PLANOS (FUNCIONALIDAD ORIGINAL MANTENIDA)
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            p_sel = c1.selectbox("Selecciona Piso", pisos_disponibles)
            ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS)
            pn = p_sel.replace("Piso ", "").strip()
            st.write("---")
            
            if ds == "Todos (Lunes a Viernes)":
                # Usar la función importada de pdfgen
                m = create_merged_pdf(p_sel, ORDER_DIAS, conn, read_distribution_df, global_logo_path, st.session_state.get('last_style_config', {}))
                if m: 
                    st.success("✅ Dossier disponible.")
                    st.download_button("📥 Descargar Semana (PDF)", m, f"Planos_{p_sel}_Semana.pdf", "application/pdf", use_container_width=True)
                else: st.warning("Sin planos generados para esta selección.")
            else:
                subset = df[(df['piso'] == p_sel) & (df['dia'] == ds)]
                current_seats = dict(zip(subset['equipo'], subset['cupos']))
                
                if not current_seats:
                    st.warning(f"No hay distribución definida para {p_sel} el día {ds}.")
                else:
                    day_config = st.session_state.get('last_style_config', {})
                    img_path = generate_colored_plan(p_sel, ds, current_seats, "PNG", day_config, global_logo_path)
                
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

        # TAB 3: RESUMEN SEMANAL
        with t3:
            st.subheader("Resumen de Uso Semanal por Equipo")
            weekly_summary = calculate_weekly_usage_summary(df_view)
            if not weekly_summary.empty:
                st.dataframe(weekly_summary, hide_index=True, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Equipos", len(weekly_summary))
                with col2:
                    total_cupos = weekly_summary['Total Cupos Semanales'].sum()
                    st.metric("Total Cupos Semanales", int(total_cupos))
                with col3:
                    avg_daily = weekly_summary['Promedio Diario'].mean()
                    st.metric("Promedio Diario General", f"{avg_daily:.1f}")
            else:
                st.info("No hay datos suficientes para generar el resumen semanal")

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
                hay_config = False; total_cupos = 0; disponibles = 0
                
                if not rg.empty:
                    hay_config = True
                    total_cupos = int(rg.iloc[0]["cupos"])
                    all_res = list_reservations_df(conn)
                    ocupados = 0
                    if not all_res.empty:
                        mask = (all_res["reservation_date"].astype(str) == str(fe)) & (all_res["piso"] == pi) & (all_res["team_area"] == "Cupos libres")
                        ocupados = len(all_res[mask])
                    disponibles = total_cupos - ocupados
                
                if not hay_config:
                    st.warning(f"⚠️ El {pi} no tiene habilitados 'Cupos libres' para los días {dn}.")
                else:
                    if disponibles > 0: st.success(f"✅ **HAY CUPO: Quedan {disponibles} puestos** (Total: {total_cupos}).")
                    else: st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")
                    
                    st.markdown("### Datos del Solicitante")
                    with st.form("form_puesto"):
                        cf1, cf2 = st.columns(2)
                        nm = cf1.text_input("Nombre Completo")
                        em = cf2.text_input("Correo Electrónico")
                        submitted = st.form_submit_button("Confirmar Reserva", type="primary", disabled=(disponibles <= 0))
                        
                        if submitted:
                            if not nm or not em: st.error("Por favor completa nombre y correo.")
                            elif user_has_reservation(conn, em, str(fe)): st.error("Ya tienes una reserva registrada para esta fecha.")
                            elif count_monthly_free_spots(conn, em, fe) >= 2: st.error("Has alcanzado tu límite de 2 reservas mensuales.")
                            elif disponibles <= 0: st.error("Lo sentimos, el cupo se acaba de agotar.")
                            else:
                                add_reservation(conn, nm, em, pi, str(fe), "Cupos libres", datetime.datetime.now(datetime.timezone.utc).isoformat())
                                msg = f"✅ Reserva Confirmada:\n\n- Usuario: {nm}\n- Fecha: {fe}\n- Piso: {pi}\n- Tipo: Puesto Flex"
                                st.success(msg)
                                send_reservation_email(em, "Confirmación Puesto", msg.replace("\n","<br>"))
                                st.rerun()

    elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
        st.subheader("Agendar Sala")
        c_sala, c_fecha = st.columns(2)
        
        salas_opciones = [
            "Sala Reuniones Pequeña Piso 1",
            "Sala Reuniones Grande Piso 1", 
            "Sala Reuniones Piso 2",
            "Sala Reuniones Piso 3"
        ]
        
        sl = c_sala.selectbox("Selecciona Sala", salas_opciones)
        
        if "Piso 1" in sl: pi_s = "Piso 1"
        elif "Piso 2" in sl: pi_s = "Piso 2" 
        elif "Piso 3" in sl: pi_s = "Piso 3"
        else: pi_s = "Piso 1"
        
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
                if not n_s: st.error("Falta el nombre.")
                elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, i, f):
                    st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
                else:
                    add_room_reservation(conn, n_s, e_s, pi_s, sl, str(fe_s), i, f, datetime.datetime.now(datetime.timezone.utc).isoformat())
                    msg = f"✅ Sala Confirmada:\n\n- Sala: {sl}\n- Fecha: {fe_s}\n- Horario: {i} - {f}"
                    st.success(msg)
                    if e_s: send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))

    elif opcion_reserva == "📋 Mis Reservas y Listados":
        st.subheader("Buscar y Cancelar mis reservas")
        q = st.text_input("Ingresa tu Correo o Nombre para buscar:")
        
        if q:
            dp = clean_reservation_df(list_reservations_df(conn), "puesto")
            if not dp.empty and 'Nombre' in dp.columns and 'Correo' in dp.columns:
                mp = dp[(dp['Nombre'].str.lower().str.contains(q.lower())) | (dp['Correo'].str.lower().str.contains(q.lower()))]
            else: mp = pd.DataFrame()

            ds = clean_reservation_df(get_room_reservations_df(conn), "sala")
            if not ds.empty and 'Nombre' in ds.columns and 'Correo' in ds.columns:
                ms = ds[(ds['Nombre'].str.lower().str.contains(q.lower())) | (ds['Correo'].str.lower().str.contains(q.lower()))]
            else: ms = pd.DataFrame()
            
            if mp.empty and ms.empty: st.warning("No encontré reservas con esos datos.")
            else:
                if not mp.empty:
                    st.markdown("#### 🪑 Tus Puestos")
                    for idx, r in mp.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([5, 1])
                            c1.markdown(f"**{r['Fecha Reserva']}** | {r['Piso']} (Cupo Libre)")
                            if c2.button("Anular", key=f"del_p_{idx}", type="primary"):
                                confirm_delete_dialog(conn, r['Nombre'], r['Fecha Reserva'], r['Ubicación'], r['Piso'])

                if not ms.empty:
                    st.markdown("#### 🏢 Tus Salas")
                    for idx, r in ms.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([5, 1])
                            c1.markdown(f"**{r['Fecha']}** | {r['Sala']} | {r['Inicio']} - {r['Fin']}")
                            if c2.button("Anular", key=f"del_s_{idx}", type="primary"):
                                confirm_delete_room_dialog(conn, r['Nombre'], r['Fecha'], r['Sala'], r['Inicio'])
        st.markdown("---")
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
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            if u == admin_user and p == admin_pass: 
                st.session_state["is_admin"] = True
                st.rerun()
            else: 
                st.error("Credenciales incorrectas")
        
        with st.expander("Recuperar Contraseña"):
            em_chk = st.text_input("Email Registrado")
            if st.button("Solicitar"):
                re = settings.get("admin_email","")
                if re and em_chk.lower() == re.lower():
                    t = generate_token()
                    save_reset_token(conn, t, (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat())
                    send_reservation_email(re, "Token", f"Token: {t}")
                    st.success("Enviado.")
                else: 
                    st.error("Email no coincide.")
            
            tk = st.text_input("Token")
            nu = st.text_input("Nuevo User")
            np = st.text_input("Nueva Pass", type="password")
            if st.button("Cambiar"):
                ok, m = validate_and_consume_token(conn, tk)
                if ok: 
                    save_setting(conn, "admin_user", nu)
                    save_setting(conn, "admin_pass", np)
                    st.success("OK")
                else: 
                    st.error(m)
        st.stop()

    if st.button("Cerrar Sesión"): 
        st.session_state["is_admin"] = False
        st.rerun()

    # --- TABS DE ADMINISTRACIÓN (INDENTACIÓN CORREGIDA) ---
    t1, t2, t3, t4, t5, t6 = st.tabs(["Excel", "Editor Visual", "Informes", "Config", "Apariencia", "Mantenimiento"])
    
    with t1:
        st.subheader("Generador de Distribución")
        up = st.file_uploader("Subir Excel", type=["xlsx"])
        
        # Opción para ignorar parámetros y usar lógica ideal
        ignore_params = st.checkbox("🎯 Ignorar hoja de parámetros (Distribución Ideal)", value=True, help="Genera distribuciones aleatorias equilibradas")
        
        if up:
            if st.button("Procesar"):
                try:
                    df_eq = pd.read_excel(up, "Equipos")
                    
                    if ignore_params:
                        # Generar 3 opciones ideales distintas
                        proposals = []
                        # Capacidad referencial (el algoritmo de seats.py usa esto)
                        cap_pisos = {"Piso 1": 50, "Piso 2": 50, "Piso 3": 50} 
                        
                        for i in range(1, 4):
                            # compute_ideal_distribution debe estar importada o disponible
                            rows, deficit = get_ideal_distribution_proposal(df_eq, variant=i*100)
                            libres = sum([r['cupos'] for r in rows if r['equipo']=='Cupos libres'])
                            
                            proposals.append({
                                'name': f"Opción {i} (Libres aprox/dia: {int(libres/5)})",
                                'rows': rows,
                                'deficit': deficit
                            })
                        
                        st.session_state['multiple_proposals'] = proposals
                        st.session_state['proposal_rows'] = proposals[0]['rows'] 
                        st.session_state['proposal_deficit'] = proposals[0]['deficit']
                    else:
                        # Lógica original
                        df_pa = pd.read_excel(up, "Parámetros")
                        rows, deficit = compute_distribution_from_excel(df_eq, df_pa)
                        st.session_state['proposal_rows'] = rows
                        st.session_state['proposal_deficit'] = deficit
                        st.session_state['multiple_proposals'] = []

                except Exception as e:
                    st.error(f"Error procesando el archivo: {e}")
        
        # Selección de opciones (si existen múltiples)
        if 'multiple_proposals' in st.session_state and st.session_state['multiple_proposals']:
            st.write("---")
            st.subheader("Selecciona una Opción:")
            cols = st.columns(len(st.session_state['multiple_proposals']))
            for idx, prop in enumerate(st.session_state['multiple_proposals']):
                if cols[idx].button(prop['name'], key=f"sel_prop_{idx}", use_container_width=True):
                    st.session_state['proposal_rows'] = prop['rows']
                    st.session_state['proposal_deficit'] = prop['deficit']
                    st.success(f"Seleccionada: {prop['name']}")

        if 'proposal_rows' in st.session_state:
            st.divider()
            st.subheader("Vista Previa")
            st.dataframe(pd.DataFrame(st.session_state['proposal_rows']), use_container_width=True)
            
            show_distribution_insights(st.session_state['proposal_rows'], st.session_state.get('proposal_deficit'))

            if st.button("💾 Guardar Distribución Definitiva", type="primary"):
                 clear_distribution(conn)
                 insert_distribution(conn, st.session_state['proposal_rows'])
                 st.success("¡Guardado exitosamente en la base de datos!")
                 st.balloons()

    with t2:
        zonas = load_zones()
        c1, c2 = st.columns(2)
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        p_sel = c1.selectbox("Piso", pisos_list)
        d_sel = c2.selectbox("Día Ref.", ORDER_DIAS)
        
        ce1, ce2 = st.columns(2)
        pos_leyenda = ce1.selectbox("Posición Leyenda", ["Izquierda", "Centro", "Derecha", "Oculta"])
        pos_logo = ce2.selectbox("Posición Logo", ["Izquierda", "Centro", "Derecha", "Oculto"])
        
        config_align = {"legend_align": pos_leyenda, "logo_align": pos_logo}
        enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path, config_align)

    with t3:
        st.subheader("Descargas")
        
        if st.button("📥 Descargar Datos Crudos (XLSX)"):
            b = BytesIO()
            try:
                import xlsxwriter
                with pd.ExcelWriter(b, engine='xlsxwriter') as w:
                    read_distribution_df(conn).to_excel(w, sheet_name="Distribucion", index=False)
                    list_reservations_df(conn).to_excel(w, sheet_name="Reservas", index=False)
                st.download_button("Descargar Excel", b.getvalue(), "data_sistema.xlsx")
            except ImportError:
                 st.error("Falta librería xlsxwriter. Instálala en requirements.txt")

        st.markdown("---")
        st.write("Reportes PDF:")
        # Nuevo generador completo con soporte admin
        if st.button("📄 Informe Completo (Admin)"):
            pdf_bytes = generate_full_pdf(
                read_distribution_df(conn),
                semanal_df=None,
                listado_reservas_df=list_reservations_df(conn),
                listado_salas_df=get_room_reservations_df(conn),
                logo_path="static/logo.png",
                is_admin=True 
            )
            st.download_button("Descargar PDF Admin", pdf_bytes, "reporte_admin.pdf", "application/pdf")

    with t4:
        nu = st.text_input("User Admin")
        np = st.text_input("Pass Admin", type="password")
        ne = st.text_input("Email Recuperación")
        if st.button("Guardar Credenciales"): 
            save_setting(conn, "admin_user", nu)
            save_setting(conn, "admin_pass", np)
            save_setting(conn, "admin_email", ne)
            st.success("Credenciales actualizadas")

    with t5:
        admin_appearance_ui(conn)

    with t6:
        st.subheader("Gestión de Reservas (Borrado Selectivo)")
        
        tab_puestos, tab_salas = st.tabs(["Puestos Flex", "Salas"])
        
        with tab_puestos:
            df_res = list_reservations_df(conn)
            if not df_res.empty:
                # Añadir columna checkbox
                df_res['Eliminar'] = False
                edited_df = st.data_editor(
                    df_res, 
                    column_config={"Eliminar": st.column_config.CheckboxColumn(required=True)},
                    disabled=[c for c in df_res.columns if c != "Eliminar"],
                    key="editor_puestos"
                )
                
                if st.button("🗑️ Eliminar Seleccionados (Puestos)", type="primary"):
                    to_delete = edited_df[edited_df['Eliminar'] == True]
                    if not to_delete.empty:
                        count = 0
                        for _, row in to_delete.iterrows():
                            # Borrado seguro
                            if delete_reservation_from_db(conn, row['user_name'], row['reservation_date'], row['team_area']):
                                count += 1
                        st.success(f"Eliminadas {count} reservas.")
                        st.rerun()
                    else:
                        st.warning("No seleccionaste nada.")
            else:
                st.info("No hay reservas de puestos.")
        
        with tab_salas:
            df_salas = get_room_reservations_df(conn)
            if not df_salas.empty:
                df_salas['Eliminar'] = False
                edited_df_s = st.data_editor(
                    df_salas, 
                    column_config={"Eliminar": st.column_config.CheckboxColumn(required=True)},
                    disabled=[c for c in df_salas.columns if c != "Eliminar"],
                    key="editor_salas"
                )
                
                if st.button("🗑️ Eliminar Seleccionados (Salas)", type="primary"):
                    to_delete_s = edited_df_s[edited_df_s['Eliminar'] == True]
                    if not to_delete_s.empty:
                        count = 0
                        for _, row in to_delete_s.iterrows():
                             if delete_room_reservation_from_db(conn, row['user_name'], row['reservation_date'], row['room_name'], row['start_time']):
                                count += 1
                        st.success(f"Eliminadas {count} reservas de sala.")
                        st.rerun()
            else:
                st.info("No hay reservas de salas.")

        st.divider()
        st.subheader("Borrado Masivo")
        opcion_borrado = st.selectbox("Selecciona qué borrar TODO:", ["Reservas", "Distribución", "Planos/Zonas", "TODO"])
        if st.button("Ejecutar Borrado Masivo"):
             perform_granular_delete(conn, opcion_borrado.upper())
             st.success("Borrado ejecutado.")
             st.rerun()
