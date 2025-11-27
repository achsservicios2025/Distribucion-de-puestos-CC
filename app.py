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
from modules.seats import compute_distribution_from_excel
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones

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
# 5. FUNCIONES HELPER & LÓGICA
# ---------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    if not isinstance(text, str): return str(text)
    replacements = {"•": "-", "—": "-", "–": "-", "⚠": "ATENCION:", "⚠️": "ATENCION:", "…": "...", "º": "o", "°": ""}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode('latin-1', 'replace').decode('latin-1')

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

def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    eq_proc = df_equipos.copy()
    pa_proc = df_parametros.copy()
    col_sort = None
    for c in eq_proc.columns:
        if c.lower().strip() == "dotacion":
            col_sort = c
            break
    if not col_sort and strategy != "random": strategy = "random"

    if strategy == "random": eq_proc = eq_proc.sample(frac=1).reset_index(drop=True)
    elif strategy == "size_desc" and col_sort: eq_proc = eq_proc.sort_values(by=col_sort, ascending=False).reset_index(drop=True)
    elif strategy == "size_asc" and col_sort: eq_proc = eq_proc.sort_values(by=col_sort, ascending=True).reset_index(drop=True)

    rows, deficit_report = compute_distribution_from_excel(eq_proc, pa_proc, 2)
    return rows, deficit_report

# --- ESTRATEGIAS DE DISTRIBUCIÓN IDEAL ---
def get_ideal_distribution_proposal(df_equipos, strategy="perfect_equity", variant=0):
    df_eq_proc = df_equipos.copy()
    dotacion_col = None
    for col in df_eq_proc.columns:
        if col.lower() in ['dotacion', 'dotación', 'total', 'empleados']:
            dotacion_col = col
            break
    
    if dotacion_col is None:
        numeric_cols = df_eq_proc.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            dotacion_col = numeric_cols[0]
        else:
            dotacion_col = df_eq_proc.columns[1] if len(df_eq_proc.columns) > 1 else df_eq_proc.columns[0]
    
    equipo_col = None
    for col in df_eq_proc.columns:
        if col.lower() in ['equipo', 'team', 'departamento', 'área']:
            equipo_col = col
            break
    if equipo_col is None:
        equipo_col = df_eq_proc.columns[0]
    
    equipos = df_eq_proc[equipo_col].tolist()
    dotaciones = df_eq_proc[dotacion_col].tolist()
    
    if strategy == "perfect_equity":
        return perfect_equity_distribution(equipos, dotaciones, variant)
    elif strategy == "balanced_flex":
        return balanced_flex_distribution(equipos, dotaciones, variant)
    elif strategy == "controlled_random":
        return controlled_random_distribution(equipos, dotaciones, variant)
    else:
        return perfect_equity_distribution(equipos, dotaciones, variant)

def perfect_equity_distribution(equipos, dotaciones, variant=0):
    rows = []
    deficit_report = []
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        base_cupos = dotacion // 5
        resto = dotacion % 5
        for j, dia in enumerate(ORDER_DIAS):
            piso = pisos[i % len(pisos)]
            cupos_dia = base_cupos + (1 if j < resto else 0)
            rows.append({'piso': piso, 'equipo': equipo, 'dia': dia, 'cupos': cupos_dia, 'dotacion_total': dotacion})
    
    total_cupos = sum(dotaciones)
    cupos_libres = max(1, total_cupos // 50)
    for piso in pisos:
        for dia in ORDER_DIAS:
            rows.append({'piso': piso, 'equipo': "Cupos libres", 'dia': dia, 'cupos': cupos_libres, 'dotacion_total': cupos_libres * 5})
    return rows, deficit_report

def balanced_flex_distribution(equipos, dotaciones, variant=0):
    rows = []
    deficit_report = []
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        cupos_fijos = int(dotacion * 0.8)
        base_cupos = cupos_fijos // 5
        resto = cupos_fijos % 5
        for j, dia in enumerate(ORDER_DIAS):
            piso = pisos[(i + variant) % len(pisos)]
            cupos_dia = base_cupos + (1 if j < resto else 0)
            rows.append({'piso': piso, 'equipo': equipo, 'dia': dia, 'cupos': cupos_dia, 'dotacion_total': dotacion})
    
    total_cupos = sum(dotaciones)
    cupos_libres = max(2, total_cupos // 30)
    for piso in pisos:
        for dia in ORDER_DIAS:
            rows.append({'piso': piso, 'equipo': "Cupos libres", 'dia': dia, 'cupos': cupos_libres, 'dotacion_total': cupos_libres * 5})
    return rows, deficit_report

def controlled_random_distribution(equipos, dotaciones, variant=0):
    rows = []
    deficit_report = []
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    np.random.seed(variant * 1000)
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        cupos_rest = dotacion
        dist = [0]*5
        for j in range(5):
            dist[j] = dotacion // 5
            cupos_rest -= dist[j]
        for _ in range(cupos_rest):
            dist[np.random.randint(0, 5)] += 1
        
        for j, dia in enumerate(ORDER_DIAS):
            piso = pisos[np.random.randint(0, len(pisos))]
            rows.append({'piso': piso, 'equipo': equipo, 'dia': dia, 'cupos': dist[j], 'dotacion_total': dotacion})
    
    total_cupos = sum(dotaciones)
    cupos_libres = max(1, total_cupos // 40)
    for piso in pisos:
        for dia in ORDER_DIAS:
            rows.append({'piso': piso, 'equipo': "Cupos libres", 'dia': dia, 'cupos': cupos_libres, 'dotacion_total': cupos_libres * 5})
    return rows, deficit_report

def calculate_distribution_stats(rows, df_equipos):
    df = pd.DataFrame(rows)
    dotacion_map = {}
    equipo_col = None; dotacion_col = None
    for col in df_equipos.columns:
        if col.lower() in ['equipo', 'team', 'departamento']: equipo_col = col
        elif col.lower() in ['dotacion', 'dotación', 'total']: dotacion_col = col
    
    if equipo_col and dotacion_col:
        for _, row in df_equipos.iterrows():
            dotacion_map[row[equipo_col]] = row[dotacion_col]
            
    stats = {'total_cupos_asignados': df['cupos'].sum(), 'cupos_libres': df[df['equipo'] == 'Cupos libres']['cupos'].sum(), 'equipos_con_deficit': 0, 'distribucion_promedio': 0, 'uniformidad': 0}
    for eq in df['equipo'].unique():
        if eq == 'Cupos libres': continue
        ct = df[df['equipo'] == eq]['cupos'].sum()
        de = dotacion_map.get(eq, ct)
        if ct < de: stats['equipos_con_deficit'] += 1
    
    stats['uniformidad'] = df.groupby('dia')['cupos'].sum().std()
    return stats

def show_distribution_insights(rows, deficit_data):
    df = pd.DataFrame(rows)
    st.subheader("📊 Métricas de la Distribución")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Cupos Asignados", df['cupos'].sum())
    with c2: st.metric("Cupos Libres", df[df['equipo'] == 'Cupos libres']['cupos'].sum())
    with c3: st.metric("Equipos Asignados", df[df['equipo'] != 'Cupos libres']['equipo'].nunique())
    with c4: st.metric("Uniformidad (σ)", f"{df.groupby('dia')['cupos'].sum().std():.1f}")
    
    st.subheader("📈 Distribución por Día")
    cupos_por_dia = df.groupby('dia')['cupos'].sum().reindex(ORDER_DIAS)
    fig, ax = plt.subplots(figsize=(10, 4))
    cupos_por_dia.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel('Total Cupos')
    ax.set_title('Distribución de Cupos por Día de la Semana')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- CORRECCIÓN CLAVE PARA RESUMEN SEMANAL ---
def calculate_weekly_usage_summary(distrib_df):
    if distrib_df.empty: return pd.DataFrame()
    
    equipo_col = None; cupos_col = None; dia_col = None
    for col in distrib_df.columns:
        cl = col.lower()
        if 'equipo' in cl: equipo_col = col
        elif 'cupos' in cl: cupos_col = col
        elif 'dia' in cl or 'día' in cl: dia_col = col
    
    if not all([equipo_col, cupos_col, dia_col]):
        st.error("No se pudieron encontrar las columnas necesarias para el cálculo del resumen semanal")
        return pd.DataFrame()
    
    equipos_df = distrib_df[distrib_df[equipo_col] != "Cupos libres"]
    if equipos_df.empty: return pd.DataFrame()
    
    weekly = equipos_df.groupby(equipo_col).agg({cupos_col: 'sum', dia_col: 'count'}).reset_index()
    
    # --- AQUÍ ESTÁ EL ARREGLO: FORZAR NOMBRE 'Equipo' ---
    weekly.columns = ['Equipo', 'Total Cupos Semanales', 'Días Asignados']
    
    weekly['Promedio Diario'] = weekly['Total Cupos Semanales'] / weekly['Días Asignados']
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

# --- GENERADORES DE PDF ---
def create_merged_pdf(piso_sel, conn, global_logo_path):
    p_num = piso_sel.replace("Piso ", "").strip()
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    found_any = False
    df = read_distribution_df(conn)
    base_config = st.session_state.get('last_style_config', {})

    for dia in ORDER_DIAS:
        subset = df[(df['piso'] == piso_sel) & (df['dia'] == dia)]
        current_seats = dict(zip(subset['equipo'], subset['cupos']))
        day_config = base_config.copy()
        if not day_config.get("subtitle_text"): day_config["subtitle_text"] = f"Día: {dia}"
        
        img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)
        if img_path and Path(img_path).exists():
            found_any = True
            pdf.add_page()
            try: pdf.image(str(img_path), x=10, y=10, w=190)
            except: pass
    if not found_any: return None
    # CORRECCIÓN PDF
    try: return pdf.output(dest='S').encode('latin-1', 'replace')
    except: return pdf.output(dest='S')

def generate_full_pdf(distrib_df, semanal_df, out_path="reporte.pdf", logo_path=Path("static/logo.png"), deficit_data=None):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    if logo_path.exists():
        try: pdf.image(str(logo_path), x=10, y=8, w=30)
        except: pass
    pdf.ln(25)
    pdf.cell(0, 10, clean_pdf_text("Informe de Distribución"), ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_pdf_text("1. Detalle de Distribución Diaria"), ln=True)

    pdf.set_font("Arial", 'B', 9)
    widths = [30, 60, 25, 25, 25]
    headers = ["Piso", "Equipo", "Día", "Cupos", "%Distrib Diario"] 
    for w, h in zip(widths, headers): pdf.cell(w, 6, clean_pdf_text(h), 1)
    pdf.ln()

    pdf.set_font("Arial", '', 9)
    def get_val(row, keys):
        for k in keys:
            if k in row: return str(row[k])
            if k.lower() in row: return str(row[k.lower()])
        return ""

    distrib_df = apply_sorting_to_df(distrib_df)
    for _, r in distrib_df.iterrows():
        pdf.cell(widths[0], 6, clean_pdf_text(get_val(r, ["Piso", "piso"])), 1)
        pdf.cell(widths[1], 6, clean_pdf_text(get_val(r, ["Equipo", "equipo"])[:40]), 1)
        pdf.cell(widths[2], 6, clean_pdf_text(get_val(r, ["Día", "dia", "Dia"])), 1)
        pdf.cell(widths[3], 6, clean_pdf_text(get_val(r, ["Cupos", "cupos", "Cupos asignados"])), 1)
        pct_val = get_val(r, ["%Distrib", "pct"])
        pdf.cell(widths[4], 6, clean_pdf_text(f"{pct_val}%"), 1)
        pdf.ln()
    
    # --- TABLA SEMANAL ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    try:
        # Calcular resumen semanal
        weekly_summary = calculate_weekly_usage_summary(distrib_df)
        
        if not weekly_summary.empty:
            pdf.set_font("Arial", 'B', 9)
            w_wk = [80, 40, 40, 40]
            h_wk = ["Equipo", "Total Semanal", "Días Asignados", "Promedio Diario"]
            start_x = 20
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for _, row in weekly_summary.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:30]), 1)
                pdf.cell(w_wk[1], 6, clean_pdf_text(str(int(row["Total Cupos Semanales"]))), 1)
                pdf.cell(w_wk[2], 6, clean_pdf_text(str(int(row["Días Asignados"]))), 1)
                pdf.cell(w_wk[3], 6, clean_pdf_text(f"{row['Promedio Diario']:.1f}"), 1)
                pdf.ln()
        else:
            pdf.set_font("Arial", 'I', 9)
            pdf.cell(0, 6, clean_pdf_text("No hay datos suficientes para calcular el resumen semanal"), ln=True)
            
    except Exception as e:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular el resumen semanal: {str(e)}"), ln=True)

    # --- GLOSARIO ---
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, clean_pdf_text("Glosario de Métricas y Cálculos:"), ln=True)
    pdf.set_font("Arial", '', 9)
    notas = [
        "1. % Distribución Diario: Cupos asignados / Dotación total.",
        "2. % Uso Semanal: Promedio de los 5 días hábiles.",
        "3. Cálculo de Déficit: Diferencia entre cupos mínimos requeridos y asignados."
    ]
    for nota in notas:
        pdf.set_x(10)
        pdf.multi_cell(185, 6, clean_pdf_text(nota))

    # --- DÉFICIT ---
    if deficit_data and len(deficit_data) > 0:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 10, clean_pdf_text("Reporte de Déficit de Cupos"), ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 8) 
        dw = [15, 45, 20, 15, 15, 15, 65]
        dh = ["Piso", "Equipo", "Día", "Dot.", "Mín.", "Falt.", "Causa Detallada"]
        for w, h in zip(dw, dh): pdf.cell(w, 8, clean_pdf_text(h), 1, 0, 'C')
        pdf.ln()
        
        pdf.set_font("Arial", '', 8)
        for d in deficit_data:
            piso = clean_pdf_text(d.get('piso',''))
            equipo = clean_pdf_text(d.get('equipo',''))
            dia = clean_pdf_text(d.get('dia',''))
            dot = str(d.get('dotacion','-'))
            mini = str(d.get('minimo','-'))
            falt = str(d.get('deficit','-'))
            causa = clean_pdf_text(d.get('causa',''))

            line_height = 5
            lines_eq = pdf.multi_cell(dw[1], line_height, equipo, split_only=True)
            lines_ca = pdf.multi_cell(dw[6], line_height, causa, split_only=True)
            max_lines = max(len(lines_eq) if lines_eq else 1, len(lines_ca) if lines_ca else 1)
            row_height = max_lines * line_height

            if pdf.get_y() + row_height > 270:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 8)
                for w, h in zip(dw, dh): pdf.cell(w, 8, clean_pdf_text(h), 1, 0, 'C')
                pdf.ln()
                pdf.set_font("Arial", '', 8)

            y_start = pdf.get_y()
            x_start = pdf.get_x()

            pdf.cell(dw[0], row_height, piso, 1, 0, 'C')
            x_curr = pdf.get_x()
            pdf.multi_cell(dw[1], line_height, equipo, 1, 'L')
            pdf.set_xy(x_curr + dw[1], y_start)
            pdf.cell(dw[2], row_height, dia, 1, 0, 'C')
            pdf.cell(dw[3], row_height, dot, 1, 0, 'C')
            pdf.cell(dw[4], row_height, mini, 1, 0, 'C')
            pdf.set_font("Arial", 'B', 8)
            pdf.set_text_color(180, 0, 0)
            pdf.cell(dw[5], row_height, falt, 1, 0, 'C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 8)
            x_curr = pdf.get_x()
            pdf.multi_cell(dw[6], line_height, causa, 1, 'L')
            pdf.set_xy(x_start, y_start + row_height)
            
    # --- CORRECCIÓN PDF: Retorno seguro ---
    try:
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except Exception:
        # Fallback a bytes directos si falla la codificación
        return pdf.output(dest='S')

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

# --- EDITOR MANUAL ARREGLADO (TAMAÑO E IDENTIFICADORES) ---
def fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, img_width, img_height):
    """Editor manual con imagen ajustada y keys únicos"""
    
    p_num = p_sel.replace("Piso ", "").strip()

    st.subheader("🎯 Modo de Dibujo Manual")
    
    # 1. AJUSTAR TAMAÑO VISUAL DEL GRÁFICO (NO GIGANTE)
    fig, ax = plt.subplots(figsize=(10, 6)) # Tamaño controlado
    ax.imshow(img)
    # Quitar ejes para que se vea más limpio
    ax.axis('off')
    ax.set_title(f"Plano del {p_sel} (Referencia)", fontsize=10)
    
    if p_sel in zonas:
        for i, zona in enumerate(zonas[p_sel]):
            rect = plt.Rectangle((zona['x'], zona['y']), zona['w'], zona['h'],
                linewidth=2, edgecolor=zona['color'], facecolor=zona['color'] + '40')
            ax.add_patch(rect)
            ax.text(zona['x'], zona['y'], zona['team'], fontsize=8, color='white', 
                    bbox=dict(facecolor='black', alpha=0.5))
    
    st.pyplot(fig, use_container_width=False) # No expandir al 100%
    
    st.subheader("🖊️ Agregar Nueva Zona")
    
    # KEYS ÚNICOS BASADOS EN EL PISO Y DÍA
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
            
            # KEYS ÚNICOS
            equipo = st.selectbox("Equipo / Sala", eqs, key=f"team_sel_adv_{p_sel}")
            color = st.color_picker("Color", "#00A04A", key=f"col_pick_adv_{p_sel}")
            
            if equipo and equipo in current_seats_dict:
                st.info(f"Cupos: {current_seats_dict[equipo]}")
        
        with col2:
            st.info("📍 Coordenadas")
            c_x, c_y = st.columns(2)
            # KEYS ÚNICOS
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
                st.success("Zona guardada")
                st.rerun()
            else:
                st.warning("Selecciona un equipo")

    st.divider()
    st.subheader("🎨 Generar Imagen Final")
    with st.expander("Configuración"):
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            # KEYS ÚNICOS INPUTS
            tit = st.text_input("Título", f"Distribución {p_sel}", key=f"tit_final_{p_sel}")
            sub = st.text_input("Subtítulo", f"Día: {d_sel}", key=f"sub_final_{p_sel}")
        with col_style2:
            # KEYS ÚNICOS COLOR PICKERS
            bg = st.color_picker("Fondo", "#FFFFFF", key=f"bg_final_{p_sel}")
            tx = st.color_picker("Texto", "#000000", key=f"tx_final_{p_sel}")
        
        lg = st.checkbox("Logo", True, key=f"lg_final_{p_sel}")
        
    # KEY ÚNICA BOTÓN GENERAR
    if st.button("Generar Vista Previa", key=f"btn_gen_{p_sel}"):
        conf = {"title_text": tit, "subtitle_text": sub, "bg_color": bg, "title_color": tx, "use_logo": lg}
        current_seats = current_seats_dict if 'current_seats_dict' in locals() else {}
        generate_colored_plan(p_sel, d_sel, current_seats, "PNG", conf, global_logo_path)
        
        ds = d_sel.lower().replace("é", "e").replace("á", "a")
        fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
        if fpng.exists():
            st.image(str(fpng), caption="Vista Previa", width=700) # Ancho controlado

def enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path):
    p_num = p_sel.replace("Piso ", "").strip()
    file_base = f"piso{p_num}"
    pim = PLANOS_DIR / f"{file_base}.png"
    if not pim.exists(): pim = PLANOS_DIR / f"{file_base}.jpg"
    if not pim.exists(): pim = PLANOS_DIR / f"Piso{p_num}.png"

    if not pim.exists():
        st.error(f"❌ No se encontró el plano para {p_sel}")
        return

    img = PILImage.open(pim)
    w, h = img.size
    
    # Mostramos imagen de referencia con tamaño controlado
    st.image(img, caption=f"Plano Base {p_sel}", width=700) 
    
    try:
        from streamlit_image_annotation import image_annotation
        # ... (Tu código de anotación avanzada iría aquí si funcionara la librería)
        # Por seguridad y estabilidad, llamamos al manual mejorado directamente
        fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, w, h)
    except ImportError:
        fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, w, h)

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
# A. VISTA PÚBLICA
# ==========================================
if menu == "Vista pública":
    st.header("Cupos y Planos")
    df = read_distribution_df(conn)
    
    if not df.empty:
        cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at']]
        df_view = df.drop(columns=cols_drop, errors='ignore')
        df_view = apply_sorting_to_df(df_view)
        pisos_disponibles = sort_floors(df["piso"].unique())
    else:
        df_view = df
        pisos_disponibles = ["Piso 1"]

    if df.empty: st.info("Sin datos.")
    else:
        t1, t2, t3 = st.tabs(["Estadísticas", "Ver Planos", "Resumen Semanal"])
        with t1:
            st.markdown("""<style>[data-testid="stElementToolbar"] {display: none;}</style>""", unsafe_allow_html=True)
            lib = df_view[df_view["equipo"]=="Cupos libres"].groupby(["piso","dia"], as_index=True, observed=False).agg({"cupos":"sum"}).reset_index()
            lib = apply_sorting_to_df(lib)
            st.subheader("Distribución completa")
            st.dataframe(df_view, hide_index=True, use_container_width=True)
            st.subheader("Cupos libres por piso y día")
            st.dataframe(lib, hide_index=True, use_container_width=True)
        
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            p_sel = c1.selectbox("Selecciona Piso", pisos_disponibles)
            ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS)
            pn = p_sel.replace("Piso ", "").strip()
            st.write("---")
            
            if ds == "Todos (Lunes a Viernes)":
                m = create_merged_pdf(p_sel, conn, global_logo_path)
                if m: 
                    st.success("✅ Dossier disponible.")
                    st.download_button("📥 Descargar Semana (PDF)", m, f"Planos_{p_sel}_Semana.pdf", "application/pdf", use_container_width=True)
                else: st.warning("Sin planos generados.")
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

        with t3:
            st.subheader("Resumen de Uso Semanal por Equipo")
            weekly_summary = calculate_weekly_usage_summary(df_view)
            if not weekly_summary.empty:
                st.dataframe(weekly_summary, hide_index=True, use_container_width=True)
                
                # Mostrar métricas generales
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
        
        # ACTUALIZADO: Nuevas opciones de salas
        salas_opciones = [
            "Sala Reuniones Pequeña Piso 1",
            "Sala Reuniones Grande Piso 1", 
            "Sala Reuniones Piso 2",
            "Sala Reuniones Piso 3"
        ]
        
        sl = c_sala.selectbox("Selecciona Sala", salas_opciones)
        
        # Extraer piso basado en la sala seleccionada
        if "Piso 1" in sl:
            pi_s = "Piso 1"
        elif "Piso 2" in sl:
            pi_s = "Piso 2" 
        elif "Piso 3" in sl:
            pi_s = "Piso 3"
        else:
            pi_s = "Piso 1"  # Valor por defecto
        
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
            # CORRECCIÓN KEY ERROR: Limpiamos y normalizamos los DFs antes de filtrar
            dp = clean_reservation_df(list_reservations_df(conn), "puesto")
            # Buscar en columnas normalizadas 'Nombre' o 'Correo'
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
    
    with t1:
        st.subheader("Generador de Distribución Inteligente")
        c_up, c_strat = st.columns([2, 1])
        up = c_up.file_uploader("Subir archivo Excel (Hojas: 'Equipos', 'Parámetros')", type=["xlsx"])
        
        # NUEVO: Checkbox para ignorar parámetros
        ignore_params = st.checkbox("🎯 Ignorar hoja de parámetros y generar distribución ideal", 
                                       help="Genera distribuciones optimizadas sin restricciones de capacidad")
        
        if ignore_params:
            estrategia = st.radio("Estrategia de Distribución Ideal:", 
                                 ["⚖️ Equitativa Perfecta", "🔄 Balanceada con Flex", "🎲 Aleatoria Controlada"])
        else:
            estrategia = c_strat.radio("Estrategia Base:", ["🎲 Aleatorio (Recomendado)", "🧩 Tetris", "🐜 Relleno"])
        
        strat_map = {
            "🧩 Tetris": "size_desc", 
            "🎲 Aleatorio (Recomendado)": "random", 
            "🐜 Relleno": "size_asc",
            "⚖️ Equitativa Perfecta": "perfect_equity",
            "🔄 Balanceada con Flex": "balanced_flex",
            "🎲 Aleatoria Controlada": "controlled_random"
        }
        sel_strat_code = strat_map.get(estrategia, "random")

        if 'excel_equipos' not in st.session_state: st.session_state['excel_equipos'] = None
        if 'excel_params' not in st.session_state: st.session_state['excel_params'] = None
        if 'proposal_rows' not in st.session_state: st.session_state['proposal_rows'] = None
        if 'proposal_deficit' not in st.session_state: st.session_state['proposal_deficit'] = None
        if 'last_optimization_stats' not in st.session_state: st.session_state['last_optimization_stats'] = None
        # NUEVO: Para almacenar múltiples opciones
        if 'multiple_proposals' not in st.session_state: st.session_state['multiple_proposals'] = []

        if up:
            try:
                if st.button("📂 Procesar Inicial", type="primary"):
                    df_eq = pd.read_excel(up, "Equipos")
                    
                    if ignore_params:
                        # Generar múltiples propuestas ideales
                        st.session_state['excel_equipos'] = df_eq
                        st.session_state['excel_params'] = None
                        
                        # Generar 3 opciones diferentes
                        proposals = []
                        for i in range(3):
                            rows, deficit = get_ideal_distribution_proposal(df_eq, strategy=sel_strat_code, variant=i)
                            proposals.append({
                                'rows': rows,
                                'deficit': deficit,
                                'name': f"Opción {i+1} - {estrategia}",
                                'stats': calculate_distribution_stats(rows, df_eq)
                            })
                        
                        st.session_state['multiple_proposals'] = proposals
                        st.session_state['proposal_rows'] = proposals[0]['rows']
                        st.session_state['proposal_deficit'] = proposals[0]['deficit']
                        
                    else:
                        # Comportamiento original
                        df_pa = pd.read_excel(up, "Parámetros")
                        st.session_state['excel_equipos'] = df_eq
                        st.session_state['excel_params'] = df_pa
                        rows, deficit = get_distribution_proposal(df_eq, df_pa, strategy=sel_strat_code)
                        st.session_state['proposal_rows'] = rows
                        st.session_state['proposal_deficit'] = deficit
                        st.session_state['multiple_proposals'] = []  # Limpiar propuestas múltiples
                        
                    st.rerun()
                    
            except Exception as e: 
                st.error(f"Error al leer el Excel: {e}")

        if st.session_state['proposal_rows'] is not None:
            st.divider()
            
            # MOSTRAR OPCIONES MÚLTIPLES SI EXISTEN
            if st.session_state['multiple_proposals'] and len(st.session_state['multiple_proposals']) > 1:
                st.subheader("🎯 Opciones de Distribución Generadas")
                
                # Mostrar estadísticas comparativas
                cols = st.columns(len(st.session_state['multiple_proposals']))
                
                for idx, proposal in enumerate(st.session_state['multiple_proposals']):
                    with cols[idx]:
                        stats = proposal['stats']
                        st.metric(
                            label=proposal['name'],
                            value=f"{stats['total_cupos_asignados']} cupos",
                            delta=f"{stats['cupos_libres']} libres"
                        )
                        st.caption(f"Uniformidad: {stats['uniformidad']:.1f}")
                        st.caption(f"Déficits: {stats['equipos_con_deficit']}")
                        
                        if st.button(f"Seleccionar Opción {idx+1}", key=f"select_{idx}", use_container_width=True):
                            st.session_state['proposal_rows'] = proposal['rows']
                            st.session_state['proposal_deficit'] = proposal['deficit']
                            st.rerun()
                
                st.markdown("---")
            
            # CONTINUAR CON LA VISUALIZACIÓN NORMAL
            n_def = len(st.session_state['proposal_deficit']) if st.session_state['proposal_deficit'] else 0
            if n_def == 0: 
                st.success("✅ **¡Distribución Perfecta!** 0 conflictos detectados.")
            else: 
                st.warning(f"⚠️ **Distribución Actual:** {n_def} cupos faltantes en total.")

            t_view, t_def = st.tabs(["📊 Distribución Visual", "🚨 Reporte de Conflictos"])
            with t_view:
                df_preview = pd.DataFrame(st.session_state['proposal_rows'])
                if not df_preview.empty:
                    st.dataframe(apply_sorting_to_df(df_preview), hide_index=True, use_container_width=True)
                else: st.warning("No se generaron asignaciones.")
                
                # AGREGAR: Mostrar insights si es distribución ideal
                if st.session_state.get('multiple_proposals'):
                    show_distribution_insights(st.session_state['proposal_rows'], st.session_state['proposal_deficit'])
                    
            with t_def:
                if st.session_state['proposal_deficit']:
                    def_df = pd.DataFrame(st.session_state['proposal_deficit'])
                    st.dataframe(def_df, use_container_width=True)
                else: st.info("Sin conflictos.")

            st.markdown("---")
            c_actions = st.columns([1, 1, 1])
            if c_actions[0].button("🔄 Probar otra suerte"):
                with st.spinner("Generando..."):
                    if st.session_state.get('multiple_proposals'):
                        # En modo ideal, generar nuevas opciones
                        proposals = []
                        for i in range(3):
                            rows, deficit = get_ideal_distribution_proposal(
                                st.session_state['excel_equipos'], 
                                strategy=sel_strat_code, 
                                variant=i+3  # Cambiar variante para nuevas opciones
                            )
                            proposals.append({
                                'rows': rows,
                                'deficit': deficit,
                                'name': f"Opción {i+1} - {estrategia}",
                                'stats': calculate_distribution_stats(rows, st.session_state['excel_equipos'])
                            })
                        st.session_state['multiple_proposals'] = proposals
                        st.session_state['proposal_rows'] = proposals[0]['rows']
                        st.session_state['proposal_deficit'] = proposals[0]['deficit']
                    else:
                        # Comportamiento original
                        rows, deficit = get_distribution_proposal(
                            st.session_state['excel_equipos'], 
                            st.session_state['excel_params'], 
                            strategy=sel_strat_code
                        )
                        st.session_state['proposal_rows'] = rows
                        st.session_state['proposal_deficit'] = deficit
                st.rerun()
            
            if c_actions[1].button("✨ Auto-Optimizar"):
                if st.session_state.get('multiple_proposals'):
                    # En modo ideal, no aplica la optimización por déficit
                    st.info("En modo ideal, la distribución ya está optimizada.")
                else:
                    NUM_INTENTOS = 20; my_bar = st.progress(0, text="Optimizando...")
                    best_rows = None; best_deficit = None; min_unfairness_score = 999999
                    for i in range(NUM_INTENTOS):
                        r, d = get_distribution_proposal(st.session_state['excel_equipos'], st.session_state['excel_params'], strategy="random")
                        unfairness_score = sum([1 for x in d]) if d else 0
                        if unfairness_score < min_unfairness_score:
                            min_unfairness_score = unfairness_score; best_rows = r; best_deficit = d
                        my_bar.progress(int((i + 1) / NUM_INTENTOS * 100))
                    st.session_state['proposal_rows'] = best_rows; st.session_state['proposal_deficit'] = best_deficit
                    my_bar.empty(); st.rerun()

            if c_actions[2].button("💾 Guardar Definitivo", type="primary"):
                clear_distribution(conn); insert_distribution(conn, st.session_state['proposal_rows'])
                if st.session_state['proposal_deficit']: st.session_state['deficit_report'] = st.session_state['proposal_deficit']
                elif 'deficit_report' in st.session_state: del st.session_state['deficit_report']
                st.success("Guardado."); st.balloons(); st.rerun()

    with t2:
        # REEMPLAZADO: Usamos el editor simplificado en lugar del canvas problemático
        zonas = load_zones()
        c1, c2 = st.columns(2)
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        p_sel = c1.selectbox("Piso", pisos_list)
        d_sel = c2.selectbox("Día Ref.", ORDER_DIAS)
        
        # Llamar al editor simplificado
        enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path)

    with t3:
        st.subheader("Generar Reportes")
        rf = st.selectbox("Formato", ["Excel", "PDF"])
        if st.button("Generar"):
            df_raw = read_distribution_df(conn)
            if "Excel" in rf:
                b = BytesIO(); 
                with pd.ExcelWriter(b) as w: df_raw.to_excel(w, index=False)
                st.download_button("Descargar", b.getvalue(), "d.xlsx")
            else:
                d_data = st.session_state.get('deficit_report', [])
                pdf_bytes = generate_full_pdf(df_raw, df_raw, logo_path=Path(global_logo_path), deficit_data=d_data)
                st.download_button("Descargar PDF", pdf_bytes, "reporte.pdf", "application/pdf")
    
    with t4:
        nu = st.text_input("User"); np = st.text_input("Pass", type="password"); ne = st.text_input("Email")
        if st.button("Guardar Credenciales"): 
            save_setting(conn, "admin_user", nu); save_setting(conn, "admin_pass", np); save_setting(conn, "admin_email", ne)
            st.success("OK")

    with t5: 
        admin_appearance_ui(conn)
        
    with t6:
        st.subheader("Opciones de Mantenimiento")
        
        # ACTUALIZADO: Más opciones de borrado
        opcion_borrado = st.selectbox(
            "Selecciona qué deseas borrar:",
            ["Reservas", "Distribución", "Planos/Zonas", "TODO"]
        )
        
        if st.button("Ejecutar Borrado", type="primary"):
            if opcion_borrado == "TODO":
                perform_granular_delete(conn, "TODO")
                st.success("✅ Todo borrado exitosamente")
            elif opcion_borrado == "Reservas":
                perform_granular_delete(conn, "RESERVAS")
                st.success("✅ Reservas borradas exitosamente")
            elif opcion_borrado == "Distribución":
                perform_granular_delete(conn, "DISTRIBUCION")
                st.success("✅ Distribución borrada exitosamente")
            elif opcion_borrado == "Planos/Zonas":
                perform_granular_delete(conn, "ZONAS")
                st.success("✅ Planos y zonas borrados exitosamente")
            st.rerun()
        
        st.markdown("---")
        st.subheader("Resumen de Uso Semanal")
        df_distrib = read_distribution_df(conn)
        if not df_distrib.empty:
            weekly_summary = calculate_weekly_usage_summary(df_distrib)
            if not weekly_summary.empty:
                st.dataframe(weekly_summary, hide_index=True, use_container_width=True)
            else:
                st.info("No hay datos suficientes para el resumen semanal")
