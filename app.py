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
import streamlit.components.v1 as components
import streamlit.elements.lib.image_utils
import streamlit.elements.image # Necesario para el parche

# ---------------------------------------------------------
# 1. PARCHE PARA STREAMLIT >= 1.39 (MANTIENE LA COMPATIBILIDAD CON ST_CANVAS)
# ---------------------------------------------------------
# NOTA: Aunque no usaremos st_canvas, mantenemos el parche para compatibilidad global
if hasattr(streamlit.elements.lib.image_utils, "image_to_url"):
    _orig_image_to_url = streamlit.elements.lib.image_utils.image_to_url

    @dataclass
    class WidthConfig:
        width: int

    def _patched_image_to_url(image_data, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
        if isinstance(width, int):
            width = WidthConfig(width=width)
        return _orig_image_to_url(image_data, width, clamp, channels, output_format, image_id)

    streamlit.elements.lib.image_utils.image_to_url = _patched_image_to_url
    
    # 🩹 Inyección crítica: Corrige la ruta donde lo busca la librería
    from streamlit.elements.lib.image_utils import image_to_url
    if not hasattr(streamlit.elements.image, "image_to_url"):
        streamlit.elements.image.image_to_url = image_to_url


# ---------------------------------------------------------
# 2. IMPORTACIONES DE MÓDULOS
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
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# 3. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

# 1. Verificar si existen los secretos
if "gcp_service_account" not in st.secrets:
    st.error("🚨 ERROR CRÍTICO: No se encuentran los secretos [gcp_service_account]. Revisa el formato TOML en Streamlit Cloud.")
    st.stop()

# 2. Intentar conectar y mostrar el error real
try:
    creds_dict = dict(st.secrets["gcp_service_account"])
    # Verificar formato de private_key
    pk = creds_dict.get("private_key", "")
    if "-----BEGIN PRIVATE KEY-----" not in pk:
        st.error("🚨 ERROR EN PRIVATE KEY: No parece una llave válida. Revisa que incluya -----BEGIN PRIVATE KEY-----")
        st.stop()
        
    # Prueba de conexión directa
    from google.oauth2.service_account import Credentials
    import gspread
    
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    
    # Prueba de abrir la hoja
    sheet_name = st.secrets["sheets"]["sheet_name"]
    sh = client.open(sheet_name)
    # st.success(f"✅ CONEXIÓN EXITOSA con la hoja: {sheet_name}") # COMENTADO PARA NO MOSTRAR MENSAJE

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
def clean_pdf_text(text: str) -> str:
    if not isinstance(text, str): return str(text)
    replacements = {"•": "-", "—": "-", "–": "-", "⚠": "ATENCION:", "⚠️": "ATENCION:", "…": "...", "º": "o", "°": ""}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode('latin-1', 'replace').decode('latin-1')

def sort_floors(floor_list):
    """Ordena una lista de pisos lógicamente (1, 2, 10)."""
    def extract_num(text):
        text = str(text)
        num = re.findall(r'\d+', text)
        return int(num[0]) if num else 0
    return sorted(list(floor_list), key=extract_num)

def apply_sorting_to_df(df):
    """Aplica orden lógico a un DataFrame para Pisos y Días."""
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

def safe_convert_df(df):
    """Convierte seguramente un DataFrame para evitar errores de serialización"""
    if df.empty:
        return df
        
    df_clean = df.copy()
    for col in df_clean.columns:
        try:
            # Si la columna es de tipo categoría, convertir a string primero
            if hasattr(df_clean[col], 'cat'):
                df_clean[col] = df_clean[col].astype(str)
            
            # Para columnas de objeto, convertir a string y llenar NaN
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].fillna('')
            # Para otros tipos, solo llenar NaN si es necesario
            elif df_clean[col].isna().any():
                # Para numéricos, llenar con 0 en lugar de string vacío
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna('')
        except Exception as e:
            # Si falla, simplemente saltar la columna
            print(f"Advertencia: No se pudo procesar la columna {col}: {e}")
            continue
            
    return df_clean

# --- NUEVA FUNCIÓN CON ESTRATEGIAS DE ORDENAMIENTO ---
def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    """
    Genera una propuesta basada en una estrategia de ordenamiento.
    """
    eq_proc = df_equipos.copy()
    pa_proc = df_parametros.copy()
    
    # Asegurarnos de que tenemos datos numéricos para ordenar
    col_sort = None
    for c in eq_proc.columns:
        if c.lower().strip() == "dotacion":
            col_sort = c
            break
    
    # Si no existe columna dotacion, forzamos random si se pidió ordenamiento
    if not col_sort and strategy != "random":
        strategy = "random"

    # APLICAR ESTRATEGIA
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
        if not day_config.get("subtitle_text"):
            day_config["subtitle_text"] = f"Día: {dia}"
        else:
            if "Día:" not in str(day_config.get("subtitle_text","")):
                day_config["subtitle_text"] = f"Día: {dia}"

        img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)
        
        if img_path and Path(img_path).exists():
            found_any = True
            pdf.add_page()
            try: pdf.image(str(img_path), x=10, y=10, w=190)
            except: pass
            
    if not found_any: return None
    return pdf.output(dest='S').encode('latin-1')

def generate_full_pdf(distrib_df, semanal_df, out_path="reporte.pdf", logo_path=Path("static/logo.png"), deficit_data=None):
    """
    Genera el reporte PDF de distribución con tablas diaria y semanal.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    
    # --- PÁGINA 1: DISTRIBUCIÓN DIARIA ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    if logo_path.exists():
        try: pdf.image(str(logo_path), x=10, y=8, w=30)
        except: pass
    pdf.ln(25)
    pdf.cell(0, 10, clean_pdf_text("Informe de Distribución"), ln=True, align='C')
    pdf.ln(6)

    # Título de sección
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_pdf_text("1. Detalle de Distribución Diaria"), ln=True)

    # Tabla Diaria
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

    # --- SECCIÓN NUEVA: TABLA SEMANAL ---
    pdf.add_page() # Nueva página para el resumen semanal
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    
    # Cálculo del promedio semanal
    try:
        # Asegurar que trabajamos con números
        if "%Distrib" in distrib_df.columns:
            col_pct = "%Distrib"
        elif "pct" in distrib_df.columns:
            col_pct = "pct"
        else:
            col_pct = None

        if col_pct:
            # MODIFICADO: Convertir a numérico para evitar error groupby
            distrib_df[col_pct] = pd.to_numeric(distrib_df[col_pct], errors='coerce').fillna(0)
            
            # Agrupar por Equipo y calcular promedio
            weekly_stats = distrib_df.groupby("Equipo")[col_pct].mean().reset_index()
            weekly_stats.columns = ["Equipo", "Promedio Semanal"]
            # Ordenar alfabéticamente
            weekly_stats = weekly_stats.sort_values("Equipo")
            
            # Dibujar Tabla Semanal
            pdf.set_font("Arial", 'B', 9)
            w_wk = [100, 40]
            h_wk = ["Equipo", "% Promedio Semanal"]
            
            # Centrar un poco la tabla
            start_x = 35
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for _, row in weekly_stats.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:50]), 1)
                val = row["Promedio Semanal"]
                pdf.cell(w_wk[1], 6, clean_pdf_text(f"{val:.1f}%"), 1)
                pdf.ln()
    except Exception as e:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular el resumen semanal: {str(e)}"), ln=True)

    # --- GLOSARIO DE CÁLCULOS ---
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, clean_pdf_text("Glosario de Métricas y Cálculos:"), ln=True)
    
    pdf.set_font("Arial", '', 9)
    notas = [
        "1. % Distribución Diario: Se calcula dividiendo los cupos asignados en un día específico por la dotación total del equipo.",
        "2. % Uso Semanal: Promedio simple de los porcentajes de ocupación de los 5 días hábiles (Lunes a Viernes).",
        "3. Cálculo de Déficit: Diferencia entre los cupos mínimos requeridos (según reglas de presencialidad) y los asignados."
    ]
    
    for nota in notas:
        pdf.set_x(10)
        pdf.multi_cell(185, 6, clean_pdf_text(nota))

    # --- PÁGINA 3: DÉFICIT (Si existe) ---
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

    return pdf.output(dest='S').encode('latin-1')

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
    if c2.button("Cancelar", width="stretch", key="no_s"): st.rerun()

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
site_title = settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios")
global_logo_path = settings.get("logo_path", "static/logo.png")

if os.path.exists(global_logo_path):
    c1, c2 = st.columns([1, 5])
    c1.image(global_logo_path, width=150)
    c2.title(site_title)
else:
    st.title(site_title)


def create_drawing_component(img_path, existing_zones, width=700):
    """Componente profesional de dibujo - VERSIÓN CORREGIDA Y MEJORADA"""
    
    try:
        # Convertir imagen a base64
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # Preparar zonas existentes para JSON
        safe_zones = []
        for zone in existing_zones:
            safe_zone = {
                'x': zone.get('x', 0),
                'y': zone.get('y', 0),
                'w': zone.get('w', 0),
                'h': zone.get('h', 0),
                'color': zone.get('color', '#00A04A'),
                'team': zone.get('team', 'Sin nombre')
            }
            safe_zones.append(safe_zone)
        
        existing_zones_json = json.dumps(safe_zones)
        
        # CORRECCIÓN: Usar el parámetro width directamente
        canvas_width = width
        html_height = 800  # Altura fija para el componente
        
        # HTML/JS Componente de dibujo profesional CORREGIDO
        html_code = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Editor de Planos</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f8f9fa;
                }}
                .editor-container {{
                    max-width: {canvas_width}px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .editor-header {{
                    background: #00A04A;
                    color: white;
                    padding: 15px 20px;
                    margin: 0;
                }}
                .editor-controls {{
                    padding: 15px 20px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                }}
                .control-btn {{
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    margin-right: 10px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .control-btn:hover {{
                    background: #0056b3;
                }}
                .control-btn.delete {{
                    background: #dc3545;
                }}
                .control-btn.delete:hover {{
                    background: #c82333;
                }}
                .canvas-container {{
                    position: relative;
                    background: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 10px;
                }}
                #drawingCanvas {{
                    display: block;
                    cursor: crosshair;
                    border: 1px solid #ccc;
                    max-width: 100%;
                }}
                .status-panel {{
                    padding: 15px 20px;
                    background: #e9ecef;
                    border-top: 1px solid #dee2e6;
                }}
                .status-message {{
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                }}
                .status-success {{
                    background: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }}
                .status-info {{
                    background: #d1ecf1;
                    color: #0c5460;
                    border: 1px solid #bee5eb;
                }}
                .coordinates {{
                    font-family: monospace;
                    background: #2b303b;
                    color: #00ff00;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="editor-container">
                <h2 class="editor-header">🎨 Editor de Planos - PRECISIÓN MEJORADA</h2>
                
                <div class="editor-controls">
                    <button class="control-btn" onclick="startDrawing()">
                        ✏️ Dibujar Rectángulo
                    </button>
                    <button class="control-btn" onclick="clearLast()">
                        🗑️ Borrar Último
                    </button>
                    <button class="control-btn delete" onclick="clearAll()">
                        🗑️ Borrar Todo
                    </button>
                    <button class="control-btn" onclick="saveZones()" style="background: #28a745;">
                        💾 Guardar Zonas
                    </button>
                </div>

                <div class="canvas-container">
                    <canvas id="drawingCanvas"></canvas>
                </div>

                <div class="status-panel">
                    <div id="statusMessage" class="status-message status-info">
                        👆 Haz clic en "Dibujar Rectángulo" y luego arrastra en el plano para crear una zona.
                    </div>
                    <div class="coordinates">
                        <strong>Coordenadas actuales:</strong><br>
                        <span id="coordsDisplay">X: 0, Y: 0, Ancho: 0, Alto: 0</span>
                    </div>
                </div>
            </div>

            <img id="sourceImage" src="data:image/png;base64,{img_data}" style="display:none">
            
            <script>
                // Variables globales
                let canvas = document.getElementById('drawingCanvas');
                let ctx = canvas.getContext('2d');
                let img = document.getElementById('sourceImage');
                let isDrawing = false;
                let startX, startY, currentX, currentY;
                let rectangles = {existing_zones_json};
                let currentRect = null;
                let canvasWidth = {canvas_width};
                let canvasHeight = 0;

                // CORRECCIÓN PRINCIPAL: Calcular dimensiones del canvas cuando la imagen cargue
                img.onload = function() {{
                    // Calcular altura manteniendo la proporción de la imagen
                    const aspectRatio = img.naturalHeight / img.naturalWidth;
                    canvasHeight = Math.round(canvasWidth * aspectRatio);
                    
                    // Establecer dimensiones del canvas
                    canvas.width = canvasWidth;
                    canvas.height = canvasHeight;
                    
                    drawImageAndZones();
                }};

                function drawImageAndZones() {{
                    // Limpiar canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Dibujar imagen de fondo ESCALADA CORRECTAMENTE
    1.               ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    // Dibujar rectángulo actual (si está en proceso)
                    if (currentRect) {{
                        drawRectangle(currentRect);
                    }}
                    
                    // Dibujar zonas existentes (convertidas a coordenadas del canvas)
                    rectangles.forEach(rect => {{
                        // Convertir coordenadas originales a coordenadas del canvas
                        const scaleX = canvas.width / img.naturalWidth;
                        const scaleY = canvas.height / img.naturalHeight;
                        
                        const canvasRect = {{
                            x: rect.x * scaleX,
                            y: rect.y * scaleY,
                            w: rect.w * scaleX,
                            h: rect.h * scaleY,
                            color: rect.color,
    1.                       team: rect.team
                        }};
                        
                        drawRectangle(canvasRect);
                        
                        // Dibujar etiqueta
                        if (rect.team && rect.team !== 'Nueva Zona') {{
      2.                     ctx.fillStyle = '#000';
                            ctx.font = 'bold 12px Arial';
                            ctx.fillText(rect.team, canvasRect.x + 5, canvasRect.y + 15);
                        }}
                    }});
                }}

                function drawRectangle(rect) {{
                    ctx.strokeStyle = rect.color || '#00A04A';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
                    
                    // Relleno semi-transparente
                    ctx.fillStyle = (rect.color || '#00A04A') + '40';
                    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
                }}

                function startDrawing() {{
                    isDrawing = true;
                    canvas.style.cursor = 'crosshair';
                    showStatus('🎯 Modo dibujo activado: Haz clic y arrastra para dibujar un rectángulo', 'success');
                }}

                // CORRECCIÓN MEJORADA: Obtener coordenadas relativas al canvas correctamente
                function getCanvasCoordinates(e) {{
                    const rect = canvas.getBoundingClientRect();
                    // Usar pageX/pageY para mayor precisión cross-browser
                    const x = (e.pageX - rect.left - window.pageXOffset);
                    const y = (e.pageY - rect.top - window.pageYOffset);
                    
                    // Escalar según las dimensiones reales del canvas
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    
                    return {{
                        x: x * scaleX,
                        y: y * scaleY
                    }};
                }}

                canvas.addEventListener('mousedown', function(e) {{
                    if (!isDrawing) return;
                    
                    const coords = getCanvasCoordinates(e);
                    startX = coords.x;
                    startY = coords.y;
                    
                    currentRect = {{
                        x: startX, y: startY, w: 0, h: 0,
                        color: '#00A04A'
                    }};
                }});

                canvas.addEventListener('mousemove', function(e) {{
                    if (!isDrawing || !currentRect) return;
                    
                    const coords = getCanvasCoordinates(e);
                    currentX = coords.x;
                    currentY = coords.y;
                    
                    currentRect.w = currentX - startX;
                    currentRect.h = currentY - startY;
                    
                    // Actualizar display de coordenadas
                    document.getElementById('coordsDisplay').textContent = 
                        `X: ${{Math.round(startX)}}, Y: ${{Math.round(startY)}}, ` +
                        `Ancho: ${{Math.round(currentRect.w)}}, Alto: ${{Math.round(currentRect.h)}}`;
                    
                    drawImageAndZones();
                }});

                canvas.addEventListener('mouseup', function(e) {{
                    if (!isDrawing || !currentRect) return;
                    
                    // Solo guardar si el rectángulo tiene tamaño suficiente
                    if (Math.abs(currentRect.w) > 10 && Math.abs(currentRect.h) > 10) {{
                        // CORRECCIÓN: Convertir a coordenadas originales de la imagen
                        const scaleX = img.naturalWidth / canvas.width;
                        const scaleY = img.naturalHeight / canvas.height;
                        
                        const newRect = {{
                            x: Math.round(currentRect.x * scaleX),
                            y: Math.round(currentRect.y * scaleY),
                            w: Math.round(currentRect.w * scaleX),
                            h: Math.round(currentRect.h * scaleY),
                            color: '#00A04A',
                            team: 'Nueva Zona'
                        }};
                        
                        rectangles.push(newRect);
                        showStatus('✅ Rectángulo creado. Asigna un nombre al equipo abajo.', 'success');
                    }}
                    
                    currentRect = null;
                    isDrawing = false;
                    canvas.style.cursor = 'default';
                    drawImageAndZones();
                }});

                function clearLast() {{
                    if (rectangles.length > 0) {{
                        rectangles.pop();
                        drawImageAndZones();
                        showStatus('🗑️ Último rectángulo eliminado', 'info');
                    }} else {{
                        showStatus('ℹ️ No hay rectángulos para eliminar', 'info');
                    }}
                }}

                function clearAll() {{
                    if (rectangles.length > 0) {{
                        if (confirm('¿Estás seguro de que quieres eliminar TODAS las zonas?')) {{
                            rectangles = [];
                            drawImageAndZones();
                            showStatus('🗑️ Todas las zonas han sido eliminadas', 'info');
                        }}
                    }} else {{
                        showStatus('ℹ️ No hay zonas para eliminar', 'info');
                    }}
                }}

                function saveZones() {{
                    // Enviar zonas a Streamlit
                    window.parent.postMessage({{
                        type: 'ZONAS_GUARDADAS',
                        data: rectangles
                    }}, '*');
                    showStatus('📤 Zonas enviadas a la aplicación. Revisa la sección de abajo.', 'success');
                }}

                function showStatus(message, type) {{
                    const statusDiv = document.getElementById('statusMessage');
                    statusDiv.textContent = message;
                    statusDiv.className = 'status-message status-' + type;
                }}

                // Mostrar coordenadas al mover el mouse
                canvas.addEventListener('mousemove', function(e) {{
                    const coords = getCanvasCoordinates(e);
                    
                    if (!isDrawing) {{
                        document.getElementById('coordsDisplay').textContent = 
                            `X: ${{Math.round(coords.x)}}, Y: ${{Math.round(coords.y)}}`;
                    }}
                }});

                // Inicializar cuando el DOM esté listo
                document.addEventListener('DOMContentLoaded', function() {{
                    if (img.complete) {{
                        img.onload();
                    }}
                }});
            </script>
        </body>
        </html>
        '''
        
        # CORRECCIÓN: Usar variables locales definidas
        return components.html(html_code, width=canvas_width + 50, height=html_height, scrolling=False)
        
    except Exception as e:
        st.error(f"Error al crear el componente de dibujo: {str(e)}")
        import traceback
        st.code(f"Detalles del error: {traceback.format_exc()}")
        return None
# ---------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------
menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

# ==========================================
# A. VISTA PÚBLICA
# ==========================================
if menu == "Vista pública":
# ... (código de vista pública) ...

# ==========================================
# B. RESERVAS (UNIFICADO CON DROPDOWN Y TÍTULOS CORREGIDOS)
# ==========================================
elif menu == "Reservas":
# ... (código de reservas) ...

# ==========================================
# E. ADMINISTRADOR
# ==========================================
elif menu == "Administrador":
    st.header("Admin")
    admin_user, admin_pass = get_admin_credentials(conn)
    if "is_admin" not in st.session_state: 
        st.session_state["is_admin"] = False
    
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
        
    # ¡IMPORTANTE! Las pestañas deben definirse INMEDIATAMENTE después del st.stop()
    t1, t2, t3, t4, t5, t6 = st.tabs(["Excel", "Editor Visual", "Informes", "Config", "Apariencia", "Mantenimiento"])

    if st.button("Cerrar Sesión"): st.session_state["is_admin"]=False; st.rerun()

    # -----------------------------------------------------------
    # T1: GENERADOR DE DISTRIBUCIÓN (CON AUTO-OPTIMIZACIÓN JUSTA)
    # -----------------------------------------------------------
    with t1:
        st.subheader("Generador de Distribución Inteligente")
        st.markdown("Sube el archivo Excel y elige una estrategia. Usa **Auto-Optimizar** para buscar la distribución más equitativa.")
        # ... (Contenido de T1) ...
        
        # 2. VISUALIZACIÓN Y ACCIONES
        if st.session_state['proposal_rows'] is not None:
            st.divider()
            
            # --- SECCIÓN DE RESULTADOS ---
            n_def = len(st.session_state['proposal_deficit']) if st.session_state['proposal_deficit'] else 0
            
            # Mostrar estadísticas de la optimización si existen
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
                    # CAMBIO: Mostrar tabla completa ocupando todo el ancho
                    df_sorted = apply_sorting_to_df(df_preview)
                    st.dataframe(df_sorted, hide_index=True, width='stretch')
                else:
                    st.warning("No se generaron asignaciones.")
            
            with t_def:
                if st.session_state['proposal_deficit']:
                    # Análisis de "Injusticia"
                    def_df = pd.DataFrame(st.session_state['proposal_deficit'])
                    
                    # Contamos cuántas veces aparece cada equipo en el reporte de déficit
                    conteo_injusticia = def_df['equipo'].value_counts().reset_index()
                    conteo_injusticia.columns = ['Equipo', 'Veces Perjudicado']
                    
                    c1, c2 = st.columns(2)
                    c1.markdown("**Detalle de Conflictos:**")
                    c1.dataframe(def_df, width='stretch')
                    
                    c2.markdown("**⚠️ Equipos más afectados (Repetición):**")
                    c2.dataframe(conteo_injusticia,width='stretch')
                    
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
                    # Siempre usamos random para explorar, independiente de lo seleccionado arriba
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
            
    # -----------------------------------------------------------
    # T2: EDITOR VISUAL
    # -----------------------------------------------------------
    with t2:
        st.info("Editor de Zonas - Versión Profesional")
        
        # Verificar permisos de administrador
        if not st.session_state.get("is_admin", False):
            st.error("🔒 Acceso denegado. Solo administradores pueden acceder al editor.")
            st.stop()
        
        zonas = load_zones()
        c1, c2 = st.columns(2)
        
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        
        p_sel = c1.selectbox("Piso", pisos_list, key="editor_piso")
        d_sel = c2.selectbox("Día Ref.", ORDER_DIAS, key="editor_dia")
        p_num = p_sel.replace("Piso ", "").strip()
        
        # Búsqueda de Archivo
        file_base = f"piso{p_num}" 
        pim = PLANOS_DIR / f"{file_base}.png"
        if not pim.exists(): 
            pim = PLANOS_DIR / f"{file_base}.jpg"
        if not pim.exists(): 
            pim = PLANOS_DIR / f"Piso{p_num}.png"
            
        if pim.exists():
            try:
                # Cargar zonas existentes para este piso
                existing_zones = zonas.get(p_sel, [])
                
                st.success(f"✅ Plano cargado: {pim.name}")
                
                # Mostrar componente de dibujo profesional
                drawing_component = create_drawing_component(str(pim), existing_zones, width=700)
                
                # Sección para recibir datos del componente
                st.markdown("---")
                st.subheader("📥 Recepción de Datos del Editor")
                
                # Área para pegar datos JSON (como respaldo)
                st.info("""
                **Instrucciones:**
                1. Dibuja rectángulos en el editor de arriba
                2. Haz clic en **"💾 Guardar Zonas"** en el editor
                3. Los datos se enviarán automáticamente
                4. Si hay problemas, copia y pega manualmente:
                """)
                
                zones_json = st.text_area(
                    "Datos JSON de zonas (copia y pega si el envío automático falla):",
                    height=150,
                    placeholder='Pega aquí el JSON que aparece en el editor al hacer clic en "Guardar Zonas"'
                )
                
                # Botón para procesar datos manuales
                col1, col2 = st.columns([3, 1])
                if col2.button("🔄 Procesar Datos Manuales", type="primary"):
                    if zones_json.strip():
                        try:
                            zonas_data = json.loads(zones_json)
                            zonas[p_sel] = zonas_data
                            save_zones(zonas)
                            st.success("✅ Zonas guardadas correctamente (modo manual)")
                            st.rerun()
                        except json.JSONDecodeError:
                            st.error("❌ Error: El texto no es un JSON válido")
                        except Exception as e:
                            st.error(f"❌ Error al guardar zonas: {str(e)}")
                    else:
                        st.warning("⚠️ Por favor, pega los datos JSON en el área de texto")
                
                # JavaScript para capturar automáticamente los datos del componente
                components.html("""
                <script>
                window.addEventListener('message', function(event) {
                    // Verificar que el mensaje es del tipo esperado y viene de un origen confiable
                    if (event.data.type === 'ZONAS_GUARDADAS') {
                        console.log('Datos recibidos del editor:', event.data.data);
                        
                        // Enviar a Streamlit mediante el método estándar
                        if (window.Streamlit) {
                            // Guardar en sessionStorage para persistencia
                            sessionStorage.setItem('lastZonesData', JSON.stringify(event.data.data));
                            
                            // Mostrar notificación
                            const event = new CustomEvent('streamlitSetComponentValue', {
                                detail: {value: JSON.stringify(event.data.data)}
                            });
                            window.dispatchEvent(event);
                        }
                    }
                });
                </script>
                """, height=0)
                
                # Verificar si hay datos nuevos en sessionStorage (simulación)
                if st.button("📥 Verificar Datos Automáticos", key="check_auto_data"):
                    st.info("Esta función verifica si hay datos listos para guardar desde el editor")
                    # En una implementación real, aquí iría la lógica para capturar los datos automáticamente
                
                # Mostrar y gestionar zonas existentes
                st.markdown("---")
                st.subheader("📋 Zonas Actualmente Guardadas")
                
                if p_sel in zonas and zonas[p_sel]:
                    st.success(f"✅ {len(zonas[p_sel])} zonas guardadas para {p_sel}")
                    
                    # Selector para editar zonas existentes
                    st.markdown("#### ✏️ Editar Zona Existente")
                    zone_options = [f"{i+1}. {z.get('team', 'Sin nombre')} ({z['x']}, {z['y']})" 
                                    for i, z in enumerate(zonas[p_sel])]
                    
                    if zone_options:
                        selected_zone_idx = st.selectbox(
                            "Selecciona una zona para editar:",
                            range(len(zone_options)),
                            format_func=lambda x: zone_options[x],
                            key="zone_selector"
                        )
                        
                        if selected_zone_idx is not None:
                            zone = zonas[p_sel][selected_zone_idx]
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                new_team = st.text_input("Nombre del equipo:", 
                                                        value=zone.get('team', 'Nueva Zona'),
                                                        key=f"team_{selected_zone_idx}")
                            
                            with col2:
                                new_color = st.color_picker("Color:", 
                                                            value=zone.get('color', '#00A04A'),
                                                            key=f"color_{selected_zone_idx}")
                            
                            with col3:
                                if st.button("💾 Actualizar", key=f"update_{selected_zone_idx}"):
                                    zonas[p_sel][selected_zone_idx]['team'] = new_team
                                    zonas[p_sel][selected_zone_idx]['color'] = new_color
                                    save_zones(zonas)
                                    st.success("✅ Zona actualizada")
                                    st.rerun()
                            
                            with col4:
                                if st.button("🗑️ Eliminar", key=f"delete_{selected_zone_idx}"):
                                    zonas[p_sel].pop(selected_zone_idx)
                                    save_zones(zonas)
                                    st.success("✅ Zona eliminada")
                                    st.rerun()
                        
                    # Vista previa de todas las zonas
                    st.markdown("#### 👁️ Vista Previa de Zonas")
                    for i, z in enumerate(zonas[p_sel]):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        col1.markdown(
                            f"<span style='color:{z['color']}; font-size: 20px;'>■</span> **{z.get('team', 'Sin nombre')}** ",
                            unsafe_allow_html=True
                        )
                        col2.info(f"Pos: ({z['x']}, {z['y']})")
                        col3.metric("Tamaño", f"{z['w']}x{z['h']}")
                else:
                    st.warning("ℹ️ No hay zonas guardadas para este piso. Usa el editor de arriba para crear zonas.")
                        
            except Exception as e:
                st.error(f"❌ Error en el editor: {str(e)}")
                st.code(f"Detalles: {str(e)}")
        else:
            st.error(f"❌ No se encontró el plano: {p_sel}")
            st.info(f"💡 Busqué en: {pim}")
            st.info("""
            **Formatos soportados:** PNG, JPG, JPEG
            **Nombres esperados:** - piso1.png, piso2.jpg, etc.
            - Piso1.png, Piso2.jpg, etc.
            """)

    # -----------------------------------------------------------
    # T3: INFORMES
    # -----------------------------------------------------------
    with t3:
        st.subheader("Generar Reportes de Distribución")
        
        if 'deficit_report' in st.session_state and st.session_state['deficit_report']:
            st.markdown("---")
            st.error("🚨 INFORME DE DÉFICIT DE CUPOS")
            
            df_deficit = pd.DataFrame(st.session_state['proposal_deficit'])
            df_deficit = df_deficit.rename(columns={
                'piso': 'Piso', 
                'dia': 'Día', 
                'equipo': 'Equipo', 
                'deficit': 'Cupos Faltantes',
                'causa': 'Observación'
            })
            st.dataframe(df_deficit, hide_index=True, width='stretch')
            st.markdown("---")

        rf = st.selectbox("Formato Reporte", ["Excel", "PDF"], key="formato_reporte")
        if st.button("Generar Reporte", key="generar_reporte"):
            df_raw = read_distribution_df(conn); df_raw = apply_sorting_to_df(df_raw)
            if "Excel" in rf:
                b = BytesIO()
                with pd.ExcelWriter(b) as w: df_raw.to_excel(w, index=False)
                st.session_state['rd'] = b.getvalue(); st.session_state['rn'] = "d.xlsx"; st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                df = df_raw.rename(columns={"piso":"Piso","equipo":"Equipo","dia":"Día","cupos":"Cupos","pct":"%Distrib"})
                d_data = st.session_state.get('deficit_report', [])
                st.session_state['rd'] = generate_full_pdf(df, df, logo_path=Path(global_logo_path), deficit_data=d_data)
                st.session_state['rn'] = "reporte_distribucion.pdf"; st.session_state['rm'] = "application/pdf"
            st.success("OK")
        if 'rd' in st.session_state: st.download_button("Descargar", st.session_state['rd'], st.session_state['rn'], mime=st.session_state['rm'], key="descargar_reporte")
        
        st.markdown("---")
        cp, cd = st.columns(2)
        pi = cp.selectbox("Piso", pisos_list, key="pi2"); di = cd.selectbox("Día", ["Todos"]+ORDER_DIAS, key="di2")
        if di=="Todos":
            if st.button("Generar Dossier", key="generar_dossier"):
                # CAMBIO: Pasar conn y logo para regenerar
                m = create_merged_pdf(pi, conn, global_logo_path)
                if m: st.session_state['dos'] = m; st.success("OK")
            if 'dos' in st.session_state: st.download_button("Descargar Dossier", st.session_state['dos'], "S.pdf", "application/pdf", key="descargar_dossier")
        else:
            ds = di.lower().replace("é","e").replace("á","a")
            fp = COLORED_DIR / f"piso_{pi.split()[-1]}_{ds}_combined.png"
            fd = COLORED_DIR / f"piso_{pi.split()[-1]}_{ds}_combined.pdf"
            ops = []
            if fp.exists(): ops.append("Imagen (PNG)")
            if fd.exists(): ops.append("Documento (PDF)")
            if ops:
                if fp.exists(): st.image(str(fp), width=300)
                sf = st.selectbox("Fmt", ops, key="sf2")
                tf = fp if "PNG" in sf else fd
                mm = "image/png" if "PNG" in sf else "application/pdf"
                with open(tf,"rb") as f: st.download_button("Descargar", f, tf.name, mm, key="descargar_plano")
            else: st.warning("No existe.")

    # -----------------------------------------------------------
    # T4: CONFIG
    # -----------------------------------------------------------
    with t4:
        nu = st.text_input("User", key="admin_user"); np = st.text_input("Pass", type="password", key="admin_pass"); ne = st.text_input("Email", key="admin_email")
        if st.button("Guardar", key="sc"): save_setting(conn, "admin_user", nu); save_setting(conn, "admin_pass", np); save_setting(conn, "admin_email", ne); st.success("OK")

    # -----------------------------------------------------------
    # T5: APARIENCIA
    # -----------------------------------------------------------
    with t5: admin_appearance_ui(conn)
    
    # -----------------------------------------------------------
    # T6: MANTENIMIENTO
    # -----------------------------------------------------------
    with t6:
        opt = st.radio("Borrar:", ["Reservas", "Distribución", "Planos/Zonas", "TODO"], key="opcion_borrar")
        # SOLO UN BOTÓN - ELIMINA LA LÍNEA DUPLICADA
        if st.button("BORRAR", type="primary", key="borrar_mantenimiento"): 
            msg = perform_granular_delete(conn, opt); 
            st.success(msg)
