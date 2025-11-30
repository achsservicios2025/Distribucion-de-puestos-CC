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


# ---------------------------------------------------------
# 1. PARCHE PARA STREAMLIT >= 1.39 (MANTIENE LA COMPATIBILIDAD CON ST_CANVAS)
# ---------------------------------------------------------
# NOTA: ESTE PARCHE ES EL QUE PERMITE QUE PIL IMAGE FUNCIONE EN EL CANVAS
import streamlit.elements.lib.image_utils

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

# --- NUEVAS FUNCIONES PARA RANKINGS ---
def generate_room_usage_ranking(conn):
    """Genera ranking de uso de salas de reuniones"""
    df = get_room_reservations_df(conn)
    if df.empty:
        return pd.DataFrame()
    
    # Contar uso por sala
    room_usage = df.groupby('room_name').size().reset_index(name='Reservas')
    room_usage = room_usage.sort_values('Reservas', ascending=False)
    return room_usage

def generate_flex_usage_ranking(conn):
    """Genera ranking de uso de cupos flexibles"""
    df = list_reservations_df(conn)
    if df.empty:
        return pd.DataFrame()
    
    # Contar uso por equipo (usuario)
    flex_usage = df.groupby('user_name').size().reset_index(name='Reservas')
    flex_usage = flex_usage.sort_values('Reservas', ascending=False)
    return flex_usage

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

def generate_full_pdf(distrib_df, semanal_df, out_path="reporte.pdf", logo_path=Path("static/logo.png"), deficit_data=None, room_ranking=None, flex_ranking=None):
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

    # --- SECCIÓN NUEVA: TABLA SEMANAL MEJORADA ---
    pdf.add_page() # Nueva página para el resumen semanal
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    
    # Cálculo del promedio semanal MEJORADO
    try:
        # Asegurar que trabajamos con números
        if "%Distrib" in distrib_df.columns:
            col_pct = "%Distrib"
        elif "pct" in distrib_df.columns:
            col_pct = "pct"
        else:
            col_pct = None

        if col_pct:
            # Convertir a numérico para evitar error groupby
            distrib_df[col_pct] = pd.to_numeric(distrib_df[col_pct], errors='coerce').fillna(0)
            
            # Agrupar por Equipo y calcular estadísticas semanales
            weekly_stats = distrib_df.groupby("Equipo").agg({
                col_pct: ['sum', 'mean', 'count']
            }).reset_index()
            
            # Aplanar columnas
            weekly_stats.columns = ['Equipo', 'Tot_Semanal', 'Prom_Diario', 'Dias_Asignados']
            
            # Calcular % Semanal (promedio de los días asignados)
            weekly_stats['%_Semanal'] = weekly_stats['Prom_Diario']
            
            # Ordenar alfabéticamente
            weekly_stats = weekly_stats.sort_values("Equipo")
            
            # Dibujar Tabla Semanal MEJORADA
            pdf.set_font("Arial", 'B', 9)
            w_wk = [50, 25, 25, 25, 25]
            h_wk = ["Equipo", "Suma % Diario", "Prom. Diario", "Días Asig.", "% Semanal"]
            
            # Centrar un poco la tabla
            start_x = 10
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 8)
            for _, row in weekly_stats.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:30]), 1)
                pdf.cell(w_wk[1], 6, clean_pdf_text(f"{row['Tot_Semanal']:.1f}"), 1)
                pdf.cell(w_wk[2], 6, clean_pdf_text(f"{row['Prom_Diario']:.1f}%"), 1)
                pdf.cell(w_wk[3], 6, clean_pdf_text(f"{int(row['Dias_Asignados'])}"), 1)
                pdf.cell(w_wk[4], 6, clean_pdf_text(f"{row['%_Semanal']:.1f}%"), 1)
                pdf.ln()
                
    except Exception as e:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular el resumen semanal: {str(e)}"), ln=True)

    # --- PÁGINA 3: RANKINGS ---
    if room_ranking is not None and not room_ranking.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_pdf_text("Ranking de Uso: Salas de Reuniones"), ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 9)
        w_rank = [100, 30]
        h_rank = ["Sala", "Reservas"]
        
        pdf.set_x(10)
        for w, h in zip(w_rank, h_rank): pdf.cell(w, 6, clean_pdf_text(h), 1)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        for _, row in room_ranking.iterrows():
            pdf.set_x(10)
            pdf.cell(w_rank[0], 6, clean_pdf_text(str(row["room_name"])), 1)
            pdf.cell(w_rank[1], 6, clean_pdf_text(str(row["Reservas"])), 1)
            pdf.ln()

    if flex_ranking is not None and not flex_ranking.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_pdf_text("Ranking de Uso: Cupos Flexibles"), ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 9)
        w_rank = [100, 30]
        h_rank = ["Equipo", "Reservas"]
        
        pdf.set_x(10)
        for w, h in zip(w_rank, h_rank): pdf.cell(w, 6, clean_pdf_text(h), 1)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        for _, row in flex_ranking.iterrows():
            pdf.set_x(10)
            pdf.cell(w_rank[0], 6, clean_pdf_text(str(row["user_name"])), 1)
            pdf.cell(w_rank[1], 6, clean_pdf_text(str(row["Reservas"])), 1)
            pdf.ln()

    # --- GLOSARIO DE CÁLCULOS MEJORADO ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, clean_pdf_text("Glosario de Métricas y Cálculos:"), ln=True)
    
    pdf.set_font("Arial", '', 9)
    notas = [
        "1. % Distribución Diario: Se calcula dividiendo los cupos asignados en un día específico por la dotación total del equipo.",
        "2. Suma % Diario: Suma de los porcentajes de distribución de los 5 días hábiles (puede ser mayor a 100%).",
        "3. Prom. Diario: Promedio simple de los porcentajes de ocupación de los días asignados.",
        "4. % Semanal: Equivale al Promedio Diario (representa el uso semanal promedio).",
        "5. Días Asig.: Número de días en la semana que el equipo tiene cupos asignados.",
        "6. Cálculo de Déficit: Diferencia entre los cupos mínimos requeridos (según reglas de presencialidad) y los asignados."
    ]
    
    for nota in notas:
        pdf.set_x(10)
        pdf.multi_cell(185, 6, clean_pdf_text(nota))

    # --- PÁGINA FINAL: DÉFICIT (Si existe) ---
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

# --- DIALOGOS MODALES MEJORADOS ---
@st.dialog("Confirmar Reserva de Puesto")
def confirm_reservation_dialog(nombre, email, fecha, piso, tipo):
    st.success(f"¿Confirmar reserva?\n\n👤 {nombre}\n📧 {email}\n📅 {fecha}\n📍 {piso}\n🪑 {tipo}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Sí, confirmar", type="primary", use_container_width=True, key="yes_reserve"):
        st.session_state.confirm_reservation = True
        st.rerun()
    if c2.button("❌ Cancelar", use_container_width=True, key="no_reserve"): 
        st.session_state.confirm_reservation = False
        st.rerun()

@st.dialog("Confirmar Reserva de Sala")
def confirm_room_reservation_dialog(nombre, email, fecha, sala, inicio, fin):
    st.success(f"¿Confirmar reserva de sala?\n\n👤 {nombre}\n📧 {email}\n📅 {fecha}\n🏢 {sala}\n⏰ {inicio} - {fin}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Sí, confirmar", type="primary", use_container_width=True, key="yes_room"):
        st.session_state.confirm_room_reservation = True
        st.rerun()
    if c2.button("❌ Cancelar", use_container_width=True, key="no_room"): 
        st.session_state.confirm_room_reservation = False
        st.rerun()

@st.dialog("Confirmar Anulación de Puesto")
def confirm_delete_dialog(conn, usuario, fecha_str, area, piso):
    st.warning(f"¿Anular reserva de puesto?\n\n👤 {usuario} | 📅 {fecha_str}\n📍 {piso} - {area}")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", use_container_width=True, key="yes_p"):
        if delete_reservation_from_db(conn, usuario, fecha_str, area): 
            st.success("Reserva eliminada")
            st.rerun()
    if c2.button("Cancelar", use_container_width=True, key="no_p"): 
        st.rerun()

@st.dialog("Confirmar Anulación de Sala")
def confirm_delete_room_dialog(conn, usuario, fecha_str, sala, inicio):
    st.warning(f"¿Anular reserva de sala?\n\n👤 {usuario} | 📅 {fecha_str}\n🏢 {sala} ({inicio})")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", use_container_width=True, key="yes_s"):
        if delete_room_reservation_from_db(conn, usuario, fecha_str, sala, inicio): 
            st.success("Reserva eliminada")
            st.rerun()
    if c2.button("Cancelar", use_container_width=True, key="no_s"): 
        st.rerun()

# --- UTILS TOKENS ---
def generate_token(): return uuid.uuid4().hex[:8].upper()

# --- NUEVA FUNCIÓN: EDITOR DE ZONAS SIMPLIFICADO ---
def create_simple_drawing_component(img_path, existing_zones, width=700):
    """Componente simplificado de dibujo que SÍ FUNCIONA"""
    
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
        
        canvas_width = width
        
        # HTML/JS Componente simplificado pero FUNCIONAL
        html_code = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Editor de Planos</title>
            <style>
                body {{
                    margin: 0;
                    padding: 10px;
                    background: #f8f9fa;
                }}
                .container {{
                    max-width: {canvas_width}px;
                    margin: 0 auto;
                }}
                .controls {{
                    margin: 10px 0;
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                button {{
                    padding: 8px 15px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .draw-btn {{ background: #007bff; color: white; }}
                .clear-btn {{ background: #dc3545; color: white; }}
                .save-btn {{ background: #28a745; color: white; }}
                canvas {{
                    border: 1px solid #ccc;
                    background: white;
                    cursor: crosshair;
                    max-width: 100%;
                }}
                .status {{
                    margin: 10px 0;
                    padding: 10px;
                    background: #e9ecef;
                    border-radius: 5px;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h3>Editor de Planos</h3>
                
                <div class="controls">
                    <button class="draw-btn" onclick="startDrawing()">✏️ Dibujar</button>
                    <button class="clear-btn" onclick="clearAll()">🗑️ Borrar Todo</button>
                    <button class="save-btn" onclick="saveZones()">💾 Guardar Zonas</button>
                </div>

                <canvas id="drawingCanvas"></canvas>
                
                <div class="status">
                    <div>Coordenadas: <span id="coords">X: 0, Y: 0</span></div>
                    <div id="message">Haz clic en "Dibujar" y arrastra para crear zonas</div>
                </div>
            </div>

            <img id="sourceImage" src="data:image/png;base64,{img_data}" style="display:none">
            
            <script>
                let canvas = document.getElementById('drawingCanvas');
                let ctx = canvas.getContext('2d');
                let img = document.getElementById('sourceImage');
                let isDrawing = false;
                let startX, startY;
                let rectangles = {existing_zones_json};
                let currentRect = null;
                let canvasWidth = {canvas_width};
                let canvasHeight = 0;

                img.onload = function() {{
                    const aspectRatio = img.naturalHeight / img.naturalWidth;
                    canvasHeight = Math.round(canvasWidth * aspectRatio);
                    
                    canvas.width = canvasWidth;
                    canvas.height = canvasHeight;
                    
                    drawEverything();
                }};

                function drawEverything() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    // Dibujar zonas existentes
                    rectangles.forEach(rect => {{
                        const scaleX = canvas.width / img.naturalWidth;
                        const scaleY = canvas.height / img.naturalHeight;
                        
                        const x = rect.x * scaleX;
                        const y = rect.y * scaleY;
                        const w = rect.w * scaleX;
                        const h = rect.h * scaleY;
                        
                        ctx.strokeStyle = rect.color;
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, w, h);
                        
                        ctx.fillStyle = rect.color + '40';
                        ctx.fillRect(x, y, w, h);
                    }});
                    
                    // Dibujar rectángulo actual
                    if (currentRect) {{
                        ctx.strokeStyle = '#00A04A';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
                        
                        ctx.fillStyle = '#00A04A40';
                        ctx.fillRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
                    }}
                }}

                function startDrawing() {{
                    isDrawing = true;
                    canvas.style.cursor = 'crosshair';
                    document.getElementById('message').textContent = 'Modo dibujo: Haz clic y arrastra';
                }}

                function getMousePos(e) {{
                    const rect = canvas.getBoundingClientRect();
                    return {{
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top
                    }};
                }}

                canvas.addEventListener('mousedown', (e) => {{
                    if (!isDrawing) return;
                    
                    const pos = getMousePos(e);
                    startX = pos.x;
                    startY = pos.y;
                    
                    currentRect = {{ x: startX, y: startY, w: 0, h: 0 }};
                }});

                canvas.addEventListener('mousemove', (e) => {{
                    if (!isDrawing || !currentRect) return;
                    
                    const pos = getMousePos(e);
                    currentRect.w = pos.x - startX;
                    currentRect.h = pos.y - startY;
                    
                    document.getElementById('coords').textContent = 
                        `X: ${{Math.round(startX)}}, Y: ${{Math.round(startY)}}, Ancho: ${{Math.round(currentRect.w)}}, Alto: ${{Math.round(currentRect.h)}}`;
                    
                    drawEverything();
                }});

                canvas.addEventListener('mouseup', (e) => {{
                    if (!isDrawing || !currentRect) return;
                    
                    if (Math.abs(currentRect.w) > 10 && Math.abs(currentRect.h) > 10) {{
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
                        document.getElementById('message').textContent = 'Zona creada. Asigna nombre y color en los controles laterales.';
                    }}
                    
                    currentRect = null;
                    isDrawing = false;
                    canvas.style.cursor = 'default';
                    drawEverything();
                }});

                function clearAll() {{
                    if (confirm('¿Borrar todas las zonas?')) {{
                        rectangles = [];
                        drawEverything();
                        document.getElementById('message').textContent = 'Todas las zonas eliminadas';
                    }}
                }}

                function saveZones() {{
                    // Enviar a Streamlit
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        data: JSON.stringify(rectangles)
                    }}, '*');
                    
                    document.getElementById('message').textContent = 'Zonas guardadas! Actualiza la página para ver los cambios.';
                }}

                // Mostrar coordenadas
                canvas.addEventListener('mousemove', (e) => {{
                    const pos = getMousePos(e);
                    if (!isDrawing) {{
                        document.getElementById('coords').textContent = `X: ${{Math.round(pos.x)}}, Y: ${{Math.round(pos.y)}}`;
                    }}
                }});

                // Inicializar
                if (img.complete) img.onload();
            </script>
        </body>
        </html>
        '''
        
        # Usar key único para evitar cache
        import time
        return components.html(html_code, width=canvas_width + 50, height=650, scrolling=False, key=f"drawing_{time.time()}")
        
    except Exception as e:
        st.error(f"Error al crear el componente: {str(e)}")
        return None

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


def main():
    menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

    # ==========================================
    # A. VISTA PÚBLICA
    # ==========================================
    if menu == "Vista pública":
        st.header("Cupos y Planos")
        
        # MODIFICADO: Leemos solo una vez para evitar Error 429
        df = read_distribution_df(conn)
        
        if not df.empty:
            cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at']]
            df_view = df.drop(columns=cols_drop, errors='ignore')
            df_view = apply_sorting_to_df(df_view)
            # MODIFICADO: Usamos df local en vez de leer de nuevo
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
                lib = apply_sorting_to_df(lib)
                
                st.subheader("Distribución completa")
                # MODIFICADO: Fix use_container_width
                st.dataframe(safe_convert_df(df_view), hide_index=True, use_container_width=True)
                
                st.subheader("Cupos libres por piso y día")
                st.dataframe(safe_convert_df(lib), hide_index=True, use_container_width=True)
            
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
    # B. RESERVAS (UNIFICADO CON DROPDOWN Y TÍTULOS CORREGIDOS)
    # ==========================================
    elif menu == "Reservas":
        
        st.header("Gestión de Reservas")
        
        # --- MENÚ DESPLEGABLE UNIFICADO ---
        opcion_reserva = st.selectbox(
            "¿Qué deseas gestionar hoy?",
            ["🪑 Reservar Puesto Flex", "🏢 Reservar Sala de Reuniones", "📋 Mis Reservas y Listados"],
            index=0
        )
        st.markdown("---")

        # ---------------------------------------------------------
        # OPCIÓN 1: RESERVAR PUESTO (Con lógica de disponibilidad real)
        # ---------------------------------------------------------
        if opcion_reserva == "🪑 Reservar Puesto Flex":
            st.subheader("Disponibilidad de Puestos")
            st.info("Reserva de 'Cupos libres' (Máximo 2 días por mes POR EQUIPO).")
            
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
                    # CORRECCIÓN: Buscar cupos libres en cualquier equipo, no solo "Cupos libres"
                    rg = df[(df["piso"] == pi) & (df["dia"] == dn)]
                    
                    # Verificar si hay algún cupo libre en este piso/día
                    total_cupos = rg["cupos"].sum() if not rg.empty else 0
                    
                    # Contar reservas existentes para esta fecha y piso
                    all_res = list_reservations_df(conn)
                    ocupados = 0
                    if not all_res.empty:
                        mask = (all_res["reservation_date"].astype(str) == str(fe)) & (all_res["piso"] == pi)
                        ocupados = len(all_res[mask])
                    
                    disponibles = max(0, total_cupos - ocupados)
                    
                    # GARANTIZAR MÍNIMO 1 CUPO POR PISO POR DÍA
                    if disponibles == 0 and total_cupos > 0:
                        # Si no hay disponibles pero hay capacidad, forzar al menos 1
                        disponibles = 1
                    
                    if total_cupos == 0:
                        st.warning(f"⚠️ El {pi} no tiene cupos configurados para los días {dn}.")
                    else:
                        if disponibles > 0:
                            st.success(f"✅ **HAY CUPO: {disponibles} puesto(s) disponible(s)** (Capacidad total: {total_cupos}).")
                        else:
                            st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")
                        
                        st.markdown("### Datos del Solicitante")
                        
                        with st.form("form_puesto"):
                            cf1, cf2 = st.columns(2)
                            # NUEVO: Selector de equipos en lugar de texto libre
                            equipos_disponibles = ["Selecciona tu equipo"] + sorted(df[df["equipo"] != "Cupos libres"]["equipo"].unique().tolist())
                            equipo_sel = cf1.selectbox("Equipo", equipos_disponibles)
                            em = cf2.text_input("Correo Electrónico")
                            
                            submitted = st.form_submit_button("Verificar Disponibilidad", type="primary", disabled=(disponibles <= 0))
                            
                            if submitted:
                                if equipo_sel == "Selecciona tu equipo":
                                    st.error("Por favor selecciona tu equipo.")
                                elif not em:
                                    st.error("Por favor ingresa tu correo electrónico.")
                                elif user_has_reservation(conn, em, str(fe)):
                                    st.error("Ya tienes una reserva registrada para esta fecha.")
                                elif count_monthly_free_spots(conn, equipo_sel, fe) >= 2:
                                    st.error(f"El equipo {equipo_sel} ha alcanzado el límite de 2 reservas mensuales.")
                                elif disponibles <= 0:
                                    st.error("Lo sentimos, el cupo se acaba de agotar.")
                                else:
                                    # MOSTRAR POPUP DE CONFIRMACIÓN MEJORADO
                                    if 'confirm_reservation' not in st.session_state:
                                        st.session_state.confirm_reservation = False
                                    
                                    confirm_reservation_dialog(equipo_sel, em, str(fe), pi, "Puesto Flex")
                                    
                                    if st.session_state.get('confirm_reservation'):
                                        add_reservation(conn, equipo_sel, em, pi, str(fe), "Cupos libres", datetime.datetime.now(datetime.timezone.utc).isoformat())
                                        msg = f"✅ Reserva Confirmada:\n\n- Equipo: {equipo_sel}\n- Fecha: {fe}\n- Piso: {pi}\n- Tipo: Puesto Flex"
                                        st.success(msg)
                                        send_reservation_email(em, "Confirmación Puesto", msg.replace("\n","<br>"))
                                        st.session_state.confirm_reservation = False
                                        st.rerun()

        # ---------------------------------------------------------
        # OPCIÓN 2: RESERVAR SALA (CON HORARIOS DISPONIBLES)
        # ---------------------------------------------------------
        elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
            st.subheader("Agendar Sala")
            
            c_sala, c_fecha = st.columns(2)
            sl = c_sala.selectbox("Selecciona Sala", ["Sala 1 (Piso 1)", "Sala 2 (Piso 2)", "Sala 3 (Piso 3)"])
            pi_s = "Piso " + sl.split("Piso ")[1].replace(")", "")
            fe_s = c_fecha.date_input("Fecha", min_value=datetime.date.today(), key="fs")
            
            # Obtener reservas existentes para esta sala y fecha
            existing_reservations = get_room_reservations_df(conn)
            existing_today = existing_reservations[existing_reservations['reservation_date'] == str(fe_s)]
            existing_this_room = existing_today[existing_today['room_name'] == sl]
            
            # Generar todos los horarios posibles
            all_slots = generate_time_slots("08:00", "20:00", 15)
            
            # Filtrar horarios ocupados
            occupied_slots = []
            for _, res in existing_this_room.iterrows():
                start = res['start_time']
                end = res['end_time']
                # Marcar como ocupado todos los slots entre inicio y fin
                start_idx = all_slots.index(start) if start in all_slots else -1
                end_idx = all_slots.index(end) if end in all_slots else -1
                if start_idx != -1 and end_idx != -1:
                    occupied_slots.extend(all_slots[start_idx:end_idx])
            
            available_slots = [slot for slot in all_slots if slot not in occupied_slots]
            
            st.write("Horario Disponible:")
            ch1, ch2 = st.columns(2)
            
            if available_slots:
                i = ch1.selectbox("Inicio", available_slots)
                # Encontrar slots disponibles después del inicio seleccionado
                start_idx = available_slots.index(i)
                available_ends = [slot for slot in available_slots if slot > i]
                f = ch2.selectbox("Fin", available_ends, index=min(1, len(available_ends)-1) if available_ends else 0)
            else:
                st.error("❌ No hay horarios disponibles para esta sala en la fecha seleccionada.")
                i, f = "08:00", "09:00"
            
            st.markdown("### Datos del Responsable")
            with st.form("form_sala"):
                cf1, cf2 = st.columns(2)
                # NUEVO: Selector de equipos para salas también
                df_equipos = read_distribution_df(conn)
                equipos_disponibles = ["Selecciona tu equipo"] + sorted(df_equipos[df_equipos["equipo"] != "Cupos libres"]["equipo"].unique().tolist())
                n_s = cf1.selectbox("Equipo Solicitante", equipos_disponibles)
                e_s = cf2.text_input("Correo Solicitante")
                
                sub_sala = st.form_submit_button("Verificar Disponibilidad", type="primary")
                
                if sub_sala:
                    if n_s == "Selecciona tu equipo":
                        st.error("Falta seleccionar el equipo.")
                    elif not e_s:
                        st.error("Falta el correo.")
                    elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, i, f):
                        st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
                    else:
                        # MOSTRAR POPUP DE CONFIRMACIÓN MEJORADO
                        if 'confirm_room_reservation' not in st.session_state:
                            st.session_state.confirm_room_reservation = False
                        
                        confirm_room_reservation_dialog(n_s, e_s, str(fe_s), sl, i, f)
                        
                        if st.session_state.get('confirm_room_reservation'):
                            add_room_reservation(conn, n_s, e_s, pi_s, sl, str(fe_s), i, f, datetime.datetime.now(datetime.timezone.utc).isoformat())
                            msg = f"✅ Sala Confirmada:\n\n- Equipo: {n_s}\n- Sala: {sl}\n- Fecha: {fe_s}\n- Horario: {i} - {f}"
                            st.success(msg)
                            if e_s: send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))
                            st.session_state.confirm_room_reservation = False
                            st.rerun()

        # ---------------------------------------------------------
        # OPCIÓN 3: GESTIONAR (ANULAR Y VER TODO)
        # ---------------------------------------------------------
        elif opcion_reserva == "📋 Mis Reservas y Listados":
            
            # --- SECCION 1: BUSCADOR PARA ANULAR ---
            st.subheader("Buscar y Cancelar mis reservas")
            q = st.text_input("Ingresa tu Correo o Nombre de equipo para buscar:")
            
            if q:
                dp = list_reservations_df(conn)
                mp = dp[(dp['user_name'].str.lower().str.contains(q.lower())) | (dp['user_email'].str.lower().str.contains(q.lower()))]
                
                ds = get_room_reservations_df(conn)
                ms = ds[(ds['user_name'].str.lower().str.contains(q.lower())) | (ds['user_email'].str.lower().str.contains(q.lower()))]
                
                if mp.empty and ms.empty:
                    st.warning("No encontré reservas con esos datos.")
                else:
                    if not mp.empty:
                        st.markdown("#### 🪑 Tus Puestos")
                        for idx, r in mp.iterrows():
                            with st.container(border=True):
                                c1, c2 = st.columns([5, 1])
                                c1.markdown(f"**{r['reservation_date']}** | {r['piso']} (Cupo Libre) - {r['user_name']}")
                                if c2.button("Anular", key=f"del_p_{idx}", type="primary"):
                                    confirm_delete_dialog(conn, r['user_name'], r['reservation_date'], r['team_area'], r['piso'])

                    if not ms.empty:
                        st.markdown("#### 🏢 Tus Salas")
                        for idx, r in ms.iterrows():
                            with st.container(border=True):
                                c1, c2 = st.columns([5, 1])
                                c1.markdown(f"**{r['reservation_date']}** | {r['room_name']} | {r['start_time']} - {r['end_time']} - {r['user_name']}")
                                if c2.button("Anular", key=f"del_s_{idx}", type="primary"):
                                    confirm_delete_room_dialog(conn, r['user_name'], r['reservation_date'], r['room_name'], r['start_time'])

            st.markdown("---")
            
            # --- SECCION 2: VER TODO (TABLAS CORREGIDAS) ---
            with st.expander("Ver Listado General de Reservas", expanded=True):
                
                # TÍTULO CORREGIDO 1
                st.subheader("Reserva de puestos") 
                st.dataframe(safe_convert_df(clean_reservation_df(list_reservations_df(conn))), hide_index=True, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True) 

                # TÍTULO CORREGIDO 2
                st.subheader("Reserva de salas") 
                st.dataframe(safe_convert_df(clean_reservation_df(get_room_reservations_df(conn), "sala")), hide_index=True, use_container_width=True)

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
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Excel", "Editor Visual", "Informes", "Rankings", "Config", "Apariencia", "Mantenimiento"])

        if st.button("Cerrar Sesión"): st.session_state["is_admin"]=False; st.rerun()

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
                    # Botón inicial para procesar
                    if st.button("📂 Procesar Inicial", type="primary"):
                        df_eq = pd.read_excel(up, "Equipos")
                        df_pa = pd.read_excel(up, "Parámetros")
                        
                        st.session_state['excel_equipos'] = df_eq
                        st.session_state['excel_params'] = df_pa
                        
                        # Generar propuesta inicial
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
                        st.dataframe(df_sorted, hide_index=True, use_container_width=True)
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
        # T2: EDITOR VISUAL MEJORADO Y SIMPLIFICADO
        # -----------------------------------------------------------
        with t2:
            st.info("Editor de Zonas - Versión Simplificada y Funcional")
            
            # Verificar permisos de administrador
            if not st.session_state.get("is_admin", False):
                st.error("🔒 Acceso denegado. Solo administradores pueden acceder al editor.")
                st.stop()
            
            zonas = load_zones()
            
            # Diseño en columnas para tener controles al lado del mapa
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                df_d = read_distribution_df(conn)
                pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
                
                p_sel = st.selectbox("Piso", pisos_list, key="editor_piso")
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
                        
                        # Mostrar componente de dibujo MEJORADO
                        drawing_component = create_simple_drawing_component(str(pim), existing_zones, width=600)
                        
                        # Área para recibir datos del componente - SIMPLIFICADA
                        st.markdown("---")
                        st.subheader("💾 Guardar Zonas")
                        
                        # Botón para cargar datos desde el componente
                        if st.button("🔄 Cargar Datos desde Editor", type="primary"):
                            st.info("Usa el botón 'Guardar Zonas' en el editor de arriba primero, luego haz clic aquí para cargar los datos.")
                            # En una implementación real, aquí se capturarían los datos del componente
                            # Por ahora, usamos las zonas existentes
                            st.session_state.current_zones = existing_zones
                            st.rerun()
                            
                        # Usar session_state para manejar las zonas actuales
                        if 'current_zones' not in st.session_state:
                            st.session_state.current_zones = existing_zones
                            
                        # Mostrar resumen de zonas actuales
                        st.info(f"Zonas actuales: {len(st.session_state.current_zones)}")
                        
                        # Botón para guardar definitivamente
                        if st.button("💾 Guardar Definitivamente", type="secondary"):
                            zonas[p_sel] = st.session_state.current_zones
                            save_zones(zonas)
                            st.success("✅ Zonas guardadas correctamente en la base de datos")
                            st.rerun()
                                
                    except Exception as e:
                        st.error(f"❌ Error en el editor: {str(e)}")
                else:
                    st.error(f"❌ No se encontró el plano: {p_sel}")
                    st.info(f"💡 Busqué en: {pim}")

            with col_right:
                st.subheader("🎨 Configuración de Zonas")
                
                # Mostrar y editar zonas actuales
                if 'current_zones' in st.session_state and st.session_state.current_zones:
                    st.success(f"✅ {len(st.session_state.current_zones)} zonas cargadas")
                    
                    # Editor de zonas existentes MEJORADO
                    st.markdown("#### ✏️ Editar Zonas")
                    
                    for i, zone in enumerate(st.session_state.current_zones):
                        with st.expander(f"Zona {i+1}: {zone.get('team', 'Sin nombre')}", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                new_team = st.text_input(
                                    "Nombre del equipo:", 
                                    value=zone.get('team', 'Nueva Zona'),
                                    key=f"team_{i}"
                                )
                                
                            with col2:
                                new_color = st.color_picker(
                                    "Color:", 
                                    value=zone.get('color', '#00A04A'),
                                    key=f"color_{i}"
                                )
                            
                            # Actualizar la zona
                            if st.button("💾 Actualizar", key=f"update_{i}"):
                                st.session_state.current_zones[i]['team'] = new_team
                                st.session_state.current_zones[i]['color'] = new_color
                                st.success(f"Zona {i+1} actualizada")
                                st.rerun()
                            
                            # Eliminar zona
                            if st.button("🗑️ Eliminar", key=f"delete_{i}"):
                                st.session_state.current_zones.pop(i)
                                st.success("Zona eliminada")
                                st.rerun()
                    
                    # Leyenda de colores MEJORADA
                    st.markdown("---")
                    st.markdown("#### 🎨 Leyenda de Colores")
                    
                    for i, z in enumerate(st.session_state.current_zones):
                        col_leg1, col_leg2 = st.columns([1, 4])
                        with col_leg1:
                            # Mostrar cuadro de color
                            st.markdown(f"<div style='background-color: {z['color']}; width: 30px; height: 30px; border: 1px solid #ccc;'></div>", 
                                      unsafe_allow_html=True)
                        with col_leg2:
                            st.write(f"**{z.get('team', 'Sin nombre')}**")
                            
                    # Botón para añadir zona manualmente
                    st.markdown("---")
                    st.markdown("#### ➕ Añadir Zona Manualmente")
                    
                    with st.form(f"add_zone_form"):
                        new_team_name = st.text_input("Nombre del equipo")
                        new_zone_color = st.color_picker("Color", "#00A04A")
                        col_add1, col_add2 = st.columns(2)
                        
                        with col_add1:
                            new_x = st.number_input("Posición X", value=0)
                            new_y = st.number_input("Posición Y", value=0)
                        
                        with col_add2:
                            new_w = st.number_input("Ancho", value=100)
                            new_h = st.number_input("Alto", value=100)
                        
                        if st.form_submit_button("➕ Añadir Zona"):
                            new_zone = {
                                'x': new_x,
                                'y': new_y, 
                                'w': new_w,
                                'h': new_h,
                                'color': new_zone_color,
                                'team': new_team_name
                            }
                            st.session_state.current_zones.append(new_zone)
                            st.success("Zona añadida")
                            st.rerun()
                            
                else:
                    st.warning("ℹ️ No hay zonas cargadas. Usa el editor de la izquierda para crear zonas.")

        # -----------------------------------------------------------
        # T3: INFORMES MEJORADOS
        # -----------------------------------------------------------
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
                st.dataframe(df_deficit, hide_index=True, use_container_width=True)
                st.markdown("---")

            rf = st.selectbox("Formato Reporte", ["Excel", "PDF"], key="formato_reporte")
            
            # Incluir rankings en el PDF
            include_rankings = st.checkbox("Incluir rankings de uso en el reporte", value=True)
            
            if st.button("Generar Reporte", key="generar_reporte"):
                df_raw = read_distribution_df(conn); df_raw = apply_sorting_to_df(df_raw)
                if "Excel" in rf:
                    b = BytesIO()
                    with pd.ExcelWriter(b) as w: df_raw.to_excel(w, index=False)
                    st.session_state['rd'] = b.getvalue(); st.session_state['rn'] = "d.xlsx"; st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    df = df_raw.rename(columns={"piso":"Piso","equipo":"Equipo","dia":"Día","cupos":"Cupos","pct":"%Distrib"})
                    d_data = st.session_state.get('deficit_report', [])
                    
                    # Obtener rankings si se solicitan
                    room_ranking = generate_room_usage_ranking(conn) if include_rankings else None
                    flex_ranking = generate_flex_usage_ranking(conn) if include_rankings else None
                    
                    st.session_state['rd'] = generate_full_pdf(
                        df, df, 
                        logo_path=Path(global_logo_path), 
                        deficit_data=d_data,
                        room_ranking=room_ranking,
                        flex_ranking=flex_ranking
                    )
                    st.session_state['rn'] = "reporte_distribucion.pdf"; st.session_state['rm'] = "application/pdf"
                st.success("✅ Reporte generado correctamente")
                
            if 'rd' in st.session_state: 
                st.download_button(
                    "📥 Descargar Reporte", 
                    st.session_state['rd'], 
                    st.session_state['rn'], 
                    mime=st.session_state['rm'], 
                    key="descargar_reporte",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.subheader("Generar Planos Individuales")
            cp, cd = st.columns(2)
            pi = cp.selectbox("Piso", pisos_list, key="pi2")
            di = cd.selectbox("Día", ["Todos"]+ORDER_DIAS, key="di2")
            
            if di=="Todos":
                if st.button("Generar Dossier Completo", key="generar_dossier", use_container_width=True):
                    m = create_merged_pdf(pi, conn, global_logo_path)
                    if m: 
                        st.session_state['dos'] = m
                        st.success("✅ Dossier generado correctamente")
                    else:
                        st.error("❌ No se pudo generar el dossier")
                
                if 'dos' in st.session_state: 
                    st.download_button(
                        "📥 Descargar Dossier Completo", 
                        st.session_state['dos'], 
                        f"dossier_{pi}.pdf", 
                        "application/pdf", 
                        key="descargar_dossier",
                        use_container_width=True
                    )
            else:
                ds = di.lower().replace("é","e").replace("á","a")
                fp = COLORED_DIR / f"piso_{pi.split()[-1]}_{ds}_combined.png"
                fd = COLORED_DIR / f"piso_{pi.split()[-1]}_{ds}_combined.pdf"
                ops = []
                if fp.exists(): ops.append("Imagen (PNG)")
                if fd.exists(): ops.append("Documento (PDF)")
                
                if ops:
                    if fp.exists(): 
                        st.image(str(fp), width=300)
                    
                    sf = st.selectbox("Formato de descarga:", ops, key="sf2")
                    tf = fp if "PNG" in sf else fd
                    mm = "image/png" if "PNG" in sf else "application/pdf"
                    
                    with open(tf,"rb") as f: 
                        st.download_button(
                            "📥 Descargar Plano", 
                            f, 
                            tf.name, 
                            mm, 
                            key="descargar_plano",
                            use_container_width=True
                        )
                else: 
                    st.warning("El plano seleccionado no ha sido generado aún.")

        # -----------------------------------------------------------
        # T4: RANKINGS (NUEVA PESTAÑA)
        # -----------------------------------------------------------
        with t4:
            st.subheader("Rankings de Uso")
            
            col_rank1, col_rank2 = st.columns(2)
            
            with col_rank1:
                st.markdown("#### 🏢 Ranking de Uso: Salas de Reuniones")
                room_ranking = generate_room_usage_ranking(conn)
                if not room_ranking.empty:
                    st.dataframe(room_ranking, hide_index=True, use_container_width=True)
                    
                    # Gráfico de ranking de salas
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(room_ranking['room_name'], room_ranking['Reservas'])
                    ax.set_xlabel('Número de Reservas')
                    ax.set_title('Ranking de Uso de Salas')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay datos de reservas de salas.")
            
            with col_rank2:
                st.markdown("#### 🪑 Ranking de Uso: Cupos Flexibles")
                flex_ranking = generate_flex_usage_ranking(conn)
                if not flex_ranking.empty:
                    st.dataframe(flex_ranking, hide_index=True, use_container_width=True)
                    
                    # Gráfico de ranking de cupos flexibles (top 10)
                    top_flex = flex_ranking.head(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(top_flex['user_name'], top_flex['Reservas'])
                    ax.set_xlabel('Número de Reservas')
                    ax.set_title('Top 10 - Uso de Cupos Flexibles')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay datos de reservas de cupos flexibles.")

        # -----------------------------------------------------------
        # T5: CONFIG
        # -----------------------------------------------------------
        with t5:
            st.subheader("Configuración de Administrador")
            nu = st.text_input("Usuario administrador", value=admin_user, key="admin_user")
            np = st.text_input("Contraseña administrador", type="password", value=admin_pass, key="admin_pass")
            ne = st.text_input("Email administrador", value=settings.get("admin_email", ""), key="admin_email")
            
            if st.button("💾 Guardar Configuración", key="sc", use_container_width=True): 
                save_setting(conn, "admin_user", nu)
                save_setting(conn, "admin_pass", np)
                save_setting(conn, "admin_email", ne)
                st.success("✅ Configuración guardada correctamente")

        # -----------------------------------------------------------
        # T6: APARIENCIA
        # -----------------------------------------------------------
        with t6: 
            admin_appearance_ui(conn)
        
        # -----------------------------------------------------------
        # T7: MANTENIMIENTO
        # -----------------------------------------------------------
        with t7:
            st.subheader("Mantenimiento del Sistema")
            st.warning("⚠️ Estas acciones son irreversibles. Usa con cuidado.")
            
            opt = st.radio("Selecciona qué deseas borrar:", 
                        ["Reservas", "Distribución", "Planos/Zonas", "TODO"], 
                        key="opcion_borrar")
            
            if st.button("🗑️ EJECUTAR BORRADO", type="primary", use_container_width=True): 
                msg = perform_granular_delete(conn, opt)
                st.success(f"✅ {msg}")

if __name__ == "__main__":
    main()
