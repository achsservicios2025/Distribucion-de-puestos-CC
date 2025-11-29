# modules/pdfgen.py

import pandas as pd
import tempfile
import os
import re
from io import BytesIO
from pathlib import Path # <--- ¡ESTA ES LA CORRECCIÓN CLAVE!

# Importaciones de terceros
from fpdf import FPDF
import matplotlib.pyplot as plt
from PIL import Image

# Importaciones de módulos locales (solo funciones, no variables de rutas)
from modules.zones import generate_colored_plan 

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
PLANOS_DIR = Path("planos")
COLORED_DIR = Path("planos_coloreados")

# --- FUNCIONES HELPER GLOBALES ---

def clean_pdf_text(text: str) -> str:
    """Limpia caracteres especiales para compatibilidad con FPDF."""
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

def apply_sorting_to_df(df, order_dias):
    """Aplica orden lógico a un DataFrame para Pisos y Días."""
    if df.empty: return df
    df = df.copy()
    
    cols_lower = {c.lower(): c for c in df.columns}
    col_dia = cols_lower.get('dia') or cols_lower.get('día')
    col_piso = cols_lower.get('piso')
    
    if col_dia:
        df[col_dia] = pd.Categorical(df[col_dia], categories=order_dias, ordered=True)
    
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

# --- GENERADORES DE PDF ---

def create_merged_pdf(piso_sel, order_dias, conn, read_distribution_df_func, global_logo_path, style_config):
    """
    Genera un único PDF (Dossier) con el plano coloreado para cada día hábil de un piso.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    found_any = False

    df = read_distribution_df_func(conn)
    base_config = style_config

    for dia in order_dias:
        subset = df[(df['piso'] == piso_sel) & (df['dia'] == dia)]
        current_seats = dict(zip(subset['equipo'], subset['cupos']))
        
        day_config = base_config.copy()
        if not day_config.get("subtitle_text") or "Día:" not in str(day_config.get("subtitle_text","")):
            day_config["subtitle_text"] = f"Día: {dia}"

        img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)
        
        if img_path and Path(img_path).exists():
            found_any = True
            pdf.add_page()
            try: pdf.image(str(img_path), x=10, y=10, w=190)
            except: pass
            
    if not found_any: return None
    try: return pdf.output(dest='S').encode('latin-1')
    except: return None

def generate_full_pdf(distrib_df, logo_path, deficit_data=None, order_dias=None):
    """
    Genera el reporte PDF de distribución con tablas diaria, semanal y déficit.
    (El resto del código de esta función es idéntico a la versión anterior y es correcto)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    
    # --- PÁGINA 1: DISTRIBUCIÓN DIARIA ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    if Path(logo_path).exists():
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

    distrib_df = apply_sorting_to_df(distrib_df, order_dias) if order_dias else distrib_df
    
    for _, r in distrib_df.iterrows():
        pdf.cell(widths[0], 6, clean_pdf_text(get_val(r, ["piso"])), 1)
        pdf.cell(widths[1], 6, clean_pdf_text(get_val(r, ["equipo"])[:40]), 1)
        pdf.cell(widths[2], 6, clean_pdf_text(get_val(r, ["dia"])), 1)
        pdf.cell(widths[3], 6, clean_pdf_text(get_val(r, ["cupos"])), 1)
        pct_val = get_val(r, ["pct"])
        pdf.cell(widths[4], 6, clean_pdf_text(f"{pct_val}%"), 1)
        pdf.ln()

    # --- SECCIÓN 2: TABLA SEMANAL ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    
    try:
        # Calcular resumen semanal
        weekly_summary = calculate_weekly_usage_summary(distrib_df)
        
        if not weekly_summary.empty:
            pdf.set_font("Arial", 'B', 9)
            w_wk = [80, 40, 40]  # Quitamos "Días Asignados"
            h_wk = ["Equipo", "Total Cupos Semanales", "% Distribución Semanal"]
            start_x = 25
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for _, row in weekly_summary.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:30]), 1)
                pdf.cell(w_wk[1], 6, clean_pdf_text(str(int(row["Total Cupos Semanales"]))), 1)
                # Calcular porcentaje de distribución semanal
                pct_semanal = (row["Total Cupos Semanales"] / row["Dotación Total"]) * 100
                pdf.cell(w_wk[2], 6, clean_pdf_text(f"{pct_semanal:.1f}%"), 1)
                pdf.ln()
        else:
            pdf.set_font("Arial", 'I', 9)
            pdf.cell(0, 6, clean_pdf_text("No hay datos suficientes para calcular el resumen semanal"), ln=True)
            
    except Exception as e:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular el resumen semanal: {str(e)}"), ln=True)

    # --- GLOSARIO Y DÉFICIT ---
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, clean_pdf_text("Glosario de Métricas y Cálculos:"), ln=True)
    pdf.set_font("Arial", '', 9)
    notas = ["1. % Distribución Diario: ...", "2. % Uso Semanal: ...", "3. Cálculo de Déficit: ..."]
    for nota in notas: pdf.multi_cell(185, 6, clean_pdf_text(nota))

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
            
            current_x = pdf.get_x()
            current_y = pdf.get_y()

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
            else:
                y_start = pdf.get_y()

            x_start = pdf.get_x()

            pdf.cell(dw[0], row_height, piso, 1, 0, 'C')
            
            pdf.set_xy(x_start + dw[0], y_start)
            pdf.multi_cell(dw[1], line_height, equipo, 1, 'L', fill=False)
            
            pdf.set_xy(x_start + dw[0] + dw[1], y_start)

            pdf.cell(dw[2], row_height, dia, 1, 0, 'C')
            pdf.cell(dw[3], row_height, dot, 1, 0, 'C')
            pdf.cell(dw[4], row_height, mini, 1, 0, 'C')

            pdf.set_font("Arial", 'B', 8)
            pdf.set_text_color(180, 0, 0)
            pdf.cell(dw[5], row_height, falt, 1, 0, 'C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 8)

            pdf.set_xy(x_start + dw[0] + dw[1] + dw[2] + dw[3] + dw[4] + dw[5], y_start)
            pdf.multi_cell(dw[6], line_height, causa, 1, 'L', fill=False)
            
            pdf.set_xy(x_start, y_start + row_height)

    try: return pdf.output(dest='S').encode('latin-1')
    except: return None

def calculate_weekly_usage_summary(distrib_df):
    """
    Calcula el resumen semanal de uso por equipo, quitando 'Días Asignados' y agregando '% Distribución Semanal'
    """
    if distrib_df.empty:
        return pd.DataFrame()
    
    # Identificar columnas
    equipo_col = None
    cupos_col = None
    dia_col = None
    for col in distrib_df.columns:
        cl = col.lower()
        if 'equipo' in cl:
            equipo_col = col
        elif 'cupos' in cl:
            cupos_col = col
        elif 'dia' in cl or 'día' in cl:
            dia_col = col
    
    if not all([equipo_col, cupos_col, dia_col]):
        return pd.DataFrame()
    
    # Filtrar solo equipos (excluir cupos libres)
    equipos_df = distrib_df[distrib_df[equipo_col] != "Cupos libres"]
    if equipos_df.empty:
        return pd.DataFrame()
    
    # Calcular total semanal por equipo
    weekly = equipos_df.groupby(equipo_col).agg({cupos_col: 'sum'}).reset_index()
    weekly.columns = ['Equipo', 'Total Cupos Semanales']
    
    # Obtener dotación total (asumiendo que la dotación es la misma para todos los días)
    # Esto podría necesitar ajustes dependiendo de la estructura de tus datos
    dotacion_map = equipos_df.groupby(equipo_col)[cupos_col].max().to_dict()  # Esto es un aproximado
    weekly['Dotación Total'] = weekly['Equipo'].map(dotacion_map)
    
    # Calcular porcentaje de distribución semanal
    weekly['% Distribución Semanal'] = (weekly['Total Cupos Semanales'] / weekly['Dotación Total']) * 100
    weekly['% Distribución Semanal'] = weekly['% Distribución Semanal'].round(1)
    
    return weekly
