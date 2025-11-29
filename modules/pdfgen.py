import pandas as pd
import tempfile
import os
import re
from io import BytesIO
from pathlib import Path 

# Importaciones de terceros
from fpdf import FPDF
from PIL import Image

# Importaciones de módulos locales (solo funciones, no variables de rutas)
from modules.zones import generate_colored_plan 

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
PLANOS_DIR = Path("planos")
COLORED_DIR = Path("planos_coloreados")

# --- FUNCIONES HELPER GLOBALES ---

def clean_pdf_text(text: str) -> str:
    """Limpia caracteres especiales y asegura codificación Latin-1."""
    if not isinstance(text, str): return str(text)
    
    # Mapeo manual extendido
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "Ñ": "N",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "•": "-", "—": "-", "–": "-", "⚠": "ATENCION:", "⚠️": "ATENCION:", 
        "…": "...", "º": "o", "°": "", "“": '"', "”": '"'
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    # Intento de codificación seguro
    try:
        return text.encode('latin-1', 'replace').decode('latin-1')
    except:
        return text

def sort_floors(floor_list):
    """Ordena una lista de pisos lógicamente (1, 2, 10)."""
    def extract_num(text):
        text = str(text)
        num = re.findall(r'\d+', text)
        return int(num[0]) if num else 0
    return sorted(list(floor_list), key=extract_num)

def apply_sorting_to_df(df, order_dias=None):
    """Aplica orden lógico a un DataFrame para Pisos y Días."""
    if df.empty: return df
    df = df.copy()
    
    cols_lower = {c.lower(): c for c in df.columns}
    col_dia = cols_lower.get('dia') or cols_lower.get('día')
    col_piso = cols_lower.get('piso')
    
    if order_dias and col_dia:
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
    try: return pdf.output(dest='S').encode('latin-1', 'replace')
    except: return pdf.output(dest='S')

def generate_full_pdf(distrib_df, listado_reservas_df=None, listado_salas_df=None, logo_path=None, deficit_data=None, order_dias=None, is_admin=False):
    """
    Genera el reporte PDF completo: Distribución, Semanal y (si es admin) Uso de Salas/Puestos.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    
    # --- PÁGINA 1: DISTRIBUCIÓN DIARIA ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    if logo_path and Path(logo_path).exists():
        try: pdf.image(str(logo_path), x=10, y=8, w=30)
        except: pass
    pdf.ln(25)
    pdf.cell(0, 10, clean_pdf_text("Informe de Distribución y Uso"), ln=True, align='C')
    pdf.ln(6)

    # Título de sección
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_pdf_text("1. Detalle de Distribución Diaria"), ln=True)

    # Tabla Diaria
    pdf.set_font("Arial", 'B', 9)
    widths = [30, 60, 25, 25, 25]
    headers = ["Piso", "Equipo", "Dia", "Cupos", "% Dia"]      
    for w, h in zip(widths, headers): pdf.cell(w, 6, clean_pdf_text(h), 1)
    pdf.ln()

    pdf.set_font("Arial", '', 9)
    def get_val(row, keys):
        for k in keys:
            if k in row: return str(row[k])
            if k.lower() in row: return str(row[k.lower()])
        return ""

    distrib_df_sorted = apply_sorting_to_df(distrib_df, order_dias) if order_dias else distrib_df
    
    for _, r in distrib_df_sorted.iterrows():
        pdf.cell(widths[0], 6, clean_pdf_text(get_val(r, ["piso"])), 1)
        pdf.cell(widths[1], 6, clean_pdf_text(get_val(r, ["equipo"])[:40]), 1)
        pdf.cell(widths[2], 6, clean_pdf_text(get_val(r, ["dia"])), 1)
        pdf.cell(widths[3], 6, clean_pdf_text(get_val(r, ["cupos"])), 1)
        pct_val = get_val(r, ["pct"])
        pdf.cell(widths[4], 6, clean_pdf_text(f"{pct_val}%"), 1)
        pdf.ln()

    # --- SECCIÓN 2: TABLA SEMANAL (MODIFICADA) ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, clean_pdf_text("2. Resumen Semanal por Equipo"), ln=True)
    
    try:
        # Filtrar cupos libres para esta tabla
        df_clean = distrib_df[distrib_df['equipo'] != "Cupos libres"].copy()
        
        if not df_clean.empty:
            # Inferir dotación si no existe columna explicita
            if 'dotacion_total' not in df_clean.columns:
                # Estimación inversa: si 10 cupos es 50%, dotacion es 20
                df_clean['dotacion_total'] = df_clean.apply(lambda x: int(x['cupos'] / (x['pct']/100)) if x['pct'] > 0 else 0, axis=1)

            # Agrupar
            grp = df_clean.groupby('equipo')
            summary = grp.agg(
                total_semanal=('cupos', 'sum'),
                dotacion_ref=('dotacion_total', 'max') # Usar max para referencia
            ).reset_index()
            
            # Calcular % semanal real
            # Formula: (Suma Cupos / (Dotación * 5)) * 100
            summary['pct_semanal'] = summary.apply(lambda x: round((x['total_semanal'] / (x['dotacion_ref'] * 5)) * 100, 1) if x['dotacion_ref'] > 0 else 0, axis=1)
            summary['promedio_diario'] = summary['total_semanal'] / 5

            pdf.set_font("Arial", 'B', 9)
            w_wk = [90, 30, 30, 30]
            h_wk = ["Equipo", "Tot. Semanal", "Prom. Diario", "% Semanal"]
            
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for _, row in summary.iterrows():
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["equipo"])[:50]), 1)
                pdf.cell(w_wk[1], 6, str(int(row["total_semanal"])), 1)
                pdf.cell(w_wk[2], 6, f"{row['promedio_diario']:.1f}", 1)
                pdf.cell(w_wk[3], 6, f"{row['pct_semanal']}%", 1)
                pdf.ln()
    except Exception as e:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular el resumen semanal: {str(e)}"), ln=True)

    # --- SECCIÓN 3 y 4: INFORMES ADMIN (SOLO SI ADMIN) ---
    if is_admin:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_pdf_text("Informes de Gestión (Solo Admin)"), ln=True, align='C')
        
        # 3. USO DE SALAS
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_pdf_text("3. Ranking de Uso: Salas de Reuniones"), ln=True)
        
        if listado_salas_df is not None and not listado_salas_df.empty:
            cols = {c.lower(): c for c in listado_salas_df.columns}
            col_user = cols.get('nombre') or cols.get('user_name') or cols.get('usuario')
            
            if col_user:
                top_users = listado_salas_df[col_user].value_counts().reset_index()
                top_users.columns = ['Usuario', 'Reservas']
                
                pdf.set_font("Arial", 'B', 9)
                pdf.cell(120, 7, "Usuario", 1)
                pdf.cell(40, 7, "Cant. Reservas", 1)
                pdf.ln()
                
                pdf.set_font("Arial", '', 9)
                for _, r in top_users.head(50).iterrows(): # Top 50
                    pdf.cell(120, 6, clean_pdf_text(str(r['Usuario'])), 1)
                    pdf.cell(40, 6, str(r['Reservas']), 1, 0, 'C')
                    pdf.ln()
            else:
                pdf.set_font("Arial", 'I', 9); pdf.cell(0,6, "No se encontró columna de usuario", ln=True)
        else:
            pdf.set_font("Arial", 'I', 9); pdf.cell(0,6, "No hay reservas de salas registradas.", ln=True)

        # 4. USO DE CUPOS FLEX
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_pdf_text("4. Ranking de Uso: Cupos Flexibles"), ln=True)
        
        if listado_reservas_df is not None and not listado_reservas_df.empty:
            cols = {c.lower(): c for c in listado_reservas_df.columns}
            col_user = cols.get('nombre') or cols.get('user_name') or cols.get('usuario')
            
            if col_user:
                top_puestos = listado_reservas_df[col_user].value_counts().reset_index()
                top_puestos.columns = ['Usuario', 'Reservas']
                
                pdf.set_font("Arial", 'B', 9)
                pdf.cell(120, 7, "Usuario", 1)
                pdf.cell(40, 7, "Cant. Reservas", 1)
                pdf.ln()
                
                pdf.set_font("Arial", '', 9)
                for _, r in top_puestos.head(50).iterrows():
                    pdf.cell(120, 6, clean_pdf_text(str(r['Usuario'])), 1)
                    pdf.cell(40, 6, str(r['Reservas']), 1, 0, 'C')
                    pdf.ln()
            else:
                pdf.set_font("Arial", 'I', 9); pdf.cell(0,6, "No se encontró columna de usuario", ln=True)
        else:
            pdf.set_font("Arial", 'I', 9); pdf.cell(0,6, "No hay reservas de puestos registradas.", ln=True)

    # --- GLOSARIO Y DÉFICIT ---
    if not is_admin: # Solo mostrar glosario si no es reporte admin (para ahorrar espacio)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, clean_pdf_text("Glosario de Métricas:"), ln=True)
        pdf.set_font("Arial", '', 9)
        notas = ["1. % Dia: Cupos asignados / Dotación total.", "2. % Semanal: (Total Semanal / (Dotación * 5)) * 100."]
        for nota in notas: pdf.multi_cell(185, 6, clean_pdf_text(nota))

    # --- DÉFICIT (Si existe) ---
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

    try: return pdf.output(dest='S').encode('latin-1', 'replace')
    except: return pdf.output(dest='S')
