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
        # Si es un string (URL base64), devolverlo directamente
        if isinstance(image_data, str):
            return image_data
        # Si width es int, convertirlo a WidthConfig
        if isinstance(width, int):
            width = WidthConfig(width=width)
        return _orig_image_to_url(image_data, width, clamp, channels, output_format, image_id)

    streamlit.elements.lib.image_utils.image_to_url = _patched_image_to_url

# Parche adicional para st_image (usado por st_canvas internamente)
try:
    import streamlit.elements.image as st_image_module
    if hasattr(st_image_module, "image_to_url"):
        _orig_st_image_to_url = st_image_module.image_to_url
        
        def _patched_st_image_to_url(image_data, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
            # Si es un string (URL base64), devolverlo directamente
            if isinstance(image_data, str):
                return image_data
            # Si width es int, convertirlo a WidthConfig
            if isinstance(width, int):
                width = WidthConfig(width=width)
            return _orig_st_image_to_url(image_data, width, clamp, channels, output_format, image_id)
        
        st_image_module.image_to_url = _patched_st_image_to_url
except:
    pass  # Si no existe el módulo, continuar sin parche

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
import streamlit.components.v1 as components
from streamlit_zone_editor import zone_editor

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

# --- NUEVA FUNCIÓN CON ESTRATEGIAS DE ORDENAMIENTO ---
def get_distribution_proposal(df_equipos, df_parametros, strategy="random", ignore_params=False):
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

    rows, deficit_report = compute_distribution_from_excel(eq_proc, pa_proc, 2, ignore_params=ignore_params)
    
    return rows, deficit_report

def generate_ideal_distributions(df_equipos, df_parametros, num_options=3):
    """
    Genera múltiples opciones ideales de distribución (aleatorias pero equitativas).
    """
    distributions = []
    for i in range(num_options):
        rows, deficit = get_distribution_proposal(df_equipos, df_parametros, strategy="random", ignore_params=True)
        distributions.append({
            'rows': rows,
            'deficit': deficit,
            'option_num': i + 1
        })
    return distributions

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

def hex_to_rgba(hex_color, alpha=0.3):
    """Convierte color hex a formato rgba para el canvas."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(0, 160, 74, {alpha})"

def create_enhanced_drawing_component(img_path, existing_zones, selected_team="", selected_color="#00A04A", width=700):
    """Componente profesional de dibujo mejorado con selección de equipo y color"""
    
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
        html_height = 650
        
        # HTML/JS Componente de dibujo profesional MEJORADO
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
                    padding: 10px;
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
                    padding: 10px 15px;
                    margin: 0;
                    font-size: 16px;
                }}
                .editor-controls {{
                    padding: 10px 15px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 5px;
                }}
                .control-btn {{
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 12px;
                    flex: 1;
                    min-width: 120px;
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
                .control-btn.save {{
                    background: #28a745;
                }}
                .control-btn.save:hover {{
                    background: #218838;
                }}
                .canvas-container {{
                    position: relative;
                    background: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 5px;
                }}
                #drawingCanvas {{
                    display: block;
                    cursor: crosshair;
                    border: 1px solid #ccc;
                    max-width: 100%;
                }}
                .status-panel {{
                    padding: 10px 15px;
                    background: #e9ecef;
                    border-top: 1px solid #dee2e6;
                    font-size: 12px;
                }}
                .coordinates {{
                    font-family: monospace;
                    background: #2b303b;
                    color: #00ff00;
                    padding: 8px;
                    border-radius: 5px;
                    margin: 5px 0;
                    font-size: 11px;
                }}
                .zones-list {{
                    max-height: 150px;
                    overflow-y: auto;
                    margin: 10px 0;
                }}
                .zone-item {{
                    padding: 5px;
                    margin: 2px 0;
                    background: white;
                    border-radius: 3px;
                    font-size: 11px;
                }}
            </style>
        </head>
        <body>
            <div class="editor-container">
                <h3 class="editor-header">🎨 Editor de Planos</h3>
                
                <div class="editor-controls">
                    <button class="control-btn" onclick="startDrawing()">✏️ Dibujar</button>
                    <button class="control-btn" onclick="clearLast()">🗑️ Borrar Último</button>
                    <button class="control-btn delete" onclick="clearAll()">🗑️ Borrar Todo</button>
                    <button class="control-btn save" onclick="saveZones()">💾 Guardar Zonas</button>
                </div>
                <div class="canvas-container">
                    <canvas id="drawingCanvas"></canvas>
                </div>
                <div class="status-panel">
                    <div class="coordinates">
                        <strong>Coordenadas:</strong><br>
                        <span id="coordsDisplay">X: 0, Y: 0</span>
                    </div>
                    <div class="zones-list" id="zonesList">
                        <strong>Zonas creadas:</strong>
                        <div id="zonesContainer"></div>
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
                let rectangles = JSON.parse(`{existing_zones_json}`.replace(/&quot;/g, '"'));
                let currentRect = null;
                let canvasWidth = {canvas_width};
                let canvasHeight = 0;
                let selectedTeam = "{selected_team}";
                let selectedColor = "{selected_color}";
                
                // Inicializar cuando la imagen cargue
                img.onload = function() {{
                    try {{
                        const aspectRatio = img.naturalHeight / img.naturalWidth;
                        canvasHeight = Math.round(canvasWidth * aspectRatio);
                        
                        canvas.width = canvasWidth;
                        canvas.height = canvasHeight;
                        
                        drawImageAndZones();
                        updateZonesList();
                        console.log('Canvas inicializado correctamente');
                    }} catch (e) {{
                        console.error('Error al inicializar canvas:', e);
                        alert('Error al cargar el plano: ' + e.message);
                    }}
                }};
                
                // Manejar errores de carga de imagen
                img.onerror = function() {{
                    console.error('Error al cargar la imagen');
                    alert('Error: No se pudo cargar la imagen del plano');
                }};
                
                // Si la imagen ya está cargada, inicializar de inmediato
                if (img.complete && img.naturalWidth > 0) {{
                    const aspectRatio = img.naturalHeight / img.naturalWidth;
                    canvasHeight = Math.round(canvasWidth * aspectRatio);
                    canvas.width = canvasWidth;
                    canvas.height = canvasHeight;
                    drawImageAndZones();
                    updateZonesList();
                    console.log('Canvas inicializado (imagen ya cargada)');
                }}
                
                function drawImageAndZones() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    if (currentRect) {{
                        drawRectangle(currentRect);
                    }}
                    
                    rectangles.forEach(rect => {{
                        const scaleX = canvas.width / img.naturalWidth;
                        const scaleY = canvas.height / img.naturalHeight;
                        
                        const canvasRect = {{
                            x: rect.x * scaleX,
                            y: rect.y * scaleY,
                            w: rect.w * scaleX,
                            h: rect.h * scaleY,
                            color: rect.color,
                            team: rect.team
                        }};
                        
                        drawRectangle(canvasRect);
                    }});
                }}
                
                function drawRectangle(rect) {{
                    ctx.strokeStyle = rect.color || selectedColor;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
                    
                    // Relleno semitransparente
                    ctx.fillStyle = (rect.color || selectedColor) + '40';
                    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
                }}
                
                function startDrawing() {{
                    isDrawing = true;
                    canvas.style.cursor = 'crosshair';
                }}
                
                function getCanvasCoordinates(e) {{
                    const rect = canvas.getBoundingClientRect();
                    const x = (e.pageX - rect.left - window.pageXOffset);
                    const y = (e.pageY - rect.top - window.pageYOffset);
                    
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
                        color: selectedColor,
                        team: selectedTeam || 'Nueva Zona'
                    }};
                }});
                
                canvas.addEventListener('mousemove', function(e) {{
                    if (!isDrawing || !currentRect) return;
                    
                    const coords = getCanvasCoordinates(e);
                    currentX = coords.x;
                    currentY = coords.y;
                    
                    currentRect.w = currentX - startX;
                    currentRect.h = currentY - startY;
                    
                    document.getElementById('coordsDisplay').textContent = 
                        `X: ${{Math.round(startX)}}, Y: ${{Math.round(startY)}}, ` +
                        `Ancho: ${{Math.round(currentRect.w)}}, Alto: ${{Math.round(currentRect.h)}}`;
                    
                    drawImageAndZones();
                }});
                
                canvas.addEventListener('mouseup', function(e) {{
                    if (!isDrawing || !currentRect) return;
                    
                    if (Math.abs(currentRect.w) > 10 && Math.abs(currentRect.h) > 10) {{
                        const scaleX = img.naturalWidth / canvas.width;
                        const scaleY = img.naturalHeight / canvas.height;
                        
                        const newRect = {{
                            x: Math.round(currentRect.x * scaleX),
                            y: Math.round(currentRect.y * scaleY),
                            w: Math.round(currentRect.w * scaleX),
                            h: Math.round(currentRect.h * scaleY),
                            color: selectedColor,
                            team: selectedTeam || 'Nueva Zona'
                        }};
                        
                        rectangles.push(newRect);
                        updateZonesList();
                        
                        // Auto-guardar después de dibujar
                        setTimeout(() => {{
                            const zonesData = JSON.stringify(rectangles);
                            localStorage.setItem('zones_auto_' + '{p_sel}', zonesData);
                        }}, 100);
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
                        updateZonesList();
                    }}
                }}
                
                function clearAll() {{
                    if (rectangles.length > 0) {{
                        if (confirm('¿Estás seguro de que quieres eliminar TODAS las zonas?')) {{
                            rectangles = [];
                            drawImageAndZones();
                            updateZonesList();
                        }}
                    }}
                }}
                
                function updateZonesList() {{
                    const container = document.getElementById('zonesContainer');
                    container.innerHTML = '';
                    
                    rectangles.forEach((rect, index) => {{
                        const zoneDiv = document.createElement('div');
                        zoneDiv.className = 'zone-item';
                        zoneDiv.style.borderLeft = `3px solid ${{rect.color}}`;
                        zoneDiv.innerHTML = `${{index + 1}}. ${{rect.team}} (${{Math.round(rect.x)}}, ${{Math.round(rect.y)}})`;
                        container.appendChild(zoneDiv);
                    }});
                }}
                
                function saveZones() {{
                    // Guardar automáticamente - enviar directamente a Streamlit
                    const zonesData = rectangles;
                    const zonesJson = JSON.stringify(zonesData);
                    
                    // Enviar a Streamlit usando postMessage
                    if (window.parent) {{
                        // Enviar mensaje al padre
                        window.parent.postMessage({{
                            type: 'zones_saved',
                            piso: '{p_sel}',
                            zones: zonesData,
                            zones_json: zonesJson
                        }}, '*');
                        
                        // También guardar en localStorage como respaldo
                        localStorage.setItem('zones_save_{p_sel}', zonesJson);
                        
                        alert('✅ Zonas guardadas automáticamente! (' + rectangles.length + ' zonas)\\n\\nLa página se actualizará en un momento.');
                    }} else {{
                        // Fallback: guardar en localStorage
                        localStorage.setItem('zones_save_{p_sel}', zonesJson);
                        alert('✅ Zonas guardadas! (' + rectangles.length + ' zonas)');
                    }}
                }}
                
                // Auto-guardar cuando se dibuja un rectángulo
                canvas.addEventListener('mouseup', function(e) {{
                    // Después de dibujar, auto-guardar después de un breve delay
                    if (isDrawing && currentRect && Math.abs(currentRect.w) > 10 && Math.abs(currentRect.h) > 10) {{
                        setTimeout(() => {{
                            // Auto-guardar silenciosamente
                            if (window.parent && rectangles.length > 0) {{
                                window.parent.postMessage({{
                                    type: 'zones_saved',
                                    piso: '{p_sel}',
                                    zones: rectangles
                                }}, '*');
                            }}
                        }}, 500);
                    }}
                }});
                
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
        
        # Componente que puede recibir valores de retorno
        # Aumentar altura para asegurar que se vea todo
        return components.html(html_code, width=canvas_width + 50, height=html_height + 100, scrolling=True)
        
    except Exception as e:
        st.error(f"Error al crear el componente de dibujo: {str(e)}")
        return None

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
    
    # Cálculo del resumen semanal (CORREGIDO: quitar "días asignados", agregar "% distr semanal")
    try:
        # Calcular total semanal por equipo
        if "Cupos" in distrib_df.columns or "cupos" in distrib_df.columns:
            col_cupos = "Cupos" if "Cupos" in distrib_df.columns else "cupos"
            col_equipo = "Equipo" if "Equipo" in distrib_df.columns else "equipo"
            
            # Agrupar por equipo y sumar cupos semanales
            weekly_stats = distrib_df.groupby(col_equipo)[col_cupos].sum().reset_index()
            weekly_stats.columns = ["Equipo", "Total Semanal"]
            
            # Calcular porcentaje de distribución semanal
            total_cupos_semana = weekly_stats["Total Semanal"].sum()
            if total_cupos_semana > 0:
                weekly_stats["% Distr Semanal"] = (weekly_stats["Total Semanal"] / total_cupos_semana * 100).round(2)
            else:
                weekly_stats["% Distr Semanal"] = 0
            
            # Ordenar por total semanal descendente
            weekly_stats = weekly_stats.sort_values("Total Semanal", ascending=False)
            
            # Dibujar Tabla Semanal
            pdf.set_font("Arial", 'B', 9)
            w_wk = [80, 40, 40]
            h_wk = ["Equipo", "Total Semanal", "% Distr Semanal"]
            
            # Centrar un poco la tabla
            start_x = 25
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for _, row in weekly_stats.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:40]), 1)
                pdf.cell(w_wk[1], 6, clean_pdf_text(str(int(row["Total Semanal"]))), 1)
                pdf.cell(w_wk[2], 6, clean_pdf_text(f"{row['% Distr Semanal']:.2f}%"), 1)
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

# CORREGIDO: Cargar logo con manejo robusto de errores
try:
    logo_path_abs = Path(global_logo_path).resolve()
    if logo_path_abs.exists() and logo_path_abs.is_file():
        c1, c2 = st.columns([1, 5])
        c1.image(str(logo_path_abs), width=150, use_container_width=False)
        c2.title(site_title)
    else:
        # Intentar con ruta relativa
        if os.path.exists(global_logo_path):
            c1, c2 = st.columns([1, 5])
            c1.image(global_logo_path, width=150, use_container_width=False)
            c2.title(site_title)
        else:
            st.title(site_title)
            if global_logo_path != "static/logo.png":
                st.info(f"💡 Logo configurado en: {global_logo_path} (archivo no encontrado)")
except Exception as e:
    # Si hay error al cargar el logo, mostrar solo el título
    st.title(site_title)
    st.warning(f"⚠️ No se pudo cargar el logo desde {global_logo_path}: {str(e)}")

# ---------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------
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
        t1, t2, t3 = st.tabs(["Estadísticas", "Ver Planos", "Reservas de Salas"])
        with t1:
            st.markdown("""
                <style>
                [data-testid="stElementToolbar"] {
                    display: none;
                }
                </style>
                """, unsafe_allow_html=True)
            
            lib = df_view[df_view["equipo"]=="Cupos libres"].groupby(["piso","dia"], as_index=True, observed=False).agg({"cupos":"sum"}).reset_index()
            
            # CORREGIDO: Asegurar que todos los piso/día tengan al menos 1 cupo libre (máximo 2)
            # Obtener todos los pisos y días únicos
            todos_pisos = df_view["piso"].unique()
            todos_dias = df_view["dia"].unique()
            
            # Crear combinaciones completas
            for piso in todos_pisos:
                for dia in todos_dias:
                    if dia != "FinDeSemana":
                        # Verificar si existe en lib
                        mask = (lib["piso"] == piso) & (lib["dia"] == dia)
                        if not mask.any():
                            # Agregar con 1 cupo mínimo
                            lib = pd.concat([lib, pd.DataFrame([{"piso": piso, "dia": dia, "cupos": 1}])], ignore_index=True)
                        else:
                            # Asegurar mínimo 1, máximo 2
                            idx = lib[mask].index[0] if mask.any() else None
                            if idx is not None:
                                current_val = int(lib.loc[idx, "cupos"]) if pd.notna(lib.loc[idx, "cupos"]) else 1
                                lib.loc[idx, "cupos"] = max(1, min(2, current_val))
            
            lib = apply_sorting_to_df(lib)
            
            st.subheader("Distribución completa")
            # MODIFICADO: Fix use_container_width
            st.dataframe(df_view, hide_index=True, width=None, use_container_width=True)
            
            st.subheader("Calendario Mensual de Reservas por Piso")
            
            # Obtener reservas de cupos libres
            all_res = list_reservations_df(conn)
            
            # Obtener mes actual o seleccionado
            mes_actual = datetime.date.today().replace(day=1)
            mes_sel = st.selectbox("Seleccionar Mes", 
                                   [mes_actual + datetime.timedelta(days=30*i) for i in range(-2, 4)],
                                   format_func=lambda x: x.strftime("%B %Y"),
                                   index=2)
            
            # Crear calendario por piso
            pisos_cal = sort_floors(pisos_disponibles)
            
            for piso_cal in pisos_cal:
                st.markdown(f"### 📅 {piso_cal}")
                
                # Obtener reservas de este piso en el mes seleccionado
                reservas_piso = []
                if not all_res.empty:
                    mask_piso = (all_res["piso"] == piso_cal) & (all_res["team_area"] == "Cupos libres")
                    for _, r in all_res[mask_piso].iterrows():
                        try:
                            fecha_res = pd.to_datetime(r["reservation_date"])
                            if fecha_res.year == mes_sel.year and fecha_res.month == mes_sel.month:
                                reservas_piso.append({
                                    "fecha": fecha_res,
                                    "equipo": r["user_name"],
                                    "correo": r["user_email"]
                                })
                        except:
                            pass
                
                # Crear calendario HTML con dimensiones ajustadas para más información
                import calendar
                cal = calendar.monthcalendar(mes_sel.year, mes_sel.month)
                
                # Crear HTML para el calendario con mejor diseño
                html_cal = f"""
                <div style="margin: 20px 0; overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; table-layout: fixed;">
                        <thead>
                            <tr style="background-color: #00A04A; color: white;">
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Lun</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Mar</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Mié</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Jue</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Vie</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Sáb</th>
                                <th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Dom</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for week in cal:
                    html_cal += "<tr style='height: 120px;'>"
                    for day in week:
                        if day == 0:
                            html_cal += '<td style="padding: 0; border: 1px solid #ddd; background-color: #f5f5f5; vertical-align: top;"></td>'
                        else:
                            fecha_dia = datetime.date(mes_sel.year, mes_sel.month, day)
                            # Buscar reservas para este día
                            reservas_dia = [r for r in reservas_piso if r["fecha"].date() == fecha_dia]
                            
                            if reservas_dia:
                                # Limitar a 3 equipos visibles, mostrar "+X más" si hay más
                                equipos_lista = [r["equipo"] for r in reservas_dia]
                                if len(equipos_lista) > 3:
                                    equipos_mostrar = equipos_lista[:3]
                                    equipos_restantes = len(equipos_lista) - 3
                                    equipos_str = "<br>".join([f"• {eq}" for eq in equipos_mostrar])
                                    equipos_str += f'<br><span style="color: #006B32; font-weight: bold;">+{equipos_restantes} más</span>'
                                else:
                                    equipos_str = "<br>".join([f"• {eq}" for eq in equipos_lista])
                                
                                html_cal += f'''
                                <td style="padding: 8px 6px; border: 1px solid #ddd; background-color: #e8f5e9; vertical-align: top; min-height: 120px;">
                                    <div style="font-size: 14px; font-weight: bold; color: #006B32; margin-bottom: 4px; border-bottom: 1px solid #c8e6c9; padding-bottom: 2px;">{day}</div>
                                    <div style="font-size: 10px; color: #2e7d32; line-height: 1.4; word-wrap: break-word; overflow-wrap: break-word;">
                                        {equipos_str}
                                    </div>
                                </td>
                                '''
                            else:
                                html_cal += f'''
                                <td style="padding: 8px 6px; border: 1px solid #ddd; vertical-align: top; min-height: 120px;">
                                    <div style="font-size: 14px; font-weight: bold; color: #666; margin-bottom: 4px;">{day}</div>
                                    <div style="font-size: 9px; color: #999; font-style: italic;">Disponible</div>
                                </td>
                                '''
                    html_cal += "</tr>"
                
                html_cal += """
                        </tbody>
                    </table>
                </div>
                <style>
                    @media (max-width: 768px) {
                        table { font-size: 10px; }
                        td { padding: 4px 2px !important; min-height: 80px !important; }
                    }
                </style>
                """
                
                st.markdown(html_cal, unsafe_allow_html=True)
                st.markdown("---")
        
        with t3:
            st.subheader("Reservas de Salas de Reuniones")
            
            # Obtener reservas de salas
            df_salas = get_room_reservations_df(conn)
            
            if df_salas.empty:
                st.info("No hay reservas de salas registradas.")
            else:
                # Crear tabla con equipo, sala y hora
                df_tabla = df_salas[['user_name', 'room_name', 'reservation_date', 'start_time', 'end_time', 'piso']].copy()
                df_tabla.columns = ['Equipo', 'Sala', 'Fecha', 'Hora Inicio', 'Hora Fin', 'Piso']
                df_tabla = df_tabla.sort_values(['Fecha', 'Hora Inicio'])
                
                st.dataframe(df_tabla, hide_index=True, width=None, use_container_width=True)
        
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
                
                # Verificar si hay zonas guardadas para este piso
                zonas = load_zones()
                has_zones = p_sel in zonas and zonas[p_sel] and len(zonas[p_sel]) > 0
                
                opts = []
                if fpng.exists(): opts.append("Imagen (PNG)")
                if fpdf.exists(): opts.append("Documento (PDF)")
                
                if opts:
                    if fpng.exists(): st.image(str(fpng), width=550, caption=f"{p_sel} - {ds}")
                    sf = st.selectbox("Formato:", opts, key=f"dl_pub_{p_sel}_{ds}")
                    tf = fpng if "PNG" in sf else fpdf
                    mim = "image/png" if "PNG" in sf else "application/pdf"
                    with open(tf,"rb") as f: st.download_button(f"📥 Descargar {sf}", f, tf.name, mim, use_container_width=True, key=f"dl_btn_{p_sel}_{ds}")
                else:
                    # Recargar zonas para verificar
                    zonas_check = load_zones()
                    has_zones_check = p_sel in zonas_check and zonas_check[p_sel] and len(zonas_check[p_sel]) > 0
                    
                    if has_zones_check:
                        st.warning(f"⚠️ Hay {len(zonas_check[p_sel])} zonas guardadas para {p_sel}, pero el plano no está generado.")
                        st.info(f"💡 Ve a 'Editor Visual de Zonas' → 'Generar Planos' y haz clic en '🎨 Generar Vista Previa' para crear el plano.")
                    else:
                        st.warning(f"⚠️ No hay planos generados para {p_sel} - {ds}.")
                        st.info(f"💡 Primero crea zonas en 'Editor Visual de Zonas' y luego genera el plano en 'Generar Planos'.")

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
                # CORREGIDO: Siempre hay máximo 2 cupos libres por piso/día (según requerimiento)
                # Si no hay configuración, se asume 1 cupo disponible mínimo
                total_cupos = 2  # Máximo 2 cupos libres por día según requerimiento
                
                # Calcular ocupados
                all_res = list_reservations_df(conn)
                ocupados = 0
                if not all_res.empty:
                    mask = (all_res["reservation_date"].astype(str) == str(fe)) & \
                           (all_res["piso"] == pi) & \
                           (all_res["team_area"] == "Cupos libres")
                    ocupados = len(all_res[mask])
                
                # Disponibles = máximo 2, menos los ocupados
                # Si hay 0 ocupados, hay 1 disponible (mínimo)
                # Si hay 1 ocupado, hay 1 disponible
                # Si hay 2 ocupados, hay 0 disponibles (agotado)
                disponibles = max(0, min(2, total_cupos - ocupados))
                
                # Asegurar que siempre haya al menos 1 disponible si no está agotado
                if ocupados == 0:
                    disponibles = 1
                elif ocupados == 1:
                    disponibles = 1
                elif ocupados >= 2:
                    disponibles = 0
                
                if disponibles > 0:
                    st.success(f"✅ **HAY CUPO: Quedan {disponibles} puestos disponibles** (Total: {total_cupos}).")
                else:
                    st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")
                
                st.markdown("### Datos del Solicitante")
                
                # Obtener lista de equipos para seleccionar área
                equipos_disponibles = sorted(df[df["piso"] == pi]["equipo"].unique().tolist())
                equipos_disponibles = [e for e in equipos_disponibles if e != "Cupos libres"]
                
                with st.form("form_puesto"):
                    cf1, cf2 = st.columns(2)
                    # CAMBIO: Área/Equipo en lugar de nombre
                    area_equipo = cf1.selectbox("Área / Equipo", equipos_disponibles if equipos_disponibles else ["General"])
                    em = cf2.text_input("Correo Electrónico")
                    
                    submitted = st.form_submit_button("Confirmar Reserva", type="primary", disabled=(disponibles <= 0))
                    
                    if submitted:
                        if not em:
                            st.error("Por favor completa el correo electrónico.")
                        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', em):
                            st.error("Por favor ingresa un correo electrónico válido (ejemplo: usuario@ejemplo.com).")
                        elif user_has_reservation(conn, em, str(fe)):
                            st.error("Ya tienes una reserva registrada para esta fecha.")
                        elif count_monthly_free_spots(conn, area_equipo, fe) >= 2:
                            st.error(f"El equipo/área '{area_equipo}' ha alcanzado el límite de 2 reservas mensuales.")
                        elif disponibles <= 0:
                            st.error("Lo sentimos, el cupo se acaba de agotar.")
                        else:
                            # Usar el área/equipo como nombre (el correo identifica al usuario)
                            add_reservation(conn, area_equipo, em, pi, str(fe), "Cupos libres", datetime.datetime.now(datetime.timezone.utc).isoformat())
                            msg = f"✅ Reserva Confirmada:\n\n- Área/Equipo: {area_equipo}\n- Fecha: {fe}\n- Piso: {pi}\n- Tipo: Puesto Flex"
                            st.success(msg)
                            email_sent = send_reservation_email(em, "Confirmación Puesto", msg.replace("\n","<br>"))
                            if email_sent:
                                st.info("📧 Correo de confirmación enviado")
                            else:
                                st.warning("⚠️ No se pudo enviar el correo. Verifica la configuración SMTP.")
                            st.rerun()

    # ---------------------------------------------------------
    # OPCIÓN 2: RESERVAR SALA
    # ---------------------------------------------------------
    elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
        st.subheader("Agendar Sala de Reuniones")
        st.info("💡 Selecciona tu equipo/área y luego elige la sala y horario disponible")
        
        # Obtener lista de equipos desde la distribución
        df_dist = read_distribution_df(conn)
        equipos_lista = []
        if not df_dist.empty:
            equipos_lista = sorted([e for e in df_dist["equipo"].unique() if e != "Cupos libres"])
        
        if not equipos_lista:
            st.warning("⚠️ No hay equipos configurados. Contacta al administrador.")
        else:
            # Pestañas plegables por equipo (estilo horas médicas)
            st.markdown("### Selecciona tu Equipo/Área")
            equipo_seleccionado = st.selectbox("Equipo/Área", equipos_lista, key="equipo_sala_sel")
            
            st.markdown("---")
            st.markdown("### Selecciona Sala y Horario")
            
            # CORREGIDO: Lista completa de salas (4 salas)
            salas_disponibles = [
                "Sala Reuniones Pequeña Piso 1",
                "Sala Reuniones Grande Piso 1", 
                "Sala Reuniones Piso 2",
                "Sala Reuniones Piso 3"
            ]
            
            c_sala, c_fecha = st.columns(2)
            sl = c_sala.selectbox("Selecciona Sala", salas_disponibles, key="sala_sel")
            
            # Determinar piso desde el nombre de la sala
            if "Piso 1" in sl:
                pi_s = "Piso 1"
            elif "Piso 2" in sl:
                pi_s = "Piso 2"
            elif "Piso 3" in sl:
                pi_s = "Piso 3"
            else:
                pi_s = "Piso 1"
            
            fe_s = c_fecha.date_input("Fecha", min_value=datetime.date.today(), key="fs_sala")
            
            # Obtener reservas existentes para esta sala y fecha
            df_reservas_sala = get_room_reservations_df(conn)
            reservas_hoy = []
            if not df_reservas_sala.empty:
                mask = (df_reservas_sala["reservation_date"].astype(str) == str(fe_s)) & \
                       (df_reservas_sala["room_name"] == sl)
                reservas_hoy = df_reservas_sala[mask].to_dict("records")
            
            # Generar slots de una hora (formato horas médicas)
            tm = generate_time_slots("08:00", "18:00", 60)  # Slots de 1 hora
            
            st.markdown("#### Horarios Disponibles")
            
            # Mostrar horarios ocupados y disponibles (estilo calendario médico)
            if reservas_hoy:
                horarios_ocupados = ', '.join([f"{r.get('start_time', '')} - {r.get('end_time', '')}" for r in reservas_hoy])
                st.warning(f"⚠️ Horarios ocupados: {horarios_ocupados}")
            
            # Crear grid de horarios disponibles
            cols_horarios = st.columns(4)
            horarios_disponibles = []
            
            for idx, hora_inicio in enumerate(tm):
                # Calcular hora fin (1 hora después)
                hora_obj = datetime.datetime.strptime(hora_inicio, "%H:%M")
                hora_fin_obj = hora_obj + datetime.timedelta(hours=1)
                hora_fin = hora_fin_obj.strftime("%H:%M")
                
                # Verificar si está ocupado
                ocupado = False
                for r in reservas_hoy:
                    r_start = r.get('start_time', '')
                    r_end = r.get('end_time', '')
                    if r_start and r_end:
                        # Verificar traslape
                        if check_room_conflict([r], str(fe_s), sl, hora_inicio, hora_fin):
                            ocupado = True
                            break
                
                if not ocupado:
                    horarios_disponibles.append((hora_inicio, hora_fin))
            
            if not horarios_disponibles:
                st.error("🔴 No hay horarios disponibles para esta sala en la fecha seleccionada.")
            else:
                # Mostrar horarios disponibles en grid
                st.markdown("**Horarios disponibles (1 hora):**")
                grid_cols = st.columns(min(4, len(horarios_disponibles)))
                horario_seleccionado = None
                
                for idx, (h_inicio, h_fin) in enumerate(horarios_disponibles):
                    col_idx = idx % 4
                    with grid_cols[col_idx]:
                        if st.button(f"{h_inicio} - {h_fin}", key=f"slot_{idx}", use_container_width=True):
                            horario_seleccionado = (h_inicio, h_fin)
                            st.session_state['selected_slot'] = horario_seleccionado
                
                # Si hay un horario seleccionado, mostrar formulario
                if 'selected_slot' in st.session_state:
                    h_inicio, h_fin = st.session_state['selected_slot']
                    
                    st.markdown("---")
                    st.markdown(f"### Confirmar Reserva: {h_inicio} - {h_fin}")
                    
                    with st.form("form_sala"):
                        st.info(f"**Equipo/Área:** {equipo_seleccionado}\n\n**Sala:** {sl}\n\n**Fecha:** {fe_s}\n\n**Horario:** {h_inicio} - {h_fin}")
                        
                        e_s = st.text_input("Correo Electrónico", key="email_sala")
                        
                        sub_sala = st.form_submit_button("✅ Confirmar Reserva", type="primary")
                        
                        if sub_sala:
                            if not e_s:
                                st.error("Por favor ingresa tu correo electrónico.")
                            elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', e_s):
                                st.error("Por favor ingresa un correo electrónico válido (ejemplo: usuario@ejemplo.com).")
                            elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, h_inicio, h_fin):
                                st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
                                del st.session_state['selected_slot']
                            else:
                                # Usar equipo como nombre, correo identifica al usuario
                                add_room_reservation(conn, equipo_seleccionado, e_s, pi_s, sl, str(fe_s), h_inicio, h_fin, datetime.datetime.now(datetime.timezone.utc).isoformat())
                                msg = f"✅ Sala Confirmada:\n\n- Equipo/Área: {equipo_seleccionado}\n- Sala: {sl}\n- Fecha: {fe_s}\n- Horario: {h_inicio} - {h_fin}"
                                st.success(msg)
                                
                                # Enviar correo de confirmación
                                if e_s: 
                                    try:
                                        email_sent = send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))
                                        if email_sent:
                                            st.info("📧 Correo de confirmación enviado")
                                        else:
                                            st.warning("⚠️ No se pudo enviar el correo. Verifica la configuración SMTP.")
                                    except Exception as email_error:
                                        st.warning(f"⚠️ Error al enviar correo: {email_error}")
                                
                                del st.session_state['selected_slot']
                                st.rerun()

    # ---------------------------------------------------------
    # OPCIÓN 3: GESTIONAR (ANULAR Y VER TODO)
    # ---------------------------------------------------------
    elif opcion_reserva == "📋 Mis Reservas y Listados":
        
        # --- SECCION 1: BUSCADOR PARA ANULAR ---
        st.subheader("Buscar y Cancelar mis reservas")
        q = st.text_input("Ingresa tu Correo o Nombre para buscar:")
        
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
            
            # TÍTULO CORREGIDO 1
            st.subheader("Reserva de puestos") 
            st.dataframe(clean_reservation_df(list_reservations_df(conn)), hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True) 

            # TÍTULO CORREGIDO 2
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
                    # Ya no usamos ensure_reset_table porque la DB está lista
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
        
        # OPCIÓN DE IGNORAR PARÁMETROS
        ignore_params = st.checkbox(
            "Ignorar hoja de parámetros y generar distribución ideal",
            value=False,
            help="Si está marcado, generará distribuciones ideales equitativas sin usar los parámetros del Excel"
        )
        
        # SELECTOR DE ESTRATEGIA (solo si no se ignoran parámetros)
        if not ignore_params:
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
        else:
            sel_strat_code = "random"  # Para distribuciones ideales siempre usamos random
            c_strat.info("💡 Modo Distribución Ideal activado")

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
                    try:
                        # Leer Excel con encoding correcto para tildes
                        df_eq = pd.read_excel(up, "Equipos", engine='openpyxl')
                        if not ignore_params:
                            df_pa = pd.read_excel(up, "Parámetros", engine='openpyxl')
                        else:
                            # Crear DataFrame vacío si se ignoran parámetros
                            df_pa = pd.DataFrame()
                        
                        st.session_state['excel_equipos'] = df_eq
                        st.session_state['excel_params'] = df_pa
                        st.session_state['ignore_params'] = ignore_params
                        
                        if ignore_params:
                            # Generar múltiples opciones ideales
                            st.info("🔄 Generando opciones ideales de distribución...")
                            ideal_options = generate_ideal_distributions(df_eq, df_pa, num_options=3)
                            st.session_state['ideal_options'] = ideal_options
                            st.session_state['selected_ideal_option'] = 0
                            # Usar la primera opción como propuesta inicial
                            rows, deficit = ideal_options[0]['rows'], ideal_options[0]['deficit']
                        else:
                            # Generar propuesta inicial normal
                            rows, deficit = get_distribution_proposal(df_eq, df_pa, strategy=sel_strat_code, ignore_params=False)
                        
                        st.session_state['proposal_rows'] = rows
                        st.session_state['proposal_deficit'] = deficit
                        st.session_state['last_optimization_stats'] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al leer el Excel: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Error al leer el Excel: {e}")

        # 2. VISUALIZACIÓN Y ACCIONES
        if st.session_state['proposal_rows'] is not None:
            st.divider()
            
            # Si hay opciones ideales, mostrar selector
            if 'ideal_options' in st.session_state and st.session_state['ideal_options']:
                st.subheader("🎯 Opciones de Distribución Ideal")
                st.info("Se generaron múltiples opciones equitativas. Selecciona la que prefieras:")
                
                option_names = [f"Opción {i+1}" for i in range(len(st.session_state['ideal_options']))]
                selected_idx = st.selectbox(
                    "Selecciona una opción:",
                    range(len(option_names)),
                    format_func=lambda x: option_names[x],
                    key="ideal_option_selector"
                )
                
                if selected_idx != st.session_state.get('selected_ideal_option', 0):
                    # Cambiar a la opción seleccionada
                    st.session_state['selected_ideal_option'] = selected_idx
                    option_data = st.session_state['ideal_options'][selected_idx]
                    st.session_state['proposal_rows'] = option_data['rows']
                    st.session_state['proposal_deficit'] = option_data['deficit']
                    st.rerun()
                
                # Mostrar estadísticas de la opción seleccionada
                option_data = st.session_state['ideal_options'][selected_idx]
                n_def = len(option_data['deficit']) if option_data['deficit'] else 0
                st.success(f"✅ Opción {selected_idx + 1} seleccionada - Déficits: {n_def}")
            else:
                # Mostrar estadísticas de la optimización si existen
                if st.session_state.get('last_optimization_stats'):
                    stats = st.session_state['last_optimization_stats']
                    st.info(f"✨ **Resultado Optimizado:** Se probaron {stats['iterations']} combinaciones. Se eligió la que menos castiga repetidamente al mismo equipo.")
            
            # --- SECCIÓN DE RESULTADOS ---
            n_def = len(st.session_state['proposal_deficit']) if st.session_state['proposal_deficit'] else 0

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
                    st.dataframe(df_sorted, hide_index=True, width=None, use_container_width=True)
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
            st.caption("💡 **Probar otra suerte:** Genera una nueva variación aleatoria | **Auto-Optimizar:** Prueba 20 variaciones y elige la más equitativa | **Guardar Definitivo:** Guarda la distribución actual en la base de datos")
            
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

    with t2:
        st.info("Editor de Zonas - Versión Profesional")
        zonas = load_zones()
        
        # MODIFICADO: Leer con funcion importada
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        
        # Layout en columnas: Editor a la izquierda, Configuración a la derecha
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
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
                    
                    # Obtener equipo y color seleccionados
                    selected_team = st.session_state.get(f"team_{p_sel}", "")
                    selected_color = st.session_state.get(f"color_{p_sel}", "#00A04A")
                    
                    # Botón para actualizar el componente cuando cambien equipo/color
                    if st.button("🔄 Actualizar Editor con Equipo/Color Seleccionado", key=f"update_editor_{p_sel}"):
                        st.rerun()
                    
                    # Mostrar componente de dibujo mejorado con guardado automático
                    st.markdown("### 🎨 Herramientas de Dibujo")
                    
                    # Mostrar información actual
                    if selected_team:
                        st.info(f"🎨 **Equipo seleccionado:** {selected_team} | **Color:** {selected_color}")
                    else:
                        st.warning("⚠️ Selecciona un equipo y color en el panel derecho antes de dibujar")
                    
                    # Usar el componente HTML mejorado con guardado automático
                    drawing_component = create_enhanced_drawing_component(
                        str(pim), 
                        existing_zones, 
                        selected_team=selected_team,
                        selected_color=selected_color,
                        width=600
                    )
                    
                    if drawing_component is None:
                        st.error("❌ No se pudo cargar el componente de dibujo")
                    
                    # Botones de acción
                    col_btn1, col_btn2 = st.columns(2)
                    
                    if col_btn1.button("🔄 Recargar Zonas", key=f"reload_{p_sel}"):
                        zonas = load_zones()
                        st.rerun()
                        
                    if col_btn2.button("🗑️ Limpiar Todas", key=f"clear_all_{p_sel}"):
                        if st.session_state.get(f"confirm_clear_{p_sel}", False):
                            zonas[p_sel] = []
                            save_zones(zonas)
                            st.success("✅ Todas las zonas eliminadas")
                            st.session_state[f"confirm_clear_{p_sel}"] = False
                            st.rerun()
                        else:
                            st.session_state[f"confirm_clear_{p_sel}"] = True
                            st.warning("⚠️ Haz clic de nuevo para confirmar la eliminación de TODAS las zonas")
                    
                    # Campo oculto para recibir datos del componente
                    zones_json_hidden = st.text_input(
                        "",
                        value=json.dumps(existing_zones),
                        key=f"zones_json_hidden_{p_sel}",
                        label_visibility="collapsed"
                    )
                    
                    # Verificar si hay zonas guardadas en localStorage y procesarlas
                    save_key = f"zones_saved_{p_sel}"
                    
                    # Script para detectar cuando se guarda desde el componente
                    auto_save_script = f"""
                    <script>
                    (function() {{
                        // Al cargar, verificar si hay zonas guardadas en localStorage
                        function processSavedZones() {{
                            const savedZones = localStorage.getItem('zones_save_{p_sel}');
                            if (savedZones) {{
                                console.log('Procesando zonas desde localStorage:', savedZones.length, 'caracteres');
                                
                                // Buscar el campo oculto
                                let hiddenInput = document.querySelector('input[data-testid*="zones_json_hidden_{p_sel}"]');
                                if (!hiddenInput) {{
                                    // Intentar otros selectores
                                    hiddenInput = document.querySelector('input[type="text"][value*="[{{"]');
                                }}
                                
                                if (hiddenInput) {{
                                    console.log('Campo encontrado, actualizando...');
                                    hiddenInput.value = savedZones;
                                    
                                    // Disparar eventos
                                    hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                    hiddenInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                    
                                    // Limpiar localStorage
                                    localStorage.removeItem('zones_save_{p_sel}');
                                    
                                    // Recargar para que Python procese
                                    console.log('Recargando página...');
                                    setTimeout(() => {{
                                        window.location.reload();
                                    }}, 200);
                                }} else {{
                                    console.log('Campo no encontrado, reintentando...');
                                    setTimeout(processSavedZones, 300);
                                }}
                            }}
                        }}
                        
                        // Escuchar mensajes del componente HTML
                        window.addEventListener('message', function(event) {{
                            if (event.data && event.data.type === 'zones_saved' && event.data.piso === '{p_sel}') {{
                                console.log('Mensaje recibido:', event.data.zones.length, 'zonas');
                                
                                if (event.data.zones_json) {{
                                    // Guardar en localStorage
                                    localStorage.setItem('zones_save_{p_sel}', event.data.zones_json);
                                    console.log('Guardado en localStorage, procesando...');
                                    
                                    // Procesar inmediatamente
                                    processSavedZones();
                                }}
                            }}
                        }}, false);
                        
                        // Procesar al cargar
                        if (document.readyState === 'loading') {{
                            window.addEventListener('load', processSavedZones);
                        }} else {{
                            setTimeout(processSavedZones, 100);
                        }}
                    }})();
                    </script>
                    """
                    st.markdown(auto_save_script, unsafe_allow_html=True)
                    
                    # Procesar guardado automático cuando cambia el campo oculto
                    # Usar session_state para detectar cambios
                    last_zones_key = f"last_zones_{p_sel}"
                    if last_zones_key not in st.session_state:
                        st.session_state[last_zones_key] = json.dumps(existing_zones, sort_keys=True)
                    
                    try:
                        zones_data = json.loads(zones_json_hidden)
                        new_json = json.dumps(zones_data, sort_keys=True)
                        last_json = st.session_state[last_zones_key]
                        
                        if new_json != last_json:
                            # Guardar inmediatamente
                            zonas[p_sel] = zones_data
                            save_zones(zonas)
                            st.session_state[last_zones_key] = new_json
                            st.success(f"✅ {len(zones_data)} zonas guardadas automáticamente!")
                            st.rerun()
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        pass
                    
                    # Botón manual de guardado como respaldo
                    if st.button("💾 Guardar Zonas Manualmente", key=f"manual_save_{p_sel}"):
                        try:
                            zones_data = json.loads(zones_json_hidden)
                            if isinstance(zones_data, list):
                                zonas[p_sel] = zones_data
                                save_zones(zonas)
                                # Actualizar session_state
                                st.session_state[last_zones_key] = json.dumps(zones_data, sort_keys=True)
                                st.success(f"✅ {len(zones_data)} zonas guardadas manualmente!")
                                # Recargar zonas
                                zonas = load_zones()
                                st.rerun()
                            else:
                                st.error("Formato de datos inválido")
                        except json.JSONDecodeError as e:
                            st.error(f"Error al parsear JSON: {e}")
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    st.info("💡 **Guardado Automático:** Dibuja zonas y haz clic en '💾 Guardar Zonas' en el editor. Las zonas se guardarán automáticamente.")
                            
                except Exception as e:
                    st.error(f"❌ Error en el editor: {str(e)}")
            else:
                st.error(f"❌ No se encontró el plano: {p_sel}")
                st.info(f"💡 Busqué en: {pim}")
        
        with col_right:
            st.subheader("🎨 Configuración de Zonas")
            
            # Preparar lista de equipos
            d_sel = st.selectbox("Día Ref.", ORDER_DIAS, key=f"dia_ref_{p_sel}")
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
            
            # Selector de equipo y color
            tn = st.selectbox("Equipo / Sala", eqs, key=f"team_{p_sel}")
            tc = st.color_picker("Color", "#00A04A", key=f"color_{p_sel}")
            
            if tn and tn in current_seats_dict:
                st.info(f"📊 Cupos: {current_seats_dict[tn]}")
            
            st.markdown("---")
            
            # Mostrar zonas guardadas
            if p_sel in zonas and zonas[p_sel]:
                st.success(f"✅ {len(zonas[p_sel])} zonas guardadas para {p_sel}")
                
                # Editor de zonas existentes
                st.markdown("#### ✏️ Editar Zona Existente")
                zone_options = [f"{i+1}. {z.get('team', 'Sin nombre')}" 
                            for i, z in enumerate(zonas[p_sel])]
                
                if zone_options:
                    selected_zone_idx = st.selectbox(
                        "Selecciona una zona:",
                        range(len(zone_options)),
                        format_func=lambda x: zone_options[x],
                        key=f"zone_selector_{p_sel}"
                    )
                    
                    if selected_zone_idx is not None:
                        zone = zonas[p_sel][selected_zone_idx]
                        
                        # Controles de edición
                        new_team = st.text_input("Nombre del equipo:", 
                                            value=zone.get('team', 'Nueva Zona'),
                                            key=f"team_edit_{selected_zone_idx}_{p_sel}")
                        
                        new_color = st.color_picker("Color:", 
                                                value=zone.get('color', '#00A04A'),
                                                key=f"color_edit_{selected_zone_idx}_{p_sel}")
                        
                        col_edit1, col_edit2 = st.columns(2)
                        
                        with col_edit1:
                            if st.button("💾 Actualizar Zona", key=f"update_{selected_zone_idx}_{p_sel}"):
                                zonas[p_sel][selected_zone_idx]['team'] = new_team
                                zonas[p_sel][selected_zone_idx]['color'] = new_color
                                save_zones(zonas)
                                st.success("✅ Zona actualizada")
                                st.rerun()
                        
                        with col_edit2:
                            if st.button("🗑️ Eliminar Zona", key=f"delete_{selected_zone_idx}_{p_sel}"):
                                zonas[p_sel].pop(selected_zone_idx)
                                save_zones(zonas)
                                st.success("✅ Zona eliminada")
                                st.rerun()
                
                # Leyenda de colores
                st.markdown("#### 🎨 Leyenda de Colores")
                for i, z in enumerate(zonas[p_sel]):
                    col_leg1, col_leg2 = st.columns([1, 4])
                    with col_leg1:
                        st.markdown(f'<div style="width:30px;height:30px;background-color:{z.get("color", "#00A04A")};border:1px solid #ccc;"></div>', unsafe_allow_html=True)
                    with col_leg2:
                        st.write(f"**{z.get('team', 'Sin nombre')}**")
                        
            else:
                st.warning("ℹ️ No hay zonas guardadas para este piso. Usa el editor de la izquierda para crear zonas.")
            
            st.markdown("---")
            st.markdown("### 📝 Instrucciones")
            st.info("""
            1. Selecciona un **Equipo** y un **Color** arriba
            2. En el editor de la izquierda, haz clic en **✏️ Dibujar**
            3. Dibuja un rectángulo en el mapa
            4. Haz clic en **💾 Guardar Zonas** en el editor
            5. Copia el JSON que aparece y pégalo en el área de texto de la izquierda
            6. Haz clic en **💾 Guardar Zonas** en Streamlit para guardar definitivamente
            """)
        
        # Sección de personalización de título y leyenda (fuera de las columnas)
        st.divider()
        st.subheader("Personalización Título y Leyenda")
        with st.expander("🎨 Editar Estilos", expanded=True):
            tm = st.text_input("Título Principal", f"Distribución {p_sel}", key=f"title_{p_sel}")
            ts = st.text_input("Subtítulo (Opcional)", f"Día: {d_sel}", key=f"subtitle_{p_sel}")
            
            align_options = ["Izquierda", "Centro", "Derecha"]

            st.markdown("##### Estilos del Título Principal")
            cf1, cf2, cf3 = st.columns(3)
            ff_t = cf1.selectbox("Tipografía (Título)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key=f"font_t_{p_sel}")
            fs_t = cf2.selectbox("Tamaño Letra (Título)", [10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 64, 72, 80], index=9, key=f"size_t_{p_sel}")
            align = cf3.selectbox("Alineación (Título)", align_options, index=1, key=f"align_{p_sel}")

            st.markdown("---")
            st.markdown("##### Estilos del Subtítulo")
            cs1, cs2, cs3 = st.columns(3)
            ff_s = cs1.selectbox("Tipografía (Subtítulo)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key=f"font_s_{p_sel}")
            fs_s = cs2.selectbox("Tamaño Letra (Subtítulo)", [10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 64, 72, 80], index=5, key=f"size_s_{p_sel}")
            align_s = cs3.selectbox("Alineación (Subtítulo)", align_options, index=1, key=f"align_s_{p_sel}")

            st.markdown("---")
            st.markdown("##### Estilos de la Leyenda")
            cl1, cl2, cl3 = st.columns(3)
            ff_l = cl1.selectbox("Tipografía (Leyenda)", ["Arial", "Arial Black", "Calibri", "Comic Sans MS", "Courier New", "Georgia", "Impact", "Lucida Console", "Roboto", "Segoe UI", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana"], key=f"font_l_{p_sel}", index=0)
            fs_l = cl2.selectbox("Tamaño Letra (Leyenda)", [8, 10, 12, 14, 16, 18, 20, 24, 28, 32], index=3, key=f"size_l_{p_sel}")
            align_l = cl3.selectbox("Alineación (Leyenda)", align_options, index=0, key=f"align_l_{p_sel}")
            
            st.markdown("---")
            cg1, cg2, cg3, cg4 = st.columns(4) 
            lg = cg1.checkbox("Logo", True, key=f"chk_logo_{p_sel}"); 
            ln = cg2.checkbox("Mostrar Leyenda", True, key=f"chk_legend_{p_sel}");
            align_logo = cg3.selectbox("Alineación Logo", align_options, index=0, key=f"logo_align_{p_sel}")
            lw = cg4.slider("Ancho Logo", 50, 300, 150, key=f"logo_w_{p_sel}")
            
            cc1, cc2 = st.columns(2)
            bg = cc1.color_picker("Fondo Header", "#FFFFFF", key=f"bg_{p_sel}"); tx = cc2.color_picker("Color Texto", "#000000", key=f"tx_{p_sel}")

        fmt_sel = st.selectbox("Formato:", ["Imagen (PNG)", "Documento (PDF)"], key=f"fmt_{p_sel}")
        f_code = "PNG" if "PNG" in fmt_sel else "PDF"
        
        # Preparar configuración
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
        # Guardar config en session_state para usarla en dossier PDF
        st.session_state['last_style_config'] = conf
        
        # Obtener current_seats_dict
        current_seats_dict = {}
        if not df_d.empty:
            subset = df_d[(df_d['piso'] == p_sel) & (df_d['dia'] == d_sel)]
            current_seats_dict = dict(zip(subset['equipo'], subset['cupos']))
        
        # Verificar que hay zonas guardadas antes de generar
        zonas_check = load_zones()
        has_zones_for_piso = p_sel in zonas_check and zonas_check[p_sel] and len(zonas_check[p_sel]) > 0
        
        if not has_zones_for_piso:
            st.warning(f"⚠️ No hay zonas guardadas para {p_sel}. Ve a 'Editor Visual de Zonas' para crear zonas primero.")
        
        # Generar automáticamente al cambiar opciones
        if st.button("🎨 Generar Vista Previa", key=f"preview_{p_sel}", type="primary"):
            # Recargar zonas antes de generar para asegurar que tenemos los datos más recientes
            zonas_check = load_zones()
            has_zones_for_piso = p_sel in zonas_check and zonas_check[p_sel] and len(zonas_check[p_sel]) > 0
            
            if not has_zones_for_piso:
                st.error("❌ No se puede generar el plano sin zonas. Crea zonas primero en 'Editor Visual de Zonas'.")
            else:
                try:
                    # Asegurar que tenemos los datos más recientes
                    current_seats_dict = {}
                    if not df_d.empty:
                        subset = df_d[(df_d['piso'] == p_sel) & (df_d['dia'] == d_sel)]
                        current_seats_dict = dict(zip(subset['equipo'], subset['cupos']))
                    
                    out = generate_colored_plan(p_sel, d_sel, current_seats_dict, f_code, conf, global_logo_path)
                    if out and Path(out).exists(): 
                        st.success(f"✅ Vista previa generada correctamente!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error al generar el plano. Zonas encontradas: {len(zonas_check.get(p_sel, []))}. Verifica que las zonas estén guardadas correctamente.")
                except Exception as e:
                    st.error(f"❌ Error al generar el plano: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Mostrar vista previa si existe
        ds = d_sel.lower().replace("é","e").replace("á","a")
        fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
        fpdf = COLORED_DIR / f"piso_{p_num}_{ds}_combined.pdf"
        
        if fpng.exists() or fpdf.exists():
            st.markdown("### 📊 Vista Previa")
            if fpng.exists(): 
                st.image(str(fpng), width=700, caption=f"Vista Previa - {p_sel} - {d_sel}")
            elif fpdf.exists(): 
                st.info("📄 PDF generado (sin vista previa de imagen)")
            
            # Botón de descarga
            tf = fpng if "PNG" in fmt_sel else fpdf
            mm = "image/png" if "PNG" in fmt_sel else "application/pdf"
            if tf.exists():
                with open(tf,"rb") as f: 
                    st.download_button(
                        f"📥 Descargar {fmt_sel}", 
                        f, 
                        tf.name, 
                        mm, 
                        use_container_width=True, 
                        key=f"dl_{p_sel}"
                    )
        else:
            st.info("ℹ️ Haz clic en '🎨 Generar Vista Previa' para crear el plano con los estilos configurados")

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

        # Separar informes de cupos y salas
        st.markdown("### 📊 Informes de Distribución")
        rf = st.selectbox("Formato Reporte", ["Excel (XLSX)", "PDF"], key="report_format")
        if st.button("Generar Reporte de Distribución", key="gen_dist_report"):
            df_raw = read_distribution_df(conn); df_raw = apply_sorting_to_df(df_raw)
            if "Excel" in rf:
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w: 
                    df_raw.to_excel(w, index=False, sheet_name='Distribución')
                st.session_state['rd'] = b.getvalue(); st.session_state['rn'] = "distribucion.xlsx"; st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                df = df_raw.rename(columns={"piso":"Piso","equipo":"Equipo","dia":"Día","cupos":"Cupos","pct":"%Distrib"})
                d_data = st.session_state.get('deficit_report', [])
                st.session_state['rd'] = generate_full_pdf(df, df, logo_path=Path(global_logo_path), deficit_data=d_data)
                st.session_state['rn'] = "reporte_distribucion.pdf"; st.session_state['rm'] = "application/pdf"
            st.success("✅ Reporte generado")
        if 'rd' in st.session_state: st.download_button("📥 Descargar Reporte de Distribución", st.session_state['rd'], st.session_state['rn'], mime=st.session_state['rm'], key="dl_dist_report")
        
        st.markdown("---")
        st.markdown("### 📈 Informes de Reservas (Solo Admin)")
        
        # Informe de reservas de cupos
        st.markdown("#### 🪑 Reservas de Cupos")
        if st.button("Generar Informe de Cupos", key="gen_cupos_report"):
            df_cupos = list_reservations_df(conn)
            if not df_cupos.empty:
                # Calcular estadísticas por persona y equipo
                df_cupos_stats = df_cupos.groupby(['user_name', 'user_email', 'team_area']).agg({
                    'reservation_date': 'count'
                }).reset_index()
                df_cupos_stats.columns = ['Nombre', 'Correo', 'Equipo/Área', 'Cantidad Reservas']
                df_cupos_stats = df_cupos_stats.sort_values('Cantidad Reservas', ascending=False)
                
                # Calcular porcentajes
                total_reservas = df_cupos_stats['Cantidad Reservas'].sum()
                if total_reservas > 0:
                    df_cupos_stats['% del Total'] = (df_cupos_stats['Cantidad Reservas'] / total_reservas * 100).round(2)
                else:
                    df_cupos_stats['% del Total'] = 0
                
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    df_cupos_stats.to_excel(w, index=False, sheet_name='Por Persona')
                    # También por equipo
                    df_equipo_stats = df_cupos.groupby('team_area').agg({
                        'reservation_date': 'count'
                    }).reset_index()
                    df_equipo_stats.columns = ['Equipo/Área', 'Cantidad Reservas']
                    df_equipo_stats = df_equipo_stats.sort_values('Cantidad Reservas', ascending=False)
                    total_equipo = df_equipo_stats['Cantidad Reservas'].sum()
                    if total_equipo > 0:
                        df_equipo_stats['% del Total'] = (df_equipo_stats['Cantidad Reservas'] / total_equipo * 100).round(2)
                    else:
                        df_equipo_stats['% del Total'] = 0
                    df_equipo_stats.to_excel(w, index=False, sheet_name='Por Equipo')
                
                st.session_state['cupos_report'] = b.getvalue()
                st.session_state['cupos_report_name'] = "reservas_cupos.xlsx"
                st.success("✅ Informe de cupos generado")
            else:
                st.warning("No hay reservas de cupos registradas")
        
        if 'cupos_report' in st.session_state:
            st.download_button("📥 Descargar Informe de Cupos", st.session_state['cupos_report'], 
                             st.session_state['cupos_report_name'], 
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             key="dl_cupos_report")
        
        # Informe de reservas de salas
        st.markdown("#### 🏢 Reservas de Salas")
        if st.button("Generar Informe de Salas", key="gen_salas_report"):
            df_salas = get_room_reservations_df(conn)
            if not df_salas.empty:
                # Calcular estadísticas por persona
                df_salas_stats = df_salas.groupby(['user_name', 'user_email']).agg({
                    'reservation_date': 'count'
                }).reset_index()
                df_salas_stats.columns = ['Nombre', 'Correo', 'Cantidad Reservas']
                df_salas_stats = df_salas_stats.sort_values('Cantidad Reservas', ascending=False)
                
                # Calcular porcentajes
                total_reservas = df_salas_stats['Cantidad Reservas'].sum()
                if total_reservas > 0:
                    df_salas_stats['% del Total'] = (df_salas_stats['Cantidad Reservas'] / total_reservas * 100).round(2)
                else:
                    df_salas_stats['% del Total'] = 0
                
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    df_salas_stats.to_excel(w, index=False, sheet_name='Por Persona')
                    # También por sala
                    df_sala_stats = df_salas.groupby('room_name').agg({
                        'reservation_date': 'count'
                    }).reset_index()
                    df_sala_stats.columns = ['Sala', 'Cantidad Reservas']
                    df_sala_stats = df_sala_stats.sort_values('Cantidad Reservas', ascending=False)
                    total_sala = df_sala_stats['Cantidad Reservas'].sum()
                    if total_sala > 0:
                        df_sala_stats['% del Total'] = (df_sala_stats['Cantidad Reservas'] / total_sala * 100).round(2)
                    else:
                        df_sala_stats['% del Total'] = 0
                    df_sala_stats.to_excel(w, index=False, sheet_name='Por Sala')
                
                st.session_state['salas_report'] = b.getvalue()
                st.session_state['salas_report_name'] = "reservas_salas.xlsx"
                st.success("✅ Informe de salas generado")
            else:
                st.warning("No hay reservas de salas registradas")
        
        if 'salas_report' in st.session_state:
            st.download_button("📥 Descargar Informe de Salas", st.session_state['salas_report'], 
                             st.session_state['salas_report_name'], 
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             key="dl_salas_report")
        
        # Informe de toma de salas y cupos (ordenado por frecuencia)
        st.markdown("---")
        st.markdown("### 📊 Informe de Toma de Salas y Cupos (Por Frecuencia)")
        if st.button("Generar Informe de Toma", key="gen_toma_report"):
            df_cupos = list_reservations_df(conn)
            df_salas = get_room_reservations_df(conn)
            
            b = BytesIO()
            with pd.ExcelWriter(b, engine='openpyxl') as w:
                if not df_cupos.empty:
                    # Por persona - cupos
                    df_cupos_persona = df_cupos.groupby(['user_name', 'user_email']).agg({
                        'reservation_date': 'count'
                    }).reset_index()
                    df_cupos_persona.columns = ['Nombre', 'Correo', 'Cantidad']
                    df_cupos_persona = df_cupos_persona.sort_values('Cantidad', ascending=False)
                    total_cupos = df_cupos_persona['Cantidad'].sum()
                    if total_cupos > 0:
                        df_cupos_persona['% Asociación'] = (df_cupos_persona['Cantidad'] / total_cupos * 100).round(2)
                    else:
                        df_cupos_persona['% Asociación'] = 0
                    df_cupos_persona.to_excel(w, index=False, sheet_name='Cupos por Persona')
                
                if not df_salas.empty:
                    # Por persona - salas
                    df_salas_persona = df_salas.groupby(['user_name', 'user_email']).agg({
                        'reservation_date': 'count'
                    }).reset_index()
                    df_salas_persona.columns = ['Nombre', 'Correo', 'Cantidad']
                    df_salas_persona = df_salas_persona.sort_values('Cantidad', ascending=False)
                    total_salas = df_salas_persona['Cantidad'].sum()
                    if total_salas > 0:
                        df_salas_persona['% Asociación'] = (df_salas_persona['Cantidad'] / total_salas * 100).round(2)
                    else:
                        df_salas_persona['% Asociación'] = 0
                    df_salas_persona.to_excel(w, index=False, sheet_name='Salas por Persona')
            
            st.session_state['toma_report'] = b.getvalue()
            st.session_state['toma_report_name'] = "toma_salas_cupos.xlsx"
            st.success("✅ Informe de toma generado")
        
        if 'toma_report' in st.session_state:
            st.download_button("📥 Descargar Informe de Toma", st.session_state['toma_report'], 
                             st.session_state['toma_report_name'], 
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             key="dl_toma_report")
        
        # Informe de planos generados
        st.markdown("---")
        st.markdown("### 🗺️ Informe de Planos Generados")
        if st.button("Generar Informe de Planos", key="gen_planos_report"):
            zonas = load_zones()
            planos_data = []
            
            for piso in pisos_list:
                p_num = piso.replace("Piso ", "").strip()
                for dia in ORDER_DIAS:
                    ds = dia.lower().replace("é","e").replace("á","a")
                    fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
                    fpdf = COLORED_DIR / f"piso_{p_num}_{ds}_combined.pdf"
                    
                    zonas_piso = zonas.get(piso, [])
                    num_zonas = len(zonas_piso) if zonas_piso else 0
                    
                    planos_data.append({
                        'Piso': piso,
                        'Día': dia,
                        'Zonas Configuradas': num_zonas,
                        'PNG Generado': 'Sí' if fpng.exists() else 'No',
                        'PDF Generado': 'Sí' if fpdf.exists() else 'No',
                        'Estado': 'Completo' if (fpng.exists() or fpdf.exists()) and num_zonas > 0 else 'Pendiente'
                    })
            
            df_planos = pd.DataFrame(planos_data)
            
            b = BytesIO()
            with pd.ExcelWriter(b, engine='openpyxl') as w:
                df_planos.to_excel(w, index=False, sheet_name='Planos')
                # Resumen por piso
                df_resumen = df_planos.groupby('Piso').agg({
                    'Zonas Configuradas': 'sum',
                    'PNG Generado': lambda x: (x == 'Sí').sum(),
                    'PDF Generado': lambda x: (x == 'Sí').sum(),
                    'Estado': lambda x: (x == 'Completo').sum()
                }).reset_index()
                df_resumen.columns = ['Piso', 'Total Zonas', 'PNGs Generados', 'PDFs Generados', 'Días Completos']
                df_resumen.to_excel(w, index=False, sheet_name='Resumen por Piso')
            
            st.session_state['planos_report'] = b.getvalue()
            st.session_state['planos_report_name'] = "informe_planos.xlsx"
            st.success("✅ Informe de planos generado")
        
        if 'planos_report' in st.session_state:
            st.download_button("📥 Descargar Informe de Planos", st.session_state['planos_report'], 
                             st.session_state['planos_report_name'], 
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             key="dl_planos_report")
        
        st.markdown("---")
        st.markdown("### 📥 Descarga de Planos Individuales")
        cp, cd = st.columns(2)
        pi = cp.selectbox("Piso", pisos_list, key="pi2"); di = cd.selectbox("Día", ["Todos"]+ORDER_DIAS, key="di2")
        if di=="Todos":
            if st.button("Generar Dossier"):
                # CAMBIO: Pasar conn y logo para regenerar
                m = create_merged_pdf(pi, conn, global_logo_path)
                if m: st.session_state['dos'] = m; st.success("OK")
            if 'dos' in st.session_state: st.download_button("Descargar Dossier", st.session_state['dos'], "S.pdf", "application/pdf")
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
                with open(tf,"rb") as f: st.download_button("Descargar", f, tf.name, mm)
            else: st.warning("No existe.")

    with t4:
        nu = st.text_input("User"); np = st.text_input("Pass", type="password"); ne = st.text_input("Email")
        if st.button("Guardar", key="sc"): save_setting(conn, "admin_user", nu); save_setting(conn, "admin_pass", np); save_setting(conn, "admin_email", ne); st.success("OK")

    with t5: admin_appearance_ui(conn)
    
    with t6:
        opt = st.radio("Borrar:", ["Reservas", "Distribución", "Planos/Zonas", "TODO"], key="delete_option")
        if st.button("BORRAR", type="primary", key="delete_button"): 
            msg = perform_granular_delete(conn, opt)
            st.success(msg)
