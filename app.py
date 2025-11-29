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
# Importaciones condicionales para evitar errores
try:
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
    from modules.seats import compute_distribution_from_excel, get_ideal_distribution_proposal, calculate_distribution_stats
    from modules.emailer import send_reservation_email
    from modules.rooms import generate_time_slots, check_room_conflict
    from modules.zones import generate_colored_plan, load_zones, save_zones
    from modules.pdfgen import generate_full_pdf, create_merged_pdf
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    # Definir funciones placeholder para evitar errores
    def get_conn(): return None
    def init_db(conn): pass
    def get_admin_credentials(conn): return "admin", "admin123"
    # ... otras funciones placeholder según sea necesario

# ---------------------------------------------------------
# 4. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

# Verificar e inicializar conexión
try:
    conn = get_conn()
    if conn is None:
        st.error("No se pudo establecer conexión con la base de datos")
        st.stop()
        
    if "db_initialized" not in st.session_state:
        with st.spinner('Conectando a Google Sheets...'):
            init_db(conn)
        st.session_state["db_initialized"] = True
except Exception as e:
    st.error(f"Error inicializando la aplicación: {e}")
    st.stop()

# Aplicar estilos
try:
    apply_appearance_styles(conn)
except:
    pass  # Continuar incluso si hay error en estilos

if "app_settings" not in st.session_state:
    try:
        st.session_state["app_settings"] = get_all_settings(conn)
    except:
        st.session_state["app_settings"] = {}

settings = st.session_state["app_settings"]

site_title = settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios")
global_logo_path = settings.get("logo_path", "static/logo.png")

# Mostrar logo y título
col1, col2 = st.columns([1, 5])
try:
    if os.path.exists(global_logo_path):
        col1.image(global_logo_path, width=150)
except:
    pass
col2.title(site_title)

# ---------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN
# ---------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
PLANOS_DIR = Path("planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

# Crear directorios necesarios
for directory in [DATA_DIR, PLANOS_DIR, COLORED_DIR]:
    directory.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 5. FUNCIONES HELPER & LÓGICA
# ---------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    if not isinstance(text, str): 
        return str(text)
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
    if df.empty: 
        return df
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
    if col_piso: 
        sort_cols.append(col_piso)
    if col_dia: 
        sort_cols.append(col_dia)
    
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df

def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    try:
        eq_proc = df_equipos.copy()
        pa_proc = df_parametros.copy()
        col_sort = None
        for c in eq_proc.columns:
            if c.lower().strip() == "dotacion":
                col_sort = c
                break
        if not col_sort and strategy != "random": 
            strategy = "random"

        if strategy == "random": 
            eq_proc = eq_proc.sample(frac=1).reset_index(drop=True)
        elif strategy == "size_desc" and col_sort: 
            eq_proc = eq_proc.sort_values(by=col_sort, ascending=False).reset_index(drop=True)
        elif strategy == "size_asc" and col_sort: 
            eq_proc = eq_proc.sort_values(by=col_sort, ascending=True).reset_index(drop=True)

        rows, deficit_report = compute_distribution_from_excel(eq_proc, pa_proc, 2)
        return rows, deficit_report
    except Exception as e:
        st.error(f"Error en distribución: {e}")
        return [], []

def show_distribution_insights(rows, deficit_data):
    try:
        df = pd.DataFrame(rows)
        st.subheader("📊 Métricas de la Distribución")
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            st.metric("Total Cupos Asignados", df['cupos'].sum())
        with c2: 
            st.metric("Cupos Libres", df[df['equipo'] == 'Cupos libres']['cupos'].sum())
        with c3: 
            st.metric("Equipos Asignados", df[df['equipo'] != 'Cupos libres']['equipo'].nunique())
        with c4: 
            st.metric("Uniformidad (σ)", f"{df.groupby('dia')['cupos'].sum().std():.1f}")
        
        st.subheader("📈 Distribución por Día")
        cupos_por_dia = df.groupby('dia')['cupos'].sum().reindex(ORDER_DIAS)
        fig, ax = plt.subplots(figsize=(10, 4))
        cupos_por_dia.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel('Total Cupos')
        ax.set_title('Distribución de Cupos por Día de la Semana')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error mostrando insights: {e}")

def calculate_weekly_usage_summary(distrib_df):
    try:
        if distrib_df.empty: 
            return pd.DataFrame()
        
        equipo_col = None
        cupos_col = None
        for col in distrib_df.columns:
            cl = col.lower()
            if 'equipo' in cl: 
                equipo_col = col
            elif 'cupos' in cl: 
                cupos_col = col
        
        if not all([equipo_col, cupos_col]):
            return pd.DataFrame()
        
        equipos_df = distrib_df[distrib_df[equipo_col] != "Cupos libres"]
        if equipos_df.empty: 
            return pd.DataFrame()
        
        weekly = equipos_df.groupby(equipo_col).agg({cupos_col: 'sum'}).reset_index()
        weekly.columns = ['Equipo', 'Total Cupos Semanales']
        
        # Calcular porcentaje de distribución semanal
        total_semanal = weekly['Total Cupos Semanales'].sum()
        if total_semanal > 0:
            weekly['% Distribución Semanal'] = (weekly['Total Cupos Semanales'] / total_semanal * 100).round(1)
        else:
            weekly['% Distribución Semanal'] = 0
        
        return weekly.sort_values('Total Cupos Semanales', ascending=False)
    except Exception as e:
        st.error(f"Error calculando resumen semanal: {e}")
        return pd.DataFrame()

def clean_reservation_df(df, tipo="puesto"):
    try:
        if df.empty: 
            return df
        cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at', 'registro', 'id.1']]
        df = df.drop(columns=cols_drop, errors='ignore')
        
        if tipo == "puesto":
            rename_map = {
                'user_name': 'Nombre', 
                'user_email': 'Correo', 
                'piso': 'Piso', 
                'reservation_date': 'Fecha Reserva', 
                'team_area': 'Ubicación'
            }
            df = df.rename(columns=rename_map)
            desired_cols = ['Fecha Reserva', 'Piso', 'Ubicación', 'Nombre', 'Correo']
            existing_cols = [c for c in desired_cols if c in df.columns]
            return df[existing_cols]
            
        elif tipo == "sala":
            rename_map = {
                'user_name': 'Nombre', 
                'user_email': 'Correo', 
                'piso': 'Piso', 
                'room_name': 'Sala', 
                'reservation_date': 'Fecha', 
                'start_time': 'Inicio', 
                'end_time': 'Fin'
            }
            df = df.rename(columns=rename_map)
            desired_cols = ['Fecha', 'Inicio', 'Fin', 'Sala', 'Piso', 'Nombre', 'Correo']
            existing_cols = [c for c in desired_cols if c in df.columns]
            return df[existing_cols]
        return df
    except Exception as e:
        st.error(f"Error limpiando DataFrame: {e}")
        return df

# ---------------------------------------------------------
# 6. GENERADORES DE PDF (funciones locales como fallback)
# ---------------------------------------------------------
def local_generate_full_pdf(distrib_df, logo_path, deficit_data=None):
    """Versión local de generate_full_pdf como fallback"""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(True, 15)
        
        # Página 1: Distribución diaria
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        
        try:
            if Path(logo_path).exists():
                pdf.image(str(logo_path), x=10, y=8, w=30)
        except:
            pass
            
        pdf.ln(25)
        pdf.cell(0, 10, clean_pdf_text("Informe de Distribución"), ln=True, align='C')
        pdf.ln(6)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, clean_pdf_text("1. Detalle de Distribución Diaria"), ln=True)
        
        # Tabla Diaria
        pdf.set_font("Arial", 'B', 9)
        widths = [30, 60, 25, 25, 25]
        headers = ["Piso", "Equipo", "Día", "Cupos", "%Distrib Diario"]    
        for w, h in zip(widths, headers): 
            pdf.cell(w, 6, clean_pdf_text(h), 1)
        pdf.ln()
        
        pdf.set_font("Arial", '', 9)
        distrib_df_sorted = apply_sorting_to_df(distrib_df)
        
        for _, r in distrib_df_sorted.iterrows():
            piso = str(r.get('piso', ''))
            equipo = str(r.get('equipo', ''))[:40]
            dia = str(r.get('dia', ''))
            cupos = str(r.get('cupos', ''))
            pct = str(r.get('pct', ''))
            
            pdf.cell(widths[0], 6, clean_pdf_text(piso), 1)
            pdf.cell(widths[1], 6, clean_pdf_text(equipo), 1)
            pdf.cell(widths[2], 6, clean_pdf_text(dia), 1)
            pdf.cell(widths[3], 6, clean_pdf_text(cupos), 1)
            pdf.cell(widths[4], 6, clean_pdf_text(f"{pct}%"), 1)
            pdf.ln()
        
        # Resumen semanal
        pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
        
        weekly_summary = calculate_weekly_usage_summary(distrib_df)
        if not weekly_summary.empty:
            pdf.set_font("Arial", 'B', 9)
            w_wk = [80, 40, 40]
            h_wk = ["Equipo", "Total Semanal", "% Distrib Semanal"]
            start_x = 25
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): 
                pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln()
            
            pdf.set_font("Arial", '', 9)
            for _, row in weekly_summary.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:30]), 1)
                pdf.cell(w_wk[1], 6, clean_pdf_text(str(int(row["Total Cupos Semanales"]))), 1)
                pdf.cell(w_wk[2], 6, clean_pdf_text(f"{row['% Distribución Semanal']}%"), 1)
                pdf.ln()
        
        return pdf.output(dest='S').encode('latin-1')
        
    except Exception as e:
        st.error(f"Error generando PDF: {e}")
        return None

def local_create_merged_pdf(piso_sel, conn, global_logo_path):
    """Versión local de create_merged_pdf como fallback"""
    try:
        p_num = piso_sel.replace("Piso ", "").strip()
        pdf = FPDF()
        pdf.set_auto_page_break(True, 15)
        found_any = False
        
        df = read_distribution_df(conn)
        base_config = st.session_state.get('last_style_config', {})

        for dia in ORDER_DIAS:
            subset = df[(df['piso'] == piso_sel) & (df['dia'] == dia)]
            if subset.empty:
                continue
                
            current_seats = dict(zip(subset['equipo'], subset['cupos']))
            day_config = base_config.copy()
            if not day_config.get("subtitle_text"): 
                day_config["subtitle_text"] = f"Día: {dia}"
            
            img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)
            if img_path and Path(img_path).exists():
                found_any = True
                pdf.add_page()
                try: 
                    pdf.image(str(img_path), x=10, y=10, w=190)
                except: 
                    pass
                    
        if not found_any: 
            return None
            
        return pdf.output(dest='S').encode('latin-1', 'replace')
        
    except Exception as e:
        st.error(f"Error creando PDF combinado: {e}")
        return None

# ---------------------------------------------------------
# 7. MODALES Y FUNCIONES DE UI
# ---------------------------------------------------------
@st.dialog("Confirmar Anulación de Puesto")
def confirm_delete_dialog(conn, usuario, fecha_str, area, piso):
    st.warning(f"¿Anular reserva de puesto?\n\n👤 {usuario} | 📅 {fecha_str}\n📍 {piso} - {area}")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", width="stretch", key="yes_p"):
        if delete_reservation_from_db(conn, usuario, fecha_str, area): 
            st.success("Eliminada")
            st.rerun()
    if c2.button("Cancelar", width="stretch", key="no_p"): 
        st.rerun()

@st.dialog("Confirmar Anulación de Sala")
def confirm_delete_room_dialog(conn, usuario, fecha_str, sala, inicio):
    st.warning(f"¿Anular reserva de sala?\n\n👤 {usuario} | 📅 {fecha_str}\n🏢 {sala} ({inicio})")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", width="stretch", key="yes_s"):
        if delete_room_reservation_from_db(conn, usuario, fecha_str, sala, inicio): 
            st.success("Eliminada")
            st.rerun()
    if c2.button("Cancelar", width="stretch", key="no_s"): 
        st.rerun()

def generate_token(): 
    return uuid.uuid4().hex[:8].upper()

# ---------------------------------------------------------
# 8. EDITOR DE ZONAS MEJORADO
# ---------------------------------------------------------
def fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, img_width, img_height):
    """Editor manual con imagen ajustada y keys únicos"""
    
    p_num = p_sel.replace("Piso ", "").strip()

    st.subheader("🎯 Modo de Dibujo Manual")
    
    # Mostrar imagen de referencia
    try:
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
    except Exception as e:
        st.error(f"Error mostrando imagen: {e}")
    
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
            if "1" in p_sel: 
                salas_piso = ["Sala Reuniones Pequeña Piso 1", "Sala Reuniones Grande Piso 1"]
            elif "2" in p_sel: 
                salas_piso = ["Sala Reuniones Piso 2"]
            elif "3" in p_sel: 
                salas_piso = ["Sala Reuniones Piso 3"]
            eqs = eqs + salas_piso
            
            equipo = st.selectbox("Equipo / Sala", eqs, key=f"team_sel_adv_{p_sel}")
            
            # MOSTRAR CUPOS INMEDIATAMENTE AL SELECCIONAR EQUIPO
            if equipo and equipo in current_seats_dict:
                st.info(f"📊 Cupos asignados para {equipo}: {current_seats_dict[equipo]} cupos")
            elif equipo:
                st.info("ℹ️ Este equipo no tiene cupos asignados para el día seleccionado")
                
            color = st.color_picker("Color", "#00A04A", key=f"col_pick_adv_{p_sel}")
            
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
                if p_sel not in zonas:
                    zonas[p_sel] = []
                zonas[p_sel].append({
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
            tit = st.text_input("Título", f"Distribución {p_sel}", key=f"tit_final_{p_sel}")
            sub = st.text_input("Subtítulo", f"Día: {d_sel}", key=f"sub_final_{p_sel}")
            
            show_legend = st.checkbox("Mostrar leyenda", True, key=f"show_leg_{p_sel}")
            legend_align = st.selectbox("Alineación leyenda", ["Izquierda", "Centro", "Derecha"], 
                                      index=0, key=f"leg_align_{p_sel}")
        with col_style2:
            bg = st.color_picker("Fondo", "#FFFFFF", key=f"bg_final_{p_sel}")
            tx = st.color_picker("Texto", "#000000", key=f"tx_final_{p_sel}")
        
        lg = st.checkbox("Logo", True, key=f"lg_final_{p_sel}")
        logo_align = st.selectbox("Alineación logo", ["Izquierda", "Centro", "Derecha"], 
                                index=0, key=f"logo_align_{p_sel}")
        
    if st.button("Generar Vista Previa", key=f"btn_gen_{p_sel}"):
        conf = {
            "title_text": tit, 
            "subtitle_text": sub, 
            "bg_color": bg, 
            "title_color": tx, 
            "use_logo": lg,
            "logo_align": logo_align,
            "show_legend": show_legend,
            "legend_align": legend_align
        }
        current_seats = current_seats_dict
        generate_colored_plan(p_sel, d_sel, current_seats, "PNG", conf, global_logo_path)
        
        ds = d_sel.lower().replace("é", "e").replace("á", "a")
        fpng = COLORED_DIR / f"piso_{p_num}_{ds}_combined.png"
        if fpng.exists():
            st.image(str(fpng), caption="Vista Previa", width=700)

def enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path):
    """Editor mejorado de zonas con fallbacks"""
    try:
        p_num = p_sel.replace("Piso ", "").strip()
        file_base = f"piso{p_num}"
        pim = PLANOS_DIR / f"{file_base}.png"
        if not pim.exists(): 
            pim = PLANOS_DIR / f"{file_base}.jpg"
        if not pim.exists(): 
            pim = PLANOS_DIR / f"Piso{p_num}.png"

        if not pim.exists():
            st.error(f"❌ No se encontró el plano para {p_sel}")
            return

        img = PILImage.open(pim)
        w, h = img.size
        
        st.image(img, caption=f"Plano Base {p_sel}", width=700) 
        
        # Usar siempre el editor manual mejorado
        fallback_manual_editor(p_sel, d_sel, zonas, df_d, img, w, h)
        
    except Exception as e:
        st.error(f"Error en editor de zonas: {e}")

# ---------------------------------------------------------
# 9. MENÚ PRINCIPAL DE LA APLICACIÓN
# ---------------------------------------------------------
menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

# ==========================================
# A. VISTA PÚBLICA
# ==========================================
if menu == "Vista pública":
    st.header("Cupos y Planos")
    
    try:
        df = read_distribution_df(conn)
        
        if not df.empty:
            cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at']]
            df_view = df.drop(columns=cols_drop, errors='ignore')
            df_view = apply_sorting_to_df(df_view)
            pisos_disponibles = sort_floors(df["piso"].unique())
        else:
            df_view = df
            pisos_disponibles = ["Piso 1"]

        if df.empty: 
            st.info("Sin datos de distribución.")
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
                p_sel = c1.selectbox("Selecciona Piso", pisos_disponibles, key="plano_piso")
                ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS, key="plano_dia")
                pn = p_sel.replace("Piso ", "").strip()
                st.write("---")
                
                if ds == "Todos (Lunes a Viernes)":
                    m = local_create_merged_pdf(p_sel, conn, global_logo_path)
                    if m: 
                        st.success("✅ Dossier disponible.")
                        st.download_button("📥 Descargar Semana (PDF)", m, f"Planos_{p_sel}_Semana.pdf", "application/pdf", use_container_width=True)
                    else: 
                        st.warning("Sin planos generados.")
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
                        if fpng.exists(): 
                            opts.append("Imagen (PNG)")
                        if fpdf.exists(): 
                            opts.append("Documento (PDF)")
                        
                        if opts:
                            if fpng.exists(): 
                                st.image(str(fpng), width=550, caption=f"{p_sel} - {ds}")
                            sf = st.selectbox("Formato:", opts, key="dl_pub")
                            tf = fpng if "PNG" in sf else fpdf
                            mim = "image/png" if "PNG" in sf else "application/pdf"
                            with open(tf,"rb") as f: 
                                st.download_button(f"📥 Descargar {sf}", f, tf.name, mim, use_container_width=True)
                        else: 
                            st.warning("No generado.")

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
                        avg_percent = weekly_summary['% Distribución Semanal'].mean()
                        st.metric("Promedio % Distribución", f"{avg_percent:.1f}%")
                else:
                    st.info("No hay datos suficientes para generar el resumen semanal")
                    
    except Exception as e:
        st.error(f"Error en vista pública: {e}")

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
        try:
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
                            mask = (all_res["reservation_date"].astype(str) == str(fe)) & (all_res["piso"] == pi) & (all_res["team_area"] == "Cupos libres")
                            ocupados = len(all_res[mask])
                        disponibles = total_cupos - ocupados
                    
                    if not hay_config:
                        st.warning(f"⚠️ El {pi} no tiene habilitados 'Cupos libres' para los días {dn}.")
                    else:
                        if disponibles > 0: 
                            st.success(f"✅ **HAY CUPO: Quedan {disponibles} puestos** (Total: {total_cupos}).")
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
                                    try:
                                        send_reservation_email(em, "Confirmación Puesto", msg.replace("\n","<br>"))
                                    except:
                                        pass  # Continuar incluso si falla el email
                                    st.rerun()
        except Exception as e:
            st.error(f"Error en reserva de puestos: {e}")

    elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
        try:
            st.subheader("Agendar Sala")
            c_sala, c_fecha = st.columns(2)
            
            salas_opciones = [
                "Sala Reuniones Pequeña Piso 1",
                "Sala Reuniones Grande Piso 1", 
                "Sala Reuniones Piso 2",
                "Sala Reuniones Piso 3"
            ]
            
            sl = c_sala.selectbox("Selecciona Sala", salas_opciones)
            
            if "Piso 1" in sl:
                pi_s = "Piso 1"
            elif "Piso 2" in sl:
                pi_s = "Piso 2" 
            elif "Piso 3" in sl:
                pi_s = "Piso 3"
            else:
                pi_s = "Piso 1"
            
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
                        if e_s: 
                            try:
                                send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))
                            except:
                                pass  # Continuar incluso si falla el email
        except Exception as e:
            st.error(f"Error en reserva de salas: {e}")

    elif opcion_reserva == "📋 Mis Reservas y Listados":
        try:
            st.subheader("Buscar y Cancelar mis reservas")
            q = st.text_input("Ingresa tu Correo o Nombre para buscar:")
            
            if q:
                dp = clean_reservation_df(list_reservations_df(conn), "puesto")
                mp = pd.DataFrame()
                if not dp.empty and 'Nombre' in dp.columns and 'Correo' in dp.columns:
                    mp = dp[(dp['Nombre'].str.lower().str.contains(q.lower())) | (dp['Correo'].str.lower().str.contains(q.lower()))]

                ds = clean_reservation_df(get_room_reservations_df(conn), "sala")
                ms = pd.DataFrame()
                if not ds.empty and 'Nombre' in ds.columns and 'Correo' in ds.columns:
                    ms = ds[(ds['Nombre'].str.lower().str.contains(q.lower())) | (ds['Correo'].str.lower().str.contains(q.lower()))]
                
                if mp.empty and ms.empty: 
                    st.warning("No encontré reservas con esos datos.")
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
                st.dataframe(clean_reservation_df(list_reservations_df(conn), "puesto"), hide_index=True, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True) 
                st.subheader("Reserva de salas") 
                st.dataframe(clean_reservation_df(get_room_reservations_df(conn), "sala"), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Error en listado de reservas: {e}")

# ==========================================
# C. ADMINISTRADOR
# ==========================================
elif menu == "Administrador":
    st.header("Panel de Administración")
    
    try:
        admin_user, admin_pass = get_admin_credentials(conn)
        if "is_admin" not in st.session_state: 
            st.session_state["is_admin"] = False
        
        if not st.session_state["is_admin"]:
            st.subheader("Inicio de Sesión")
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
                if st.button("Solicitar Token"):
                    re = settings.get("admin_email", "")
                    if re and em_chk.lower() == re.lower():
                        t = generate_token()
                        save_reset_token(conn, t, (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat())
                        try:
                            send_reservation_email(re, "Token de Recuperación", f"Token: {t}")
                            st.success("Token enviado al correo registrado.")
                        except:
                            st.info(f"Token generado: {t} (El servicio de email no está disponible)")
                    else: 
                        st.error("Email no coincide con el registrado.")
                
                tk = st.text_input("Token")
                nu = st.text_input("Nuevo Usuario")
                np = st.text_input("Nueva Contraseña", type="password")
                if st.button("Cambiar Credenciales"):
                    ok, m = validate_and_consume_token(conn, tk)
                    if ok: 
                        save_setting(conn, "admin_user", nu)
                        save_setting(conn, "admin_pass", np)
                        st.success("Credenciales actualizadas correctamente")
                    else: 
                        st.error(f"Error: {m}")
            st.stop()

        if st.button("Cerrar Sesión"):
            st.session_state["is_admin"] = False
            st.rerun()

        t1, t2, t3, t4, t5, t6 = st.tabs(["Excel", "Editor Visual", "Informes", "Config", "Apariencia", "Mantenimiento"])
        
        with t1:
            st.subheader("Generador de Distribución Inteligente")
            
            c_up, c_strat = st.columns([2, 1])
            up = c_up.file_uploader("Subir archivo Excel (Hojas: 'Equipos', 'Parámetros')", type=["xlsx"])
            
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

            # Inicializar session state
            if 'excel_equipos' not in st.session_state: 
                st.session_state['excel_equipos'] = None
            if 'excel_params' not in st.session_state: 
                st.session_state['excel_params'] = None
            if 'proposal_rows' not in st.session_state: 
                st.session_state['proposal_rows'] = None
            if 'proposal_deficit' not in st.session_state: 
                st.session_state['proposal_deficit'] = None
            if 'multiple_proposals' not in st.session_state: 
                st.session_state['multiple_proposals'] = []

            if up:
                try:
                    if st.button("📂 Procesar Inicial", type="primary"):
                        df_eq = pd.read_excel(up, "Equipos")
                        
                        if ignore_params:
                            st.session_state['excel_equipos'] = df_eq
                            st.session_state['excel_params'] = None
                            
                            proposals = []
                            for i in range(3):
                                rows, deficit = get_ideal_distribution_proposal(df_eq, strategy=sel_strat_code, variant=i)
                                stats = calculate_distribution_stats(rows, df_eq)
                                proposals.append({
                                    'rows': rows,
                                    'deficit': deficit,
                                    'name': f"Opción {i+1} - {estrategia}",
                                    'stats': stats
                                })
                            
                            st.session_state['multiple_proposals'] = proposals
                            st.session_state['proposal_rows'] = proposals[0]['rows']
                            st.session_state['proposal_deficit'] = proposals[0]['deficit']
                            
                        else:
                            df_pa = pd.read_excel(up, "Parámetros")
                            st.session_state['excel_equipos'] = df_eq
                            st.session_state['excel_params'] = df_pa
                            rows, deficit = get_distribution_proposal(df_eq, df_pa, strategy=sel_strat_code)
                            st.session_state['proposal_rows'] = rows
                            st.session_state['proposal_deficit'] = deficit
                            st.session_state['multiple_proposals'] = []
                            
                        st.rerun()
                        
                except Exception as e: 
                    st.error(f"Error al leer el Excel: {e}")

            if st.session_state['proposal_rows'] is not None:
                st.divider()
                
                if st.session_state['multiple_proposals'] and len(st.session_state['multiple_proposals']) > 1:
                    st.subheader("🎯 Opciones de Distribución Generadas")
                    
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
                    else: 
                        st.warning("No se generaron asignaciones.")
                    
                    if st.session_state.get('multiple_proposals'):
                        show_distribution_insights(st.session_state['proposal_rows'], st.session_state['proposal_deficit'])
                        
                with t_def:
                    if st.session_state['proposal_deficit']:
                        def_df = pd.DataFrame(st.session_state['proposal_deficit'])
                        st.dataframe(def_df, use_container_width=True)
                    else: 
                        st.info("Sin conflictos.")

                st.markdown("---")
                c_actions = st.columns([1, 1, 1])
                if c_actions[0].button("🔄 Probar otra suerte"):
                    with st.spinner("Generando..."):
                        if st.session_state.get('multiple_proposals'):
                            proposals = []
                            for i in range(3):
                                rows, deficit = get_ideal_distribution_proposal(
                                    st.session_state['excel_equipos'], 
                                    strategy=sel_strat_code, 
                                    variant=i+3
                                )
                                stats = calculate_distribution_stats(rows, st.session_state['excel_equipos'])
                                proposals.append({
                                    'rows': rows,
                                    'deficit': deficit,
                                    'name': f"Opción {i+1} - {estrategia}",
                                    'stats': stats
                                })
                            st.session_state['multiple_proposals'] = proposals
                            st.session_state['proposal_rows'] = proposals[0]['rows']
                            st.session_state['proposal_deficit'] = proposals[0]['deficit']
                        else:
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
                        st.info("En modo ideal, la distribución ya está optimizada.")
                    else:
                        NUM_INTENTOS = 20
                        my_bar = st.progress(0, text="Optimizando...")
                        best_rows = None
                        best_deficit = None
                        min_unfairness_score = 999999
                        for i in range(NUM_INTENTOS):
                            r, d = get_distribution_proposal(st.session_state['excel_equipos'], st.session_state['excel_params'], strategy="random")
                            unfairness_score = sum([1 for x in d]) if d else 0
                            if unfairness_score < min_unfairness_score:
                                min_unfairness_score = unfairness_score
                                best_rows = r
                                best_deficit = d
                            my_bar.progress(int((i + 1) / NUM_INTENTOS * 100))
                        st.session_state['proposal_rows'] = best_rows
                        st.session_state['proposal_deficit'] = best_deficit
                        my_bar.empty()
                        st.rerun()

                if c_actions[2].button("💾 Guardar Definitivo", type="primary"):
                    clear_distribution(conn)
                    insert_distribution(conn, st.session_state['proposal_rows'])
                    if st.session_state['proposal_deficit']: 
                        st.session_state['deficit_report'] = st.session_state['proposal_deficit']
                    elif 'deficit_report' in st.session_state: 
                        del st.session_state['deficit_report']
                    st.success("Distribución guardada exitosamente.")
                    st.balloons()
                    st.rerun()

        with t2:
            try:
                zonas = load_zones()
                c1, c2 = st.columns(2)
                df_d = read_distribution_df(conn)
                pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
                p_sel = c1.selectbox("Piso", pisos_list, key="editor_piso")
                d_sel = c2.selectbox("Día Ref.", ORDER_DIAS, key="editor_dia")
                
                enhanced_zone_editor(p_sel, d_sel, zonas, df_d, global_logo_path)
            except Exception as e:
                st.error(f"Error en editor visual: {e}")

        with t3:
            st.subheader("Generar Reportes")
            
            with st.expander("📊 Informes de Uso y Reservas", expanded=True):
                st.subheader("Resumen de Uso por Persona/Equipo")
                
                reservas_puestos = clean_reservation_df(list_reservations_df(conn), "puesto")
                if not reservas_puestos.empty:
                    st.markdown("#### 🪑 Uso de Cupos Libres por Persona")
                    uso_personas = reservas_puestos.groupby('Nombre').agg({
                        'Fecha Reserva': 'count',
                        'Correo': 'first'
                    }).reset_index()
                    uso_personas = uso_personas.rename(columns={'Fecha Reserva': 'Reservas Totales'})
                    uso_personas = uso_personas.sort_values('Reservas Totales', ascending=False)
                    st.dataframe(uso_personas, hide_index=True, use_container_width=True)
                
                reservas_salas = clean_reservation_df(get_room_reservations_df(conn), "sala")
                if not reservas_salas.empty:
                    st.markdown("#### 🏢 Uso de Salas por Persona")
                    uso_salas = reservas_salas.groupby('Nombre').agg({
                        'Fecha': 'count',
                        'Correo': 'first',
                        'Sala': lambda x: ', '.join(x.unique())
                    }).reset_index()
                    uso_salas = uso_salas.rename(columns={'Fecha': 'Reservas Totales'})
                    uso_salas = uso_salas.sort_values('Reservas Totales', ascending=False)
                    st.dataframe(uso_salas, hide_index=True, use_container_width=True)

            rf = st.selectbox("Formato de Reporte", ["Excel", "PDF"], key="report_format")
            if st.button("Generar Reporte de Distribución"):
                df_raw = read_distribution_df(conn)
                if "Excel" in rf:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_raw.to_excel(writer, sheet_name='Distribución', index=False)
                        
                        weekly_summary = calculate_weekly_usage_summary(df_raw)
                        if not weekly_summary.empty:
                            weekly_summary.to_excel(writer, sheet_name='Resumen Semanal', index=False)
                        
                        if not reservas_puestos.empty:
                            uso_personas.to_excel(writer, sheet_name='Uso Cupos', index=False)
                        if not reservas_salas.empty:
                            uso_salas.to_excel(writer, sheet_name='Uso Salas', index=False)
                            
                    st.download_button("📥 Descargar Excel", output.getvalue(), "reporte_completo.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                else:
                    d_data = st.session_state.get('deficit_report', [])
                    pdf_bytes = local_generate_full_pdf(df_raw, global_logo_path, deficit_data=d_data)
                    if pdf_bytes:
                        st.download_button("📥 Descargar PDF", pdf_bytes, "reporte.pdf", "application/pdf", use_container_width=True)
                    else:
                        st.error("Error generando PDF")

        with t4:
            st.subheader("Configuración de Credenciales")
            nu = st.text_input("Nuevo Usuario", key="new_user")
            np = st.text_input("Nueva Contraseña", type="password", key="new_pass")
            ne = st.text_input("Email de Administrador", key="admin_email")
            if st.button("Guardar Credenciales", key="save_creds"): 
                save_setting(conn, "admin_user", nu)
                save_setting(conn, "admin_pass", np)
                save_setting(conn, "admin_email", ne)
                st.success("Credenciales guardadas correctamente")

        with t5: 
            try:
                admin_appearance_ui(conn)
            except Exception as e:
                st.error(f"Error en configuración de apariencia: {e}")
                
        with t6:
            st.subheader("Opciones de Mantenimiento")
            
            st.markdown("### 🗑️ Eliminación Selectiva de Reservas")
            
            tipo_eliminacion = st.selectbox(
                "Tipo de reserva a eliminar:",
                ["Puestos", "Salas", "Ambos"],
                key="delete_type"
            )
            
            if tipo_eliminacion in ["Puestos", "Ambos"]:
                st.markdown("#### Reservas de Puestos")
                reservas_puestos = clean_reservation_df(list_reservations_df(conn), "puesto")
                if not reservas_puestos.empty:
                    reservas_seleccionadas = st.multiselect(
                        "Selecciona reservas de puestos a eliminar:",
                        options=reservas_puestos.to_dict('records'),
                        format_func=lambda x: f"{x['Nombre']} - {x['Fecha Reserva']} - {x['Piso']} - {x['Ubicación']}",
                        key="delete_puestos"
                    )
                    
                    if reservas_seleccionadas and st.button("Eliminar Reservas de Puestos Seleccionadas", type="primary", key="btn_delete_puestos"):
                        for reserva in reservas_seleccionadas:
                            if delete_reservation_from_db(conn, reserva['Nombre'], reserva['Fecha Reserva'], reserva['Ubicación']):
                                st.success(f"Reserva de {reserva['Nombre']} eliminada")
                        st.rerun()

            if tipo_eliminacion in ["Salas", "Ambos"]:
                st.markdown("#### Reservas de Salas")
                reservas_salas = clean_reservation_df(get_room_reservations_df(conn), "sala")
                if not reservas_salas.empty:
                    reservas_salas_seleccionadas = st.multiselect(
                        "Selecciona reservas de salas a eliminar:",
                        options=reservas_salas.to_dict('records'),
                        format_func=lambda x: f"{x['Nombre']} - {x['Fecha']} - {x['Sala']} ({x['Inicio']}-{x['Fin']})",
                        key="delete_salas"
                    )
                    
                    if reservas_salas_seleccionadas and st.button("Eliminar Reservas de Salas Seleccionadas", type="primary", key="btn_delete_salas"):
                        for reserva in reservas_salas_seleccionadas:
                            if delete_room_reservation_from_db(conn, reserva['Nombre'], reserva['Fecha'], reserva['Sala'], reserva['Inicio']):
                                st.success(f"Reserva de sala {reserva['Sala']} eliminada")
                        st.rerun()
            
            st.markdown("---")
            st.subheader("Opciones de Borrado Masivo")
            
            opcion_borrado = st.selectbox(
                "Selecciona qué deseas borrar:",
                ["Reservas", "Distribución", "Planos/Zonas", "TODO"],
                key="mass_delete"
            )
            
            if st.button("Ejecutar Borrado Masivo", type="primary", key="btn_mass_delete"):
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
            else:
                st.info("No hay datos de distribución cargados")
                
    except Exception as e:
        st.error(f"Error en panel de administración: {e}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sistema de Gestión de Espacios - ACHS Servicios"
    "</div>", 
    unsafe_allow_html=True
)
