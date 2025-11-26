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
import base64 # Se mantiene la importación para el parche de st_canvas

# ---------------------------------------------------------
# 1. PARCHE PARA STREAMLIT >= 1.39 (FIX st_canvas)
# ---------------------------------------------------------
# NOTA: Este parche es necesario para versiones específicas de Streamlit con st_canvas.
# La conversión Base64 manual se ha eliminado en el uso, pero la utilidad base sigue siendo requerida.
import streamlit.elements.lib.image_utils

if hasattr(streamlit.elements.lib.image_utils, "image_to_url"):
    _orig_image_to_url = streamlit.elements.lib.image_to_url

    @dataclass
    class WidthConfig:
        width: int

    def _patched_image_to_url(image_data, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
        if isinstance(width, int):
            width = WidthConfig(width=width)
        return _orig_image_to_url(image_data, width, clamp, channels, output_format, image_id)

    streamlit.elements.lib.image_utils.image_to_url = _patched_image_to_url

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
from modules.zones import generate_colored_plan, load_zones, save_zones, create_header_image # Se añade create_header_image si fuera necesaria
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
# 4. FUNCIONES HELPER & LÓGICA (Funciones movidas a pdfgen eliminadas)
# ---------------------------------------------------------

# La lógica de PDF y ordenamiento ahora se importa de modules.pdfgen

def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    """
    Genera una propuesta basada en una estrategia de ordenamiento.
    (Se mantiene en app.py para usar st.session_state)
    """
    eq_proc = df_equipos.copy()
    pa_proc = df_parametros.copy()
    
    # Asegurarnos de que tenemos datos numéricos para ordenar
    col_sort = next((c for c in eq_proc.columns if c.lower().strip() == "dotacion"), None)
    
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
    # ... (sin cambios) ...
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

# (Los generadores de PDF se importan de modules.pdfgen)

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

# ... (código de inicialización de DB y configuración, sin cambios) ...

# ==========================================
# A. VISTA PÚBLICA (Uso de función importada)
# ==========================================
if menu == "Vista pública":
    # ...
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            p_sel = c1.selectbox("Selecciona Piso", pisos_disponibles)
            ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS)
            pn = p_sel.replace("Piso ", "").strip()
            st.write("---")
            
            if ds == "Todos (Lunes a Viernes)":
                # LLAMADA A FUNCIÓN CONSOLIDADA
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
# B. RESERVAS (CORRECCIÓN DE ROBUSTEZ)
# ==========================================
# ...
    # ---------------------------------------------------------
    # OPCIÓN 3: GESTIONAR (ANULAR Y VER TODO) - CORRECCIÓN DE ROBUSTEZ APLICADA
    # ---------------------------------------------------------
    elif opcion_reserva == "📋 Mis Reservas y Listados":
        
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
# ... (código para mostrar resultados sin cambios) ...

# ==========================================
# E. ADMINISTRADOR
# ==========================================
# ...

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
            # OPTIMIZACIÓN: Se lee a bytes sin manipulación Base64 explícita
            img = PILImage.open(pim)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # st_canvas puede consumir el objeto bytes o la imagen PIL
            cw = 800; w, h = img.size
            ch = int(h * (cw/w)) if w>cw else h
            cw = w if w<=cw else cw

            # Se usa el objeto Image original (o la versión escalada/bytes si es necesario)
            # El parche al inicio de app.py se encarga de la conversión interna
            canvas = st_canvas(fill_color="rgba(0, 160, 74, 0.3)", stroke_width=2, stroke_color="#00A04A", background_image=img, update_streamlit=True, width=cw, height=ch, drawing_mode="rect", key=f"cv_{p_sel}")
        
            # ... (Lógica de dibujo de zonas, sin cambios) ...

    with t3:
        st.subheader("Generar Reportes de Distribución")
        # ...
        rf = st.selectbox("Formato Reporte", ["Excel", "PDF"])
        if st.button("Generar Reporte"):
            # Uso de apply_sorting_to_df y ORDER_DIAS
            df_raw = read_distribution_df(conn); df_raw = apply_sorting_to_df(df_raw, ORDER_DIAS)
            if "Excel" in rf:
                b = BytesIO()
                with pd.ExcelWriter(b) as w: df_raw.to_excel(w, index=False)
                st.session_state['rd'] = b.getvalue(); st.session_state['rn'] = "d.xlsx"; st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                # Columnas a minúsculas para compatibilidad con generate_full_pdf
                df = df_raw.rename(columns={c: c.lower() for c in df_raw.columns})
                d_data = st.session_state.get('deficit_report', [])
                # LLAMADA A FUNCIÓN CONSOLIDADA
                st.session_state['rd'] = generate_full_pdf(df, logo_path=Path(global_logo_path), deficit_data=d_data, order_dias=ORDER_DIAS)
                st.session_state['rn'] = "reporte_distribucion.pdf"; st.session_state['rm'] = "application/pdf"
            st.success("OK")
        if 'rd' in st.session_state: st.download_button("Descargar", st.session_state['rd'], st.session_state['rn'], mime=st.session_state['rm'])
        
        st.markdown("---")
        cp, cd = st.columns(2)
        pi = cp.selectbox("Piso", pisos_list, key="pi2"); di = cd.selectbox("Día", ["Todos"]+ORDER_DIAS, key="di2")
        if di=="Todos":
            if st.button("Generar Dossier"):
                # LLAMADA A FUNCIÓN CONSOLIDADA
                m = create_merged_pdf(pi, ORDER_DIAS, conn, read_distribution_df, global_logo_path, st.session_state.get('last_style_config', {}))
                if m: st.session_state['dos'] = m; st.success("OK")
            if 'dos' in st.session_state: st.download_button("Descargar Dossier", st.session_state['dos'], "S.pdf", "application/pdf")
        else:
            # ... (código igual) ...
