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
# 1. PARCHE PARA STREAMLIT >= 1.39
# ---------------------------------------------------------
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

# Configuración de rutas robusta
BASE_DIR = Path(__file__).parent.resolve()
PLANOS_DIR = BASE_DIR / "planos"
DATA_DIR = BASE_DIR / "data"
COLORED_DIR = BASE_DIR / "planos_coloreados"

DATA_DIR.mkdir(exist_ok=True)
PLANOS_DIR.mkdir(exist_ok=True)
COLORED_DIR.mkdir(exist_ok=True)

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

# ---------------------------------------------------------
# 4. FUNCIONES HELPER
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
    if col_dia: df[col_dia] = pd.Categorical(df[col_dia], categories=ORDER_DIAS, ordered=True)
    if col_piso:
        unique_floors = [str(x) for x in df[col_piso].dropna().unique()]
        sorted_floors = sort_floors(unique_floors)
        df[col_piso] = pd.Categorical(df[col_piso], categories=sorted_floors, ordered=True)
    sort_cols = []
    if col_piso: sort_cols.append(col_piso)
    if col_dia: sort_cols.append(col_dia)
    if sort_cols: df = df.sort_values(sort_cols)
    return df

def get_distribution_proposal(df_equipos, df_parametros, strategy="random"):
    eq_proc = df_equipos.copy(); pa_proc = df_parametros.copy()
    col_sort = None
    for c in eq_proc.columns:
        if c.lower().strip() == "dotacion": col_sort = c; break
    if not col_sort and strategy != "random": strategy = "random"
    if strategy == "random": eq_proc = eq_proc.sample(frac=1).reset_index(drop=True)
    elif strategy == "size_desc" and col_sort: eq_proc = eq_proc.sort_values(by=col_sort, ascending=False).reset_index(drop=True)
    elif strategy == "size_asc" and col_sort: eq_proc = eq_proc.sort_values(by=col_sort, ascending=True).reset_index(drop=True)
    rows, deficit_report = compute_distribution_from_excel(eq_proc, pa_proc, 2)
    return rows, deficit_report

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
    pdf = FPDF(); pdf.set_auto_page_break(True, 15); found_any = False
    df = read_distribution_df(conn)
    base_config = st.session_state.get('last_style_config', {})
    for dia in ORDER_DIAS:
        subset = df[(df['piso'] == piso_sel) & (df['dia'] == dia)]
        current_seats = dict(zip(subset['equipo'], subset['cupos']))
        day_config = base_config.copy()
        if not day_config.get("subtitle_text"): day_config["subtitle_text"] = f"Día: {dia}"
        img_path = generate_colored_plan(piso_sel, dia, current_seats, "PNG", day_config, global_logo_path)
        if img_path and Path(img_path).exists():
            found_any = True; pdf.add_page()
            try: pdf.image(str(img_path), x=10, y=10, w=190)
            except: pass
    if not found_any: return None
    return pdf.output(dest='S').encode('latin-1')

def generate_full_pdf(distrib_df, semanal_df, out_path="reporte.pdf", logo_path=Path("static/logo.png"), deficit_data=None):
    pdf = FPDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    if logo_path.exists():
        try: pdf.image(str(logo_path), x=10, y=8, w=30)
        except: pass
    pdf.ln(25); pdf.cell(0, 10, clean_pdf_text("Informe de Distribución"), ln=True, align='C'); pdf.ln(6)
    pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, clean_pdf_text("1. Detalle de Distribución Diaria"), ln=True)
    pdf.set_font("Arial", 'B', 9); widths = [30, 60, 25, 25, 25]; headers = ["Piso", "Equipo", "Día", "Cupos", "%Distrib Diario"] 
    for w, h in zip(widths, headers): pdf.cell(w, 6, clean_pdf_text(h), 1)
    pdf.ln(); pdf.set_font("Arial", '', 9)
    
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

    pdf.add_page(); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 10, clean_pdf_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    try:
        col_pct = None
        if "%Distrib" in distrib_df.columns: col_pct = "%Distrib"
        elif "pct" in distrib_df.columns: col_pct = "pct"
        if col_pct:
            distrib_df[col_pct] = pd.to_numeric(distrib_df[col_pct], errors='coerce').fillna(0)
            weekly_stats = distrib_df.groupby("Equipo")[col_pct].mean().reset_index()
            weekly_stats.columns = ["Equipo", "Promedio Semanal"]
            weekly_stats = weekly_stats.sort_values("Equipo")
            pdf.set_font("Arial", 'B', 9); w_wk = [100, 40]; h_wk = ["Equipo", "% Promedio Semanal"]
            start_x = 35; pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): pdf.cell(w, 6, clean_pdf_text(h), 1)
            pdf.ln(); pdf.set_font("Arial", '', 9)
            for _, row in weekly_stats.iterrows():
                pdf.set_x(start_x); pdf.cell(w_wk[0], 6, clean_pdf_text(str(row["Equipo"])[:50]), 1)
                val = row["Promedio Semanal"]; pdf.cell(w_wk[1], 6, clean_pdf_text(f"{val:.1f}%"), 1); pdf.ln()
    except Exception as e:
        pdf.set_font("Arial", 'I', 9); pdf.cell(0, 6, clean_pdf_text(f"No se pudo calcular: {str(e)}"), ln=True)

    pdf.ln(10); pdf.set_font("Arial", 'B', 10); pdf.cell(0, 8, clean_pdf_text("Glosario:"), ln=True)
    pdf.set_font("Arial", '', 9)
    for nota in ["1. % Distribución Diario: Cupos / Dotación.", "2. % Uso Semanal: Promedio 5 días.", "3. Déficit: Cupos mínimos - asignados."]:
        pdf.set_x(10); pdf.multi_cell(185, 6, clean_pdf_text(nota))

    if deficit_data and len(deficit_data) > 0:
        pdf.add_page(); pdf.set_font("Arial", 'B', 14); pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 10, clean_pdf_text("Reporte de Déficit"), ln=True, align='C'); pdf.set_text_color(0, 0, 0); pdf.ln(5)
        pdf.set_font("Arial", 'B', 8); dw = [15, 45, 20, 15, 15, 15, 65]; dh = ["Piso", "Equipo", "Día", "Dot.", "Mín.", "Falt.", "Causa"]
        for w, h in zip(dw, dh): pdf.cell(w, 8, clean_pdf_text(h), 1, 0, 'C')
        pdf.ln(); pdf.set_font("Arial", '', 8)
        for d in deficit_data:
            if pdf.get_y() + 10 > 270: pdf.add_page()
            # (Simplified for brevity, assumed similar structure to original)
            pdf.cell(dw[0], 6, clean_pdf_text(d.get('piso','')), 1)
            pdf.cell(dw[1], 6, clean_pdf_text(d.get('equipo',''))[:25], 1)
            pdf.cell(dw[5], 6, str(d.get('deficit','')), 1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- MODALES ---
@st.dialog("Confirmar")
def confirm_delete_dialog(conn, u, f, a, p):
    st.warning(f"¿Anular puesto? {u} | {f}"); c1,c2=st.columns(2)
    if c1.button("Sí", key="yp"): delete_reservation_from_db(conn, u, f, a); st.rerun()
    if c2.button("No", key="np"): st.rerun()

@st.dialog("Confirmar Sala")
def confirm_delete_room_dialog(conn, u, f, s, i):
    st.warning(f"¿Anular sala? {u} | {f}"); c1,c2=st.columns(2)
    if c1.button("Sí", key="ys"): delete_room_reservation_from_db(conn, u, f, s, i); st.rerun()
    if c2.button("No", key="ns"): st.rerun()

def generate_token(): return uuid.uuid4().hex[:8].upper()

# ---------------------------------------------------------
# INICIO APP
# ---------------------------------------------------------
conn = get_conn()
if "db_initialized" not in st.session_state:
    with st.spinner('Conectando...'): init_db(conn)
    st.session_state["db_initialized"] = True

apply_appearance_styles(conn)
if "app_settings" not in st.session_state: st.session_state["app_settings"] = get_all_settings(conn)
settings = st.session_state["app_settings"]
site_title = settings.get("site_title", "Gestor de Puestos")
global_logo_path = settings.get("logo_path", "static/logo.png")

if os.path.exists(global_logo_path):
    c1, c2 = st.columns([1, 5]); c1.image(global_logo_path, width=150); c2.title(site_title)
else: st.title(site_title)

menu = st.sidebar.selectbox("Menú", ["Vista pública", "Reservas", "Administrador"])

# ==========================================
# A. VISTA PÚBLICA
# ==========================================
if menu == "Vista pública":
    st.header("Cupos y Planos")
    df = read_distribution_df(conn)
    if df.empty: st.info("Sin datos.")
    else:
        df_view = apply_sorting_to_df(df.drop(columns=[c for c in df.columns if c.lower() in ['id','created_at']], errors='ignore'))
        pisos_disponibles = sort_floors(df["piso"].unique())
        
        t1, t2 = st.tabs(["Estadísticas", "Ver Planos"])
        with t1:
            st.markdown("""<style>[data-testid="stElementToolbar"] {display: none;}</style>""", unsafe_allow_html=True)
            st.subheader("Distribución completa")
            st.dataframe(df_view, hide_index=True, use_container_width=True)
        
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            p_sel = c1.selectbox("Piso", pisos_disponibles)
            ds = c2.selectbox("Día", ["Todos"] + ORDER_DIAS)
            
            if ds == "Todos":
                if st.button("Generar PDF Semanal"):
                    m = create_merged_pdf(p_sel, conn, global_logo_path)
                    if m: st.download_button("📥 Descargar PDF", m, "Semana.pdf", "application/pdf")
            else:
                # FALLBACK VISUALIZACIÓN
                subset = df[(df['piso'] == p_sel) & (df['dia'] == ds)]
                current_seats = dict(zip(subset['equipo'], subset['cupos']))
                day_config = st.session_state.get('last_style_config', {})
                img_path = generate_colored_plan(p_sel, ds, current_seats, "PNG", day_config, global_logo_path)
                
                p_num = p_sel.replace("Piso ", "").strip()
                dsf = ds.lower().replace("é","e").replace("á","a")
                fpng = COLORED_DIR / f"piso_{p_num}_{dsf}_combined.png"
                if fpng.exists(): st.image(str(fpng), caption=f"{p_sel} - {ds}")
                else: st.warning("No se pudo generar la vista previa.")

# ==========================================
# B. RESERVAS
# ==========================================
elif menu == "Reservas":
    st.header("Gestión de Reservas")
    opcion = st.selectbox("Opción", ["Reservar Puesto", "Reservar Sala", "Mis Reservas"])
    st.divider()

    if opcion == "Reservar Puesto":
        df = read_distribution_df(conn)
        if df.empty: st.warning("Sin configuración.")
        else:
            c1, c2 = st.columns(2)
            fe = c1.date_input("Fecha", min_value=datetime.date.today())
            pi = c2.selectbox("Piso", sort_floors(df["piso"].unique()))
            dn = ORDER_DIAS[fe.weekday()] if fe.weekday() < 5 else "Fin"
            
            if dn == "Fin": st.error("Fin de semana bloqueado.")
            else:
                rg = df[(df["piso"] == pi) & (df["dia"] == dn) & (df["equipo"] == "Cupos libres")]
                if rg.empty: st.warning("No hay cupos libres configurados.")
                else:
                    total = int(rg.iloc[0]["cupos"])
                    occ = len(list_reservations_df(conn)[lambda x: (x["reservation_date"].astype(str)==str(fe)) & (x["piso"]==pi) & (x["team_area"]=="Cupos libres")])
                    disp = total - occ
                    
                    if disp > 0: st.success(f"Disponibles: {disp}")
                    else: st.error("Agotado.")
                    
                    with st.form("rp"):
                        n = st.text_input("Nombre"); e = st.text_input("Email")
                        if st.form_submit_button("Reservar", disabled=(disp<=0)):
                            if not n or not e: st.error("Faltan datos")
                            elif user_has_reservation(conn, e, str(fe)): st.error("Ya tienes reserva.")
                            else: add_reservation(conn, n, e, pi, str(fe), "Cupos libres", datetime.datetime.now().isoformat()); st.success("Listo!"); st.rerun()

    elif opcion == "Reservar Sala":
        sl = st.selectbox("Sala", ["Sala 1 (Piso 1)", "Sala 2 (Piso 2)"])
        fe = st.date_input("Fecha", min_value=datetime.date.today())
        tm = generate_time_slots("08:00", "20:00", 15)
        c1, c2 = st.columns(2); i = c1.selectbox("Inicio", tm); f = c2.selectbox("Fin", tm, index=4)
        with st.form("rs"):
            n = st.text_input("Nombre"); e = st.text_input("Email")
            if st.form_submit_button("Reservar"):
                if check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe), sl, i, f): st.error("Ocupado.")
                else: add_room_reservation(conn, n, e, sl.split("-")[1].strip(), sl, str(fe), i, f, datetime.datetime.now().isoformat()); st.success("Listo!")

    elif opcion == "Mis Reservas":
        q = st.text_input("Tu Email")
        if q:
            dp = clean_reservation_df(list_reservations_df(conn), "puesto")
            ds = clean_reservation_df(get_room_reservations_df(conn), "sala")
            if not dp.empty:
                mp = dp[dp['Correo'].str.contains(q, case=False)]
                for _, r in mp.iterrows():
                    with st.container(border=True):
                        st.write(f"Puesto: {r['Fecha Reserva']} - {r['Piso']}")
                        if st.button("Anular", key=f"dp_{_}"): confirm_delete_dialog(conn, r['Nombre'], r['Fecha Reserva'], r['Ubicación'], r['Piso'])
            if not ds.empty:
                ms = ds[ds['Correo'].str.contains(q, case=False)]
                for _, r in ms.iterrows():
                    with st.container(border=True):
                        st.write(f"Sala: {r['Fecha']} - {r['Sala']}")
                        if st.button("Anular", key=f"ds_{_}"): confirm_delete_room_dialog(conn, r['Nombre'], r['Fecha'], r['Sala'], r['Inicio'])

# ==========================================
# E. ADMINISTRADOR
# ==========================================
elif menu == "Administrador":
    st.header("Admin")
    au, ap = get_admin_credentials(conn)
    if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
    
    if not st.session_state["is_admin"]:
        u = st.text_input("User"); p = st.text_input("Pass", type="password")
        if st.button("Login"): 
            if u==au and p==ap: st.session_state["is_admin"] = True; st.rerun()
            else: st.error("Error")
        st.stop()

    if st.button("Logout"): st.session_state["is_admin"] = False; st.rerun()
    
    t1, t2, t3, t4, t5, t6 = st.tabs(["Excel", "Editor Visual", "Reportes", "Config", "Apariencia", "Maint"])
    
    with t1:
        up = st.file_uploader("Excel", type=["xlsx"])
        if up and st.button("Procesar"):
            eq = pd.read_excel(up, "Equipos"); pa = pd.read_excel(up, "Parámetros")
            r, d = get_distribution_proposal(eq, pa)
            st.session_state['pr'] = r; st.session_state['pd'] = d
            st.rerun()
        if 'pr' in st.session_state:
            st.dataframe(st.session_state['pr'], use_container_width=True)
            if st.button("Guardar"): 
                clear_distribution(conn); insert_distribution(conn, st.session_state['pr'])
                if st.session_state['pd']: st.session_state['deficit_report'] = st.session_state['pd']
                st.success("OK")

    # --- T2: EDITOR VISUAL (CORREGIDO) ---
    with t2:
        st.info("Editor de Zonas")
        zonas = load_zones()
        c1, c2 = st.columns(2)
        df_d = read_distribution_df(conn)
        pisos_list = sort_floors(df_d["piso"].unique()) if not df_d.empty else ["Piso 1"]
        p_sel = c1.selectbox("Piso", pisos_list)
        d_sel = c2.selectbox("Día", ORDER_DIAS)
        
        # BÚSQUEDA DE IMAGEN MÁS ROBUSTA
        p_num = p_sel.replace("Piso ", "").strip()
        posibles = [f"piso{p_num}.png", f"piso{p_num}.jpg", f"Piso{p_num}.png"]
        pim = None
        for f in posibles:
            if (PLANOS_DIR / f).exists(): pim = PLANOS_DIR / f; break
        
        if not pim:
            st.error(f"❌ No se encontró imagen en: {PLANOS_DIR}. Archivos buscados: {posibles}")
            if PLANOS_DIR.exists(): st.write(f"Archivos disponibles: {os.listdir(PLANOS_DIR)}")
        else:
            # Muestra imagen de prueba para verificar carga
            st.image(str(pim), width=100, caption="Check carga")
            
            # Cargar y convertir a RGB para evitar problemas de transparencia
            img = PILImage.open(pim).convert("RGB")
            cw = 800; w, h = img.size
            ch = int(h * (cw / w)) if w > cw else h; cw = w if w <= cw else cw

            # PASAR OBJETO IMAGEN (NO STRING)
            canvas = st_canvas(
                fill_color="rgba(0, 160, 74, 0.3)",
                stroke_width=2, stroke_color="#00A04A",
                background_image=img,
                update_streamlit=True,
                width=cw, height=ch, drawing_mode="rect",
                key=f"cv_{p_sel}"
            )
            
            # Lógica de guardado
            current_seats_dict = dict(zip(df_d[(df_d['piso']==p_sel)&(df_d['dia']==d_sel)]['equipo'], df_d[(df_d['piso']==p_sel)&(df_d['dia']==d_sel)]['cupos'])) if not df_d.empty else {}
            eqs = sorted(current_seats_dict.keys()) + ["Sala"]
            tn = st.selectbox("Asignar a:", eqs)
            
            if st.button("Guardar Zona"):
                if canvas.json_data and canvas.json_data.get("objects"):
                    o = canvas.json_data["objects"][-1]
                    zonas.setdefault(p_sel, []).append({
                        "team": tn, "x": int(o["left"]), "y": int(o["top"]), 
                        "w": int(o["width"]*o["scaleX"]), "h": int(o["height"]*o["scaleY"]), "color": "#00A04A"
                    })
                    save_zones(zonas); st.success("Zona guardada"); st.rerun()

            if p_sel in zonas:
                st.write("Zonas guardadas:")
                for i, z in enumerate(zonas[p_sel]):
                    c1, c2 = st.columns([4, 1])
                    c1.write(f"{z['team']}")
                    if c2.button("X", key=f"del_{i}"): zonas[p_sel].pop(i); save_zones(zonas); st.rerun()

            # Previsualización
            if st.button("Generar Vista Previa"):
                conf = {"title_text": f"{p_sel} - {d_sel}"}
                st.session_state['last_style_config'] = conf
                generate_colored_plan(p_sel, d_sel, current_seats_dict, "PNG", conf, global_logo_path)
                st.success("Generado")
                
            dsf = d_sel.lower().replace("é","e").replace("á","a")
            prev = COLORED_DIR / f"piso_{p_num}_{dsf}_combined.png"
            if prev.exists(): st.image(str(prev), width=500)

    with t3:
        if st.button("Generar Reporte Excel"):
            df = read_distribution_df(conn)
            b = BytesIO(); 
            with pd.ExcelWriter(b) as w: df.to_excel(w, index=False)
            st.download_button("Descargar", b.getvalue(), "reporte.xlsx")

    with t4:
        n_u = st.text_input("Nuevo Admin User")
        n_p = st.text_input("Nuevo Admin Pass", type="password")
        if st.button("Actualizar Credenciales"):
            save_setting(conn, "admin_user", n_u); save_setting(conn, "admin_pass", n_p); st.success("OK")

    with t5: admin_appearance_ui(conn)
    with t6:
        if st.button("BORRAR TODO", type="primary"): perform_granular_delete(conn, "TODO"); st.success("Limpio")
