import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound, APIError
import pandas as pd
import datetime
import time

# --- CONFIGURACIÓN DE CONEXIÓN ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_conn():
    """Conecta a Google Sheets."""
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            client = gspread.authorize(creds)
            sheet_name = st.secrets["sheets"]["sheet_name"] 
            return client.open(sheet_name)
        return None
    except Exception as e:
        st.error(f"Error conectando a Google Sheets: {e}")
        return None

def get_worksheet(conn, sheet_name):
    """Obtiene pestaña con reintento anti-429."""
    for attempt in range(3):
        try:
            return conn.worksheet(sheet_name)
        except WorksheetNotFound:
            try:
                time.sleep(1)
                return conn.add_worksheet(title=sheet_name, rows=100, cols=20)
            except APIError:
                time.sleep(1)
                return conn.worksheet(sheet_name)
        except APIError as e:
            if "429" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            raise e
    return None

def init_db(conn):
    """Inicializa DB una sola vez."""
    if not conn: return
    sheets_config = {
        "reservations": ["user_name", "user_email", "piso", "reservation_date", "team_area", "created_at"],
        "room_reservations": ["user_name", "user_email", "piso", "room_name", "reservation_date", "start_time", "end_time", "created_at"],
        "distribution": ["piso", "equipo", "dia", "cupos", "pct", "created_at"],
        "settings": ["key", "value", "updated_at"],
        "reset_tokens": ["token", "created_at", "expires_at", "used"]
    }
    for name, headers in sheets_config.items():
        ws = get_worksheet(conn, name)
        if ws:
            try:
                if not ws.row_values(1): ws.append_row(headers)
            except: pass
        time.sleep(0.2)

# --- FUNCIONES DE LECTURA (CON CACHÉ Y LIMPIEZA) ---

@st.cache_data(ttl=60, show_spinner=False)
def read_distribution_df(_conn):
    ws = get_worksheet(_conn, "distribution")
    try:
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

def insert_distribution(conn, rows):
    ws = get_worksheet(conn, "distribution")
    ws.clear()
    ws.append_row(["piso", "equipo", "dia", "cupos", "pct", "created_at"])
    
    data = []
    now = datetime.datetime.now().isoformat()
    for r in rows:
        data.append([
            str(r.get('Piso', r.get('piso',''))), 
            str(r.get('Equipo', r.get('equipo',''))), 
            str(r.get('Día', r.get('dia',''))), 
            str(r.get('Cupos', r.get('cupos',0))), 
            str(r.get('%Distrib', r.get('pct',0))), 
            now
        ])
    
    if data: ws.append_rows(data)
    
    # --- LA SOLUCIÓN MÁGICA ---
    # Borramos el caché para que la app sepa que hay datos nuevos
    read_distribution_df.clear() 
    st.cache_data.clear() 

def clear_distribution(conn):
    ws = get_worksheet(conn, "distribution")
    ws.clear()
    read_distribution_df.clear() # Limpiar caché también aquí

# --- RESERVAS PUESTOS ---

def add_reservation(conn, name, email, piso, date_str, area, created_at):
    ws = get_worksheet(conn, "reservations")
    ws.append_row([name, email, piso, date_str, area, created_at])
    list_reservations_df.clear() # Limpiar caché al reservar

def user_has_reservation(conn, email, date_str):
    # Sin caché para validación en tiempo real
    ws = get_worksheet(conn, "reservations")
    try:
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        if df.empty: return False
        match = df[(df['user_email'].astype(str) == str(email)) & (df['reservation_date'].astype(str) == str(date_str))]
        return not match.empty
    except: return False

@st.cache_data(ttl=60, show_spinner=False)
def list_reservations_df(_conn):
    ws = get_worksheet(_conn, "reservations")
    try: return pd.DataFrame(ws.get_all_records())
    except: return pd.DataFrame()

def delete_reservation_from_db(conn, user_name, date_str, team_area):
    ws = get_worksheet(conn, "reservations")
    vals = ws.get_all_values()
    for i in range(len(vals)-1, 0, -1):
        r = vals[i]
        if len(r)>=5 and r[0]==user_name and r[3]==str(date_str) and r[4]==team_area:
            ws.delete_rows(i+1)
            list_reservations_df.clear() # Limpiar caché al borrar
            return True
    return False

def count_monthly_free_spots(conn, identifier, date_obj):
    df = list_reservations_df(conn) 
    if df.empty: return 0
    m_str = date_obj.strftime("%Y-%m")
    mask = ((df['user_email'].astype(str)==identifier)|(df['user_name'].astype(str)==identifier)) & \
           (df['reservation_date'].astype(str).str.contains(m_str)) & \
           (df['team_area']=='Cupos libres')
    return len(df[mask])

# --- SALAS ---

def add_room_reservation(conn, name, email, piso, room, date, start, end, created):
    ws = get_worksheet(conn, "room_reservations")
    ws.append_row([name, email, piso, room, date, start, end, created])
    get_room_reservations_df.clear() # Limpiar caché

@st.cache_data(ttl=60, show_spinner=False)
def get_room_reservations_df(_conn):
    ws = get_worksheet(_conn, "room_reservations")
    try: return pd.DataFrame(ws.get_all_records())
    except: return pd.DataFrame()

def delete_room_reservation_from_db(conn, user, date, room, start):
    ws = get_worksheet(conn, "room_reservations")
    vals = ws.get_all_values()
    for i in range(len(vals)-1, 0, -1):
        r = vals[i]
        if len(r)>=6 and r[0]==user and r[4]==str(date) and r[3]==room and r[5]==str(start):
            ws.delete_rows(i+1)
            get_room_reservations_df.clear() # Limpiar caché
            return True
    return False

# --- SETTINGS & TOKENS ---

def save_setting(conn, key, value):
    ws = get_worksheet(conn, "settings")
    try:
        cell = ws.find(key, in_column=1)
        ws.update_cell(cell.row, 2, value)
    except: ws.append_row([key, value, datetime.datetime.now().isoformat()])
    get_all_settings.clear() # Limpiar caché

@st.cache_data(ttl=300, show_spinner=False)
def get_all_settings(_conn):
    ws = get_worksheet(_conn, "settings")
    try: return {r['key']: r['value'] for r in ws.get_all_records()}
    except: return {}

def ensure_reset_table(conn): pass
def save_reset_token(conn, t, e): get_worksheet(conn, "reset_tokens").append_row([t, datetime.datetime.now().isoformat(), e, 0])
def validate_and_consume_token(conn, t):
    ws = get_worksheet(conn, "reset_tokens")
    try:
        cell = ws.find(t)
        if not cell: return False, "Inválido"
        row = ws.row_values(cell.row)
        if int(row[3])==1 or datetime.datetime.utcnow()>datetime.datetime.fromisoformat(row[2]): return False, "Expirado"
        ws.update_cell(cell.row, 4, 1)
        return True, "OK"
    except: return False, "Error"

def perform_granular_delete(conn, option):
    if "Reservas" in option or "TODO" in option:
        ws = get_worksheet(conn, "reservations"); ws.clear(); ws.append_row(["user_name", "user_email", "piso", "reservation_date", "team_area", "created_at"])
        list_reservations_df.clear()
        
        ws2 = get_worksheet(conn, "room_reservations"); ws2.clear(); ws2.append_row(["user_name", "user_email", "piso", "room_name", "reservation_date", "start_time", "end_time", "created_at"])
        get_room_reservations_df.clear()
        
    if "Distribución" in option or "TODO" in option:
        ws = get_worksheet(conn, "distribution"); ws.clear(); ws.append_row(["piso", "equipo", "dia", "cupos", "pct", "created_at"])
        read_distribution_df.clear()
        
    return "Datos eliminados."