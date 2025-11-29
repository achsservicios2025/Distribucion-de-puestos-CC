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
            
            if "sheets" in st.secrets and "sheet_name" in st.secrets["sheets"]:
                sheet_name = st.secrets["sheets"]["sheet_name"] 
                return client.open(sheet_name)
            else:
                st.error("Falta la configuración 'sheet_name' en st.secrets")
                return None
        return None
    except Exception as e:
        print(f"Error conectando a Google Sheets: {e}") 
        st.error(f"Error conectando a Google Sheets: {e}") 
        return None

def get_worksheet(conn, sheet_name):
    """Obtiene pestaña con reintento anti-429 y protección contra None."""
    if conn is None:
        return None

    for attempt in range(3):
        try:
            return conn.worksheet(sheet_name)
        except WorksheetNotFound:
            try:
                time.sleep(1)
                return conn.add_worksheet(title=sheet_name, rows=100, cols=20)
            except APIError:
                time.sleep(1)
                if conn: return conn.worksheet(sheet_name)
        except APIError as e:
            if "429" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            print(f"Error API (get_worksheet): {e}")
            return None 
        except Exception as e:
            print(f"Error inesperado (get_worksheet): {e}")
            return None
    return None

def init_db(conn):
    """Inicializa DB una sola vez."""
    if conn is None: return 
    
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
    if ws is None: return pd.DataFrame() 
    
    try:
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error leyendo distribution: {e}")
        return pd.DataFrame()

def insert_distribution(conn, rows):
    ws = get_worksheet(conn, "distribution")
    if ws is None: return 
    
    try:
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
        
        read_distribution_df.clear() 
        st.cache_data.clear() 
    except Exception as e:
        st.error(f"Error guardando distribución: {e}")

def clear_distribution(conn):
    ws = get_worksheet(conn, "distribution")
    if ws is None: return
    
    try:
        ws.clear()
        read_distribution_df.clear()
    except: pass

# --- RESERVAS PUESTOS ---

def add_reservation(conn, name, email, piso, date_str, area, created_at):
    ws = get_worksheet(conn, "reservations")
    if ws is None: return
    
    try:
        ws.append_row([name, email, piso, date_str, area, created_at])
        list_reservations_df.clear()
    except Exception as e:
        st.error(f"Error al reservar: {e}")

def user_has_reservation(conn, email, date_str):
    ws = get_worksheet(conn, "reservations")
    if ws is None: return False
    
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
    if ws is None: return pd.DataFrame()
    
    try: return pd.DataFrame(ws.get_all_records())
    except: return pd.DataFrame()

def delete_reservation_from_db(conn, user_name, date_str, team_area):
    ws = get_worksheet(conn, "reservations")
    if ws is None: return False
    
    try:
        # Búsqueda optimizada por fecha primero
        cell_list = ws.findall(str(date_str))
        for cell in cell_list:
            row_val = ws.row_values(cell.row)
            # row_val indexes: 0=name, 1=email, 2=piso, 3=date, 4=area
            if len(row_val) >= 5 and row_val[0] == user_name and row_val[4] == team_area:
                ws.delete_rows(cell.row)
                list_reservations_df.clear()
                return True
        return False
    except Exception as e:
        print(f"Error borrando reserva: {e}")
        return False

def count_monthly_free_spots(conn, identifier, date_obj):
    df = list_reservations_df(conn) 
    if df.empty: return 0
    
    try:
        m_str = date_obj.strftime("%Y-%m")
        mask = ((df['user_email'].astype(str)==identifier)|(df['user_name'].astype(str)==identifier)) & \
               (df['reservation_date'].astype(str).str.contains(m_str)) & \
               (df['team_area']=='Cupos libres')
        return len(df[mask])
    except: return 0

# --- SALAS ---

def add_room_reservation(conn, name, email, piso, room, date, start, end, created):
    ws = get_worksheet(conn, "room_reservations")
    if ws is None: return
    
    try:
        ws.append_row([name, email, piso, room, date, start, end, created])
        get_room_reservations_df.clear()
    except: pass

@st.cache_data(ttl=60, show_spinner=False)
def get_room_reservations_df(_conn):
    ws = get_worksheet(_conn, "room_reservations")
    if ws is None: return pd.DataFrame()
    
    try: return pd.DataFrame(ws.get_all_records())
    except: return pd.DataFrame()

def delete_room_reservation_from_db(conn, user, date, room, start):
    ws = get_worksheet(conn, "room_reservations")
    if ws is None: return False
    
    try:
        cell_list = ws.findall(str(date))
        for cell in cell_list:
            row_val = ws.row_values(cell.row)
            # Indexes: 0=name, 3=room, 4=date, 5=start
            if len(row_val) >= 6 and row_val[0] == user and row_val[3] == room and row_val[5] == str(start):
                ws.delete_rows(cell.row)
                get_room_reservations_df.clear()
                return True
        return False
    except: return False

# --- SETTINGS & TOKENS ---

def save_setting(conn, key, value):
    ws = get_worksheet(conn, "settings")
    if ws is None: return
    
    try:
        cell = ws.find(key, in_column=1)
        ws.update_cell(cell.row, 2, value)
    except Exception as e:
        if "cell must be a Cell object" in str(e):
             print(f"Error de gspread: {e}")
        try: 
            ws.append_row([key, value, datetime.datetime.now().isoformat()])
        except: 
            pass
    get_all_settings.clear()

@st.cache_data(ttl=300, show_spinner=False)
def get_all_settings(_conn):
    ws = get_worksheet(_conn, "settings")
    if ws is None: return {}
    
    try: return {r['key']: r['value'] for r in ws.get_all_records()}
    except: return {}

def ensure_reset_table(conn): 
    if conn is None: return
    pass

def save_reset_token(conn, t, e): 
    ws = get_worksheet(conn, "reset_tokens")
    if ws:
        try: ws.append_row([t, datetime.datetime.now().isoformat(), e, 0])
        except: pass

def validate_and_consume_token(conn, t):
    ws = get_worksheet(conn, "reset_tokens")
    if ws is None: return False, "Error de conexión"
    
    try:
        cell = ws.find(t)
        if not cell: return False, "Inválido"
        row = ws.row_values(cell.row)
        if int(row[3])==1 or datetime.datetime.utcnow()>datetime.datetime.fromisoformat(row[2]): return False, "Expirado"
        ws.update_cell(cell.row, 4, 1)
        return True, "OK"
    except: return False, "Error"

def perform_granular_delete(conn, option):
    if conn is None: return "Error: No hay conexión."
    
    msg = []
    if "RESERVAS" in option or "TODO" in option:
        ws = get_worksheet(conn, "reservations")
        if ws:
            ws.clear()
            ws.append_row(["user_name", "user_email", "piso", "reservation_date", "team_area", "created_at"])
            list_reservations_df.clear()
            msg.append("Reservas eliminadas")
        
        ws2 = get_worksheet(conn, "room_reservations")
        if ws2:
            ws2.clear()
            ws2.append_row(["user_name", "user_email", "piso", "room_name", "reservation_date", "start_time", "end_time", "created_at"])
            get_room_reservations_df.clear()
            msg.append("Salas eliminadas")
        
    if "DISTRIBUCION" in option or "TODO" in option:
        ws = get_worksheet(conn, "distribution")
        if ws:
            ws.clear()
            ws.append_row(["piso", "equipo", "dia", "cupos", "pct", "created_at"])
            read_distribution_df.clear()
            msg.append("Distribución eliminada")
        
    return ", ".join(msg) + "."
