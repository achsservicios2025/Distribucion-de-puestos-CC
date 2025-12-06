# modules/database.py
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound, APIError
import pandas as pd
import datetime
import time
import re

# --- CONFIGURACIÓN DE CONEXIÓN ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# =========================================================
# Helpers
# =========================================================
def _to_plain(v):
    try:
        if hasattr(v, "to_pydatetime"):
            v = v.to_pydatetime()
    except Exception:
        pass

    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()

    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass

    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    return str(v) if not isinstance(v, (str, int, float, bool)) else v


def _norm_piso(p):
    """Normaliza piso a formato 'Piso X' (string)."""
    if p is None:
        return ""
    s = str(p).strip()
    if not s:
        return ""
    m = re.findall(r"\d+", s)
    if m:
        num = str(int(m[0]))
        return f"Piso {num}"
    s_low = s.lower()
    if s_low.startswith("piso"):
        return "Piso " + s[4:].strip()
    return s


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip().replace("%", "").replace(",", ".")
        if s.lower() in ("", "nan", "none"):
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip().replace(",", ".")
        if s.lower() in ("", "nan", "none"):
            return default
        return int(float(s))
    except Exception:
        return default


def _ensure_headers(ws, headers):
    try:
        ws.clear()
        ws.append_row(headers)
        return True
    except Exception:
        return False


@st.cache_resource
def get_conn():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Faltan secretos: gcp_service_account")
            return None

        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)

        if "sheets" not in st.secrets or "sheet_name" not in st.secrets["sheets"]:
            st.error("Falta la configuración sheets.sheet_name en st.secrets")
            return None

        sheet_name = st.secrets["sheets"]["sheet_name"]
        return client.open(sheet_name)

    except Exception as e:
        st.error(f"Error conectando a Google Sheets: {e}")
        return None


def get_worksheet(conn, sheet_name):
    """Obtiene pestaña con reintento anti-429 y protección contra None."""
    if conn is None:
        return None

    for attempt in range(4):
        try:
            return conn.worksheet(sheet_name)

        except WorksheetNotFound:
            try:
                time.sleep(1)
                return conn.add_worksheet(title=sheet_name, rows=200, cols=40)
            except Exception:
                try:
                    return conn.worksheet(sheet_name)
                except Exception:
                    return None

        except APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            return None

        except Exception:
            return None

    return None


# =========================================================
# Init (crear sheets + headers)
# =========================================================
def init_db(conn):
    """Inicializa DB una sola vez."""
    if conn is None:
        return

    sheets_config = {
        "reservations": ["user_name", "user_email", "piso", "reservation_date", "team_area", "created_at"],
        "room_reservations": ["user_name", "user_email", "piso", "room_name", "reservation_date", "start_time", "end_time", "created_at"],
        "distribution": ["piso", "equipo", "dia", "cupos", "dotacion", "% uso diario", "% uso semanal", "created_at"],
        "settings": ["key", "value", "updated_at"],
        "reset_tokens": ["token", "created_at", "expires_at", "used"],
    }

    for name, headers in sheets_config.items():
        ws = get_worksheet(conn, name)
        if ws:
            try:
                first = ws.row_values(1)
                if not first:
                    ws.append_row(headers)
            except Exception:
                pass
        time.sleep(0.2)


# =========================================================
# READS (cache)
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def read_distribution_df(_conn):
    ws = get_worksheet(_conn, "distribution")
    if ws is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def list_reservations_df(_conn):
    ws = get_worksheet(_conn, "reservations")
    if ws is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_room_reservations_df(_conn):
    ws = get_worksheet(_conn, "room_reservations")
    if ws is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_all_settings(_conn):
    ws = get_worksheet(_conn, "settings")
    if ws is None:
        return {}
    try:
        recs = ws.get_all_records()
        out = {}
        for r in recs:
            k = str(r.get("key", "")).strip()
            v = str(r.get("value", "")).strip()
            if k:
                out[k] = v
        return out
    except Exception:
        return {}


# =========================================================
# WRITES / MUTATIONS
# =========================================================
def insert_distribution(conn, rows):
    ws = get_worksheet(conn, "distribution")
    if ws is None:
        return

    headers = ["piso", "equipo", "dia", "cupos", "dotacion", "% uso diario", "% uso semanal", "created_at"]

    try:
        _ensure_headers(ws, headers)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data = []

        for r in rows or []:
            piso = r.get("piso", r.get("Piso", ""))
            equipo = r.get("equipo", r.get("Equipo", ""))
            dia = r.get("dia", r.get("Día", r.get("Dia", "")))
            cupos = r.get("cupos", r.get("Cupos", 0))

            dotacion = r.get("dotacion", r.get("Dotación", r.get("Dotacion", r.get("dotación", None))))

            uso_diario = r.get("% uso diario", r.get("uso_diario", r.get("pct_uso_diario", r.get("%uso_diario", None))))
            uso_semanal = r.get("% uso semanal", r.get("uso_semanal", r.get("pct_uso_semanal", r.get("%uso_semanal", None))))

            # Normalización
            piso_norm = _norm_piso(piso)
            equipo_s = str(equipo).strip()
            dia_s = str(dia).strip()

            cupos_i = _safe_int(cupos, 0)

            # IMPORTANTE:
            # - si dotacion/uso_* vienen None (ej: "Cupos libres"), los guardamos como "" (vacío), no 0.
            dot_i = _safe_int(dotacion, None)
            uso_d_f = _safe_float(uso_diario, None)
            uso_s_f = _safe_float(uso_semanal, None)

            data.append([
                _to_plain(piso_norm),
                _to_plain(equipo_s),
                _to_plain(dia_s),
                _to_plain(cupos_i),
                _to_plain("" if dot_i is None else dot_i),
                _to_plain("" if uso_d_f is None else uso_d_f),
                _to_plain("" if uso_s_f is None else uso_s_f),
                _to_plain(now),
            ])

        if data:
            ws.append_rows(data, value_input_option="USER_ENTERED")

        read_distribution_df.clear()
        st.cache_data.clear()

    except Exception as e:
        st.error(f"Error guardando distribución: {e}")


def clear_distribution(conn):
    ws = get_worksheet(conn, "distribution")
    if ws is None:
        return
    try:
        headers = ["piso", "equipo", "dia", "cupos", "dotacion", "% uso diario", "% uso semanal", "created_at"]
        _ensure_headers(ws, headers)
        read_distribution_df.clear()
    except Exception:
        pass


# =========================================================
# Reservas puestos
# =========================================================
def add_reservation(conn, name, email, piso, date_str, area, created_at):
    ws = get_worksheet(conn, "reservations")
    if ws is None:
        return
    try:
        ws.append_row([
            _to_plain(name),
            _to_plain(email),
            _to_plain(_norm_piso(piso)),
            _to_plain(date_str),
            _to_plain(area),
            _to_plain(created_at),
        ], value_input_option="USER_ENTERED")
        list_reservations_df.clear()
    except Exception as e:
        st.error(f"Error al reservar: {e}")


def user_has_reservation(conn, email, date_str):
    ws = get_worksheet(conn, "reservations")
    if ws is None:
        return False
    try:
        df = pd.DataFrame(ws.get_all_records())
        if df.empty:
            return False
        match = df[
            (df["user_email"].astype(str) == str(email)) &
            (df["reservation_date"].astype(str) == str(date_str))
        ]
        return not match.empty
    except Exception:
        return False


def delete_reservation_from_db(conn, user_mail, date_str, team_area):
    ws = get_worksheet(conn, "reservations")
    if ws is None:
        return False
    try:
        vals = ws.get_all_values()
        for i in range(len(vals) - 1, 0, -1):
            r = vals[i]
            if len(r) >= 5 and r[0] == str(user_mail) and r[3] == str(date_str) and r[4] == str(team_area):
                ws.delete_rows(i + 1)
                list_reservations_df.clear()
                return True
        return False
    except Exception:
        return False


def count_monthly_free_spots(conn, identifier, date_obj):
    df = list_reservations_df(conn)
    if df.empty:
        return 0
    try:
        m_str = date_obj.strftime("%Y-%m")
        mask = (
            ((df["user_email"].astype(str) == str(identifier)) | (df["user_name"].astype(str) == str(identifier))) &
            (df["reservation_date"].astype(str).str.contains(m_str))
        )
        return int(len(df[mask]))
    except Exception:
        return 0


# =========================================================
# Reservas salas
# =========================================================
def add_room_reservation(conn, name, email, piso, room, date, start, end, created):
    ws = get_worksheet(conn, "room_reservations")
    if ws is None:
        return
    try:
        ws.append_row([
            _to_plain(name),
            _to_plain(email),
            _to_plain(_norm_piso(piso)),
            _to_plain(room),
            _to_plain(date),
            _to_plain(start),
            _to_plain(end),
            _to_plain(created),
        ], value_input_option="USER_ENTERED")
        get_room_reservations_df.clear()
    except Exception as e:
        st.error(f"Error al reservar sala: {e}")


def delete_room_reservation_from_db(conn, user, date, room, start):
    ws = get_worksheet(conn, "room_reservations")
    if ws is None:
        return False
    try:
        vals = ws.get_all_values()
        for i in range(len(vals) - 1, 0, -1):
            r = vals[i]
            if len(r) >= 6 and r[0] == str(user) and r[4] == str(date) and r[3] == str(room) and r[5] == str(start):
                ws.delete_rows(i + 1)
                get_room_reservations_df.clear()
                return True
        return False
    except Exception:
        return False


# =========================================================
# Settings & Tokens
# =========================================================
def save_setting(conn, key, value):
    ws = get_worksheet(conn, "settings")
    if ws is None:
        return

    key_s = str(key).strip()
    val_s = _to_plain(value)

    try:
        cell = ws.find(key_s, in_column=1)
        ws.update_cell(cell.row, 2, val_s)
        ws.update_cell(cell.row, 3, datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception:
        try:
            ws.append_row([key_s, val_s, datetime.datetime.now(datetime.timezone.utc).isoformat()],
                          value_input_option="USER_ENTERED")
        except Exception:
            pass

    get_all_settings.clear()


def ensure_reset_table(conn):
    return


def save_reset_token(conn, t, e):
    ws = get_worksheet(conn, "reset_tokens")
    if ws:
        try:
            ws.append_row([_to_plain(t), datetime.datetime.now(datetime.timezone.utc).isoformat(), _to_plain(e), 0],
                          value_input_option="USER_ENTERED")
        except Exception:
            pass


def validate_and_consume_token(conn, t):
    ws = get_worksheet(conn, "reset_tokens")
    if ws is None:
        return False, "Error de conexión"

    try:
        cell = ws.find(str(t))
        if not cell:
            return False, "Inválido"

        row = ws.row_values(cell.row)
        if len(row) < 4:
            return False, "Formato inválido"

        used = int(_safe_int(row[3], 0) or 0)
        expires_at = row[2]

        now = datetime.datetime.now(datetime.timezone.utc)
        exp = datetime.datetime.fromisoformat(expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=datetime.timezone.utc)

        if used == 1 or now > exp:
            return False, "Expirado"

        ws.update_cell(cell.row, 4, 1)
        return True, "OK"

    except Exception:
        return False, "Error"


# =========================================================
# Borrado granular
# =========================================================
def perform_granular_delete(conn, option):
    if conn is None:
        return "Error: No hay conexión."

    msg = []

    if "Reservas" in option or "TODO" in option:
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

    if "Distribución" in option or "TODO" in option:
        ws = get_worksheet(conn, "distribution")
        if ws:
            headers = ["piso", "equipo", "dia", "cupos", "dotacion", "% uso diario", "% uso semanal", "created_at"]
            _ensure_headers(ws, headers)
            read_distribution_df.clear()
            msg.append("Distribución eliminada")

    return ", ".join(msg) + "."


# =========================================================
# Borrado individual de distribution
# =========================================================
def delete_distribution_row(conn, piso, equipo, dia):
    ws = get_worksheet(conn, "distribution")
    if ws is None:
        return False

    piso_n = _norm_piso(piso)
    equipo_s = str(equipo).strip()
    dia_s = str(dia).strip()

    try:
        vals = ws.get_all_values()
        if len(vals) <= 1:
            return False

        header = [h.strip().lower() for h in vals[0]]

        def _idx(name, fallback):
            try:
                return header.index(name)
            except ValueError:
                return fallback

        i_p = _idx("piso", 0)
        i_e = _idx("equipo", 1)
        i_d = _idx("dia", 2)

        deleted_any = False
        for i in range(len(vals) - 1, 0, -1):
            r = vals[i]
            if len(r) <= max(i_p, i_e, i_d):
                continue
            if _norm_piso(r[i_p]) == piso_n and str(r[i_e]).strip() == equipo_s and str(r[i_d]).strip() == dia_s:
                ws.delete_rows(i + 1)
                deleted_any = True

        if deleted_any:
            read_distribution_df.clear()
        return deleted_any

    except Exception:
        return False


def delete_distribution_rows_by_indices(conn, indices):
    ws = get_worksheet(conn, "distribution")
    if ws is None:
        return False

    try:
        if not indices:
            return False

        for idx in sorted(set(indices), reverse=True):
            ws.delete_rows(int(idx) + 2)

        read_distribution_df.clear()
        return True

    except Exception:
        return False




