import streamlit as st
import streamlit
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
from typing import Optional
import numpy as np
import random

# ---------------------------------------------------------
# 1. PARCHE PARA STREAMLIT >= 1.39 (MANTIENE COMPATIBILIDAD ST_CANVAS)
# ---------------------------------------------------------
from dataclasses import dataclass

@dataclass
class WidthConfig:
    width: int

try:
    import streamlit.elements.lib.image_utils as _img_utils

    if hasattr(_img_utils, "image_to_url"):
        _orig_image_to_url = _img_utils.image_to_url

        def _patched_image_to_url(image_data, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
            if isinstance(image_data, str):
                return image_data
            if isinstance(width, int):
                width = WidthConfig(width=width)
            return _orig_image_to_url(image_data, width, clamp, channels, output_format, image_id)

        _img_utils.image_to_url = _patched_image_to_url

    # Parche adicional para st_image (usado por st_canvas internamente)
    try:
        import streamlit.elements.image as st_image_module
        if hasattr(st_image_module, "image_to_url"):
            _orig_st_image_to_url = st_image_module.image_to_url

            def _patched_st_image_to_url(image_data, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
                if isinstance(image_data, str):
                    return image_data
                if isinstance(width, int):
                    width = WidthConfig(width=width)
                return _orig_st_image_to_url(image_data, width, clamp, channels, output_format, image_id)

            st_image_module.image_to_url = _patched_st_image_to_url
    except Exception:
        pass

except Exception:
    pass

def resolve_logo_source(raw_path: Optional[str], logo_b64: Optional[str]):
    """
    Devuelve bytes o URL del logo.
    Prioriza base64 (subido desde admin), luego URL, luego rutas locales.
    """
    if logo_b64:
        try:
            return base64.b64decode(logo_b64)
        except Exception:
            pass

    if raw_path:
        raw_path = raw_path.strip()
        if raw_path.lower().startswith(("http://", "https://")):
            return raw_path
        raw_path = raw_path.replace("\\", "/")
    else:
        raw_path = ""

    candidates = []
    if raw_path:
        candidates.append(raw_path)

    candidates.extend([
        "static/logo.png",
        str(Path("static/logo.png")),
        str(Path("static") / "logo.png"),
    ])

    for candidate in candidates:
        candidate = candidate.replace("\\", "/")
        path_obj = Path(candidate)
        if path_obj.exists() and path_obj.is_file():
            try:
                return path_obj.read_bytes()
            except Exception:
                continue

    return None

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
    ensure_reset_table, save_reset_token, validate_and_consume_token,
    get_worksheet
)

# Importar funciones opcionales para borrado individual (pueden no existir en versiones antiguas)
try:
    from modules.database import delete_distribution_row, delete_distribution_rows_by_indices
except ImportError:
    # Si no existen, crear funciones stub que usen perform_granular_delete
    def delete_distribution_row(conn, piso, equipo, dia):
        """Elimina una fila específica de distribución (fallback)"""
        # Fallback: no se puede borrar individualmente sin estas funciones
        return False
    
    def delete_distribution_rows_by_indices(conn, indices):
        """Elimina múltiples filas de distribución (fallback)"""
        # Fallback: no se puede borrar individualmente sin estas funciones
        return False
from modules.auth import get_admin_credentials
from modules.layout import admin_appearance_ui, apply_appearance_styles
from modules.seats import compute_distribution_from_excel, compute_distribution_variants
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 3. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

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

# ---------------------------------------------------------
# FUNCIONES DE DISTRIBUCIÓN CORREGIDAS
# ---------------------------------------------------------
def _get_team_and_dotacion_cols(df_eq: pd.DataFrame):
    if df_eq is None or df_eq.empty:
        return None, None

    cols = list(df_eq.columns)
    col_team = next((c for c in cols if "equipo" in c.lower()), None)
    col_dot = next((c for c in cols if "personas" in c.lower()), None)
    if not col_dot:
        col_dot = next((c for c in cols if "dot" in c.lower()), None)
    return col_team, col_dot


def _build_dotacion_map(df_eq: pd.DataFrame) -> dict:
    col_team, col_dot = _get_team_and_dotacion_cols(df_eq)
    if not col_team or not col_dot:
        return {}

    dot = {}
    for _, r in df_eq.iterrows():
        team = str(r.get(col_team, "")).strip()
        if not team or team.lower() == "cupos libres":
            continue
        try:
            val = float(str(r.get(col_dot, "0")).replace(",", "."))
        except Exception:
            continue
        if val > 0:
            dot[team] = val
    return dot


def _equity_score(rows, deficit, dot_map: dict, days_per_week=5):
    """
    Score de equidad:
    - Convierte asignación semanal por equipo a "fracción de cobertura"
      coverage = cupos_asignados / (personas * 5)
    - Queremos que todas las coverages sean lo más parecidas posible.
    """
    assigned = {}
    for r in rows or []:
        eq = str(r.get("equipo", "")).strip()
        if not eq or eq.lower() == "cupos libres":
            continue
        try:
            cup = int(float(str(r.get("cupos", 0)).replace(",", ".")))
        except Exception:
            cup = 0
        assigned[eq] = assigned.get(eq, 0) + cup

    coverages = []
    for eq, people in dot_map.items():
        needed = float(people) * float(days_per_week)
        if needed <= 0:
            continue
        coverages.append(assigned.get(eq, 0) / needed)

    # Si no hay datos para comparar, castiga fuerte
    if not coverages:
        return 9999.0 + (len(deficit) if deficit else 0)

    coverages = np.array(coverages, dtype=float)
    std = float(np.std(coverages))                 # dispersión
    rng = float(np.max(coverages) - np.min(coverages))  # diferencia max-min
    conflicts = float(len(deficit) if deficit else 0)   # penalización por conflictos

    # Pesos: std manda, luego rango, luego conflictos
    return (1.0 * std) + (0.6 * rng) + (0.2 * conflicts)

def generate_balanced_distribution(
    df_eq: pd.DataFrame,
    df_pa: pd.DataFrame,
    df_cap: pd.DataFrame,
    ignore_params: bool,
    num_attempts: int = 80,
    seed: Optional[int] = None
):
    """
    Genera varias distribuciones aleatorias y elige la más EQUITATIVA.
    """
    if seed is None:
        seed = random.randint(1, 10_000_000)

    # dotación por equipo (seguro)
    dot_map = _build_dotacion_map(df_eq)  # usa tus helpers ya definidos arriba

    best_score = float("inf")
    best_rows = None
    best_def = None
    best_meta = None

    for i in range(num_attempts):
        attempt_seed = int(seed) + i * 9973

        # shuffle reproducible
        eq_shuffled = df_eq.sample(frac=1, random_state=attempt_seed).reset_index(drop=True)

        # usa tu motor actual (que ya respeta capacidades)
        rows, deficit = get_distribution_proposal(
            eq_shuffled,
            df_pa,
            df_cap,
            strategy="random",
            ignore_params=ignore_params
        )

        def_clean = filter_minimum_deficits(deficit)

        if ignore_params:
            def_clean = []

        score = _equity_score(rows, def_clean, dot_map)

        if score < best_score:
            best_score = score
            best_rows = rows
            best_def = def_clean
            best_meta = {"score": best_score, "seed": attempt_seed, "attempts": num_attempts}

    return best_rows, best_def, best_meta

def _dot_map_from_equipos(df_eq: pd.DataFrame) -> dict:
    if df_eq is None or df_eq.empty:
        return {}
    col_team = next((c for c in df_eq.columns if "equipo" in c.lower()), None)
    col_dot = None
    # soporta "personas", "dotacion", "dotación", "dot"
    for key in ["personas", "dotacion", "dotación", "dot"]:
        col_dot = next((c for c in df_eq.columns if key in c.lower()), None)
        if col_dot:
            break
    if not col_team or not col_dot:
        return {}

    out = {}
    for _, r in df_eq.iterrows():
        team = str(r.get(col_team, "")).strip()
        if not team or team.lower() == "cupos libres":
            continue
        try:
            val = int(float(str(r.get(col_dot, 0)).replace(",", ".")))
        except Exception:
            continue
        if val > 0:
            out[team] = val
    return out

def _cap_map_from_capacidades(df_cap: pd.DataFrame) -> dict:
    if df_cap is None or df_cap.empty:
        return {}

    col_piso = next((c for c in df_cap.columns if "piso" in c.lower()), None)
    col_dia = next((c for c in df_cap.columns if "dia" in c.lower() or "día" in c.lower()), None)
    col_cap = next((c for c in df_cap.columns if "cap" in c.lower() or "cupo" in c.lower()), None)

    if not col_piso or not col_cap:
        return {}

    out = {}
    for _, r in df_cap.iterrows():
        piso_raw = str(r.get(col_piso, "")).strip()
        if not piso_raw or piso_raw.lower() == "nan":
            continue
        piso = piso_raw if piso_raw.lower().startswith("piso") else f"Piso {piso_raw}"

        dia = str(r.get(col_dia, "")).strip() if col_dia else ""
        if dia.lower() == "nan":
            dia = ""

        try:
            cap = int(float(str(r.get(col_cap, 0)).replace(",", ".")))
        except Exception:
            continue

        if cap <= 0:
            continue

        out[(piso, dia)] = cap

    return out

def _min_daily_for_team(dotacion: int, factor: float = 1.0) -> int:
    if dotacion >= 13: base = 6
    elif dotacion >= 8: base = 4
    elif dotacion >= 5: base = 3
    elif dotacion >= 3: base = 2
    else: base = 0
    return int(round(base * factor))

def _largest_remainder_allocation(weights: dict, total: int) -> dict:
    """
    Reparte 'total' proporcional a weights (enteros) con método Hamilton.
    """
    if total <= 0 or not weights:
        return {k: 0 for k in weights.keys()}

    s = sum(max(0, int(v)) for v in weights.values())
    if s <= 0:
        return {k: 0 for k in weights.keys()}

    quotas = {k: (max(0, int(w)) / s) * total for k, w in weights.items()}
    base = {k: int(q) for k, q in quotas.items()}
    used = sum(base.values())
    remain = total - used

    # ordenar por mayor fracción
    frac = sorted(((k, quotas[k] - base[k]) for k in quotas), key=lambda x: x[1], reverse=True)
    i = 0
    while remain > 0 and i < len(frac):
        k = frac[i][0]
        base[k] += 1
        remain -= 1
        i += 1
        if i >= len(frac) and remain > 0:
            i = 0
    return base

def generate_distribution_math_correct(
    df_eq: pd.DataFrame,
    df_cap: pd.DataFrame,
    cupos_libres_diarios: int = 2,
    min_dotacion_para_garantia: int = 3,
    min_factor: float = 1.0
):
    """
    Genera distribución diaria por piso/día:
    - Mantiene cupos libres fijos.
    - Garantiza mínimos diarios escalados para equipos con dotación >= 3.
    - Resto se reparte proporcional a dotación.
    """
    dot = _dot_map_from_equipos(df_eq)
    cap_map = _cap_map_from_capacidades(df_cap)

    # pisos: desde capacidades si existe, si no desde dot (un piso default)
    pisos = sorted({p for (p, _) in cap_map.keys()} or {"Piso 1"})

    rows = []
    deficits = []

    for piso in pisos:
        for dia in ORDER_DIAS:
            # capacidad para (piso,dia) o fallback (piso,"")
            cap = cap_map.get((piso, dia)) or cap_map.get((piso, "")) or 0
            if cap <= 0:
                # si no hay capacidades, no podemos ser "matemáticamente correctos" por piso/día
                deficits.append({"piso": piso, "dia": dia, "causa": "Sin capacidad definida en hoja Capacidades"})
                continue

            # reservar cupos libres
            libres = min(cupos_libres_diarios, cap)
            cap_rest = cap - libres

            # equipos elegibles
            teams = {k: v for k, v in dot.items() if int(v) >= min_dotacion_para_garantia}

            # mínimos diarios
            mins = {t: _min_daily_for_team(int(v), factor=min_factor) for t, v in teams.items()}
            sum_mins = sum(mins.values())

            if sum_mins > cap_rest:
                # no alcanza ni para mínimos => matemáticamente imposible con esa capacidad
                deficits.append({
                    "piso": piso, "dia": dia,
                    "causa": f"Capacidad insuficiente: mínimos({sum_mins}) > cap_restante({cap_rest})"
                })
                # asignar lo que se pueda proporcionalmente a mins (o 0)
                alloc = _largest_remainder_allocation(mins, cap_rest)
            else:
                # asigna mínimos + resto proporcional a dotación
                alloc = mins.copy()
                extra = cap_rest - sum_mins
                extra_alloc = _largest_remainder_allocation(teams, extra)
                for t, v in extra_alloc.items():
                    alloc[t] = alloc.get(t, 0) + v

            # construir filas
            total_asignado = 0
            for t, c in alloc.items():
                if c <= 0:
                    continue
                total_asignado += c
                rows.append({"piso": piso, "dia": dia, "equipo": t, "cupos": int(c), "pct": 0})

            # cupos libres al final
            rows.append({"piso": piso, "dia": dia, "equipo": "Cupos libres", "cupos": int(libres), "pct": 0})

            # sanity check
            if total_asignado + libres != cap:
                deficits.append({
                    "piso": piso, "dia": dia,
                    "causa": f"Mismatch: asignado({total_asignado + libres}) != capacidad({cap})"
                })

    return rows, deficits

def get_distribution_proposal(
    df_equipos,
    df_parametros,
    df_capacidades,
    strategy="random",
    ignore_params=False,
    variant_seed=None,
    variant_mode="holgura",
):
    """
    Propuesta única.
    - ignore_params=True => NO mínimos, NO días completos. Solo Saint-Laguë + reserva.
    - ignore_params=False => aplica día completo + mínimos + remanente Saint-Laguë.
    """
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

    CUPOS_LIBRES_FIJOS = 2

    rows, deficit_report, audit, score = compute_distribution_from_excel(
        equipos_df=eq_proc,
        parametros_df=pa_proc,
        df_capacidades=df_capacidades,
        cupos_reserva=CUPOS_LIBRES_FIJOS,
        ignore_params=ignore_params,
        variant_seed=variant_seed,
        variant_mode=variant_mode,
    )

    final_deficits = filter_minimum_deficits(deficit_report)

    if ignore_params:
        final_deficits = []

    # opcional: devolver score/audit si te sirve
    return rows, final_deficits, audit, score

def generate_balanced_distribution(
    df_eq: pd.DataFrame,
    df_pa: pd.DataFrame,
    df_cap: pd.DataFrame,
    ignore_params: bool,
    num_attempts: int = 80,
    seed: Optional[int] = None
):
    if seed is None:
        seed = random.randint(1, 10_000_000)

    dot_map = _build_dotacion_map(df_eq)

    best_score = float("inf")
    best_rows, best_def, best_meta = None, None, None

    if ignore_params:
        rng = random.Random(seed)
        factors = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]

        for i in range(num_attempts):
            f = factors[i % len(factors)]
            f = max(0.7, min(1.3, f + rng.uniform(-0.03, 0.03)))

            rows, deficits = generate_distribution_math_correct(
                df_eq, df_cap,
                cupos_libres_diarios=2,
                min_dotacion_para_garantia=3,
                min_factor=f
            )

            def_clean = filter_minimum_deficits(deficits)
            score = _equity_score(rows, def_clean, dot_map)

            hard = sum(1 for d in (deficits or []) if "Capacidad insuficiente" in str(d.get("causa", "")))
            score += hard * 10.0

            if score < best_score:
                best_score = score
                best_rows = rows
                best_def = def_clean
                best_meta = {"score": best_score, "seed": seed, "attempts": num_attempts, "min_factor": f}

        return best_rows, best_def, best_meta

    for i in range(num_attempts):
        attempt_seed = int(seed) + i * 9973
        eq_shuffled = df_eq.sample(frac=1, random_state=attempt_seed).reset_index(drop=True)

        rows, deficit = get_distribution_proposal(
            eq_shuffled, df_pa, df_cap,
            strategy="random",
            ignore_params=False
        )

        def_clean = filter_minimum_deficits(deficit)
        score = _equity_score(rows, def_clean, dot_map)

        if score < best_score:
            best_score = score
            best_rows = rows
            best_def = def_clean
            best_meta = {"score": best_score, "seed": attempt_seed, "attempts": num_attempts}

    return best_rows, best_def, best_meta

def filter_minimum_deficits(deficit_list):
    """Recalcula los déficits únicamente cuando el mínimo no se cumple."""
    filtered = []
    for item in deficit_list or []:
        try:
            minimo = int(float(str(item.get("minimo", 0)).strip()))
            asignado = int(float(str(item.get("asignado", 0)).strip()))
        except (TypeError, ValueError):
            continue
        deficit_val = max(0, minimo - asignado)
        if deficit_val > 0:
            fixed = dict(item)
            fixed["minimo"] = minimo
            fixed["asignado"] = asignado
            fixed["deficit"] = deficit_val
            fixed["causa"] = f"Faltan {deficit_val} puestos (capacidad insuficiente)"
            filtered.append(fixed)
    return filtered

def recompute_pct(rows):
    df = pd.DataFrame(rows)
    if df.empty or not {"piso","dia","equipo","cupos"}.issubset(df.columns):
        return rows

    df["cupos"] = pd.to_numeric(df["cupos"], errors="coerce").fillna(0).astype(int)

    # total por piso/día sin cupos libres
    base = df[df["equipo"].str.lower() != "cupos libres"].groupby(["piso","dia"])["cupos"].sum()
    base = base.rename("total").reset_index()

    df = df.merge(base, on=["piso","dia"], how="left")
    df["total"] = df["total"].fillna(0)

    def calc_pct(r):
        if str(r["equipo"]).lower() == "cupos libres":
            return 0
        if r["total"] <= 0:
            return 0
        return round((r["cupos"] / r["total"]) * 100, 2)

    df["pct"] = df.apply(calc_pct, axis=1)
    return df.to_dict("records")

def infer_team_dotacion_map(df):
    """Intenta inferir la dotación total por equipo usando la data disponible."""
    if df is None or df.empty:
        return {}
    
    cols_lower = {c.lower(): c for c in df.columns}
    col_equipo = None
    for key, col in cols_lower.items():
        if "equipo" in key:
            col_equipo = col
            break
    if not col_equipo:
        return {}
    
    col_dot = next((col for key, col in cols_lower.items() if "dotacion" in key or "dotación" in key), None)
    cupos_col = next((col for key, col in cols_lower.items() if "cupo" in key), None)
    pct_col = None
    for key, col in cols_lower.items():
        if "%distrib" in key or "pct" in key or "porcentaje" in key:
            pct_col = col
            break
    
    dot_map = {}
    
    if col_dot:
        series = df[[col_equipo, col_dot]].dropna()
        for _, row in series.iterrows():
            eq = str(row[col_equipo]).strip()
            if not eq or eq.lower().startswith("cupos libres"):
                continue
            try:
                dot = int(float(row[col_dot]))
            except (TypeError, ValueError):
                continue
            if dot > 0:
                dot_map.setdefault(eq, dot)
    
    if not dot_map and cupos_col and pct_col:
        temp = df[[col_equipo, cupos_col, pct_col]].dropna()
        for _, row in temp.iterrows():
            eq = str(row[col_equipo]).strip()
            if not eq or eq.lower().startswith("cupos libres"):
                continue
            try:
                cupos_val = float(str(row[cupos_col]).replace(",", "."))
                pct_val = float(str(row[pct_col]).replace("%", "").replace(",", "."))
            except (TypeError, ValueError):
                continue
            if pct_val <= 0:
                continue
            dot_est = int(round(cupos_val * 100 / pct_val))
            if dot_est > 0 and eq not in dot_map:
                dot_map[eq] = dot_est
    
    return dot_map

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

# --- GENERADORES DE PDF ---
def create_merged_pdf(piso_sel, conn, global_logo_path):
    # Blindaje: piso_sel puede venir None / NaN / int, etc.
    if piso_sel is None or (isinstance(piso_sel, float) and pd.isna(piso_sel)):
        piso_sel = "Piso 1"
    piso_sel = str(piso_sel)

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
            try:
                pdf.image(str(img_path), x=10, y=10, w=190)
            except:
                pass

    if not found_any:
        return None
    return pdf.output(dest='S').encode('latin-1')

def generate_full_pdf(
    distrib_df: pd.DataFrame,
    semanal_df: pd.DataFrame,
    out_path: str = "reporte.pdf",
    logo_path: Path = Path("static/logo.png"),
    deficit_data=None
):
    """
    PDF = INFORME (no motor de cálculo).
    - Usa los mismos valores que ves en la app (ideal: lo que viene de BD).
    - %Distrib diario:
        * si viene en la data, se usa
        * si no viene o viene vacío, se recalcula SOLO desde los cupos del mismo piso/día
          para que la suma por piso/día sea 100% (incluyendo Cupos libres)
    - Incluye Reporte de Déficit si existe.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)

    # -------------------------
    # Helpers
    # -------------------------
    def _clean_text(x):
        return clean_pdf_text(str(x) if x is not None else "")

    def _find_col(df, candidates):
        if df is None or df.empty:
            return None
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.strip().lower()
            if key in lower_map:
                return lower_map[key]
        # fallback: contains
        for cand in candidates:
            key = cand.strip().lower()
            hit = next((orig for low, orig in lower_map.items() if key in low), None)
            if hit:
                return hit
        return None

    def _to_num(x, default=0.0):
        try:
            s = str(x).replace("%", "").strip().replace(",", ".")
            if s.lower() in ["nan", "none", ""]:
                return default
            return float(s)
        except Exception:
            return default

    def _norm_piso(x):
        s = str(x).strip()
        if s.lower() in ["nan", "none", ""]:
            return "-"
        # si viene "2" => "Piso 2"
        if not s.lower().startswith("piso"):
            s = f"Piso {s}"
        return s

    def _pct_fmt(v):
        try:
            return f"{float(v):.1f}%"
        except Exception:
            return "-"

    # -------------------------
    # 1) Preparar data (copias)
    # -------------------------
    df_print = distrib_df.copy() if distrib_df is not None else pd.DataFrame()

    # columnas esperadas (acepta mayúsc/minúsc)
    col_piso  = _find_col(df_print, ["piso"])
    col_dia   = _find_col(df_print, ["dia", "día"])
    col_equ   = _find_col(df_print, ["equipo"])
    col_cup   = _find_col(df_print, ["cupos", "cupo"])
    col_pct   = _find_col(df_print, ["pct", "%distrib", "% distrib", "%distrib diario", "%distribdiario"])

    # si faltan columnas críticas, igual genera PDF con lo que haya
    if df_print.empty or not all([col_dia, col_equ, col_cup]):
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        if logo_path and Path(logo_path).exists():
            try:
                pdf.image(str(logo_path), x=10, y=8, w=30)
            except Exception:
                pass
        pdf.ln(25)
        pdf.cell(0, 10, _clean_text("Informe de Distribución"), ln=True, align="C")
        pdf.ln(8)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, _clean_text("No hay datos suficientes para generar el informe."))
        return pdf.output(dest="S").encode("latin-1")

    # Normalizar piso
    if col_piso:
        df_print[col_piso] = df_print[col_piso].apply(_norm_piso)
    else:
        df_print["piso_tmp"] = "-"
        col_piso = "piso_tmp"

    # Normalizar día/equipo
    df_print[col_dia] = df_print[col_dia].astype(str).str.strip()
    df_print[col_equ] = df_print[col_equ].astype(str).str.strip()

    # Normalizar cupos a int
    df_print[col_cup] = df_print[col_cup].apply(lambda x: int(round(_to_num(x, 0))))

    # -------------------------
    # 2) %Distrib diario coherente (no > 100)
    # -------------------------
    # Si pct no existe o está todo vacío => recalcular desde cupos por piso/día
    need_recalc = False
    if not col_pct:
        need_recalc = True
        df_print["pct_tmp"] = 0.0
        col_pct = "pct_tmp"
    else:
        # si casi todo viene NaN/0 o strings vacíos, recalcula
        pct_vals = df_print[col_pct].apply(lambda x: _to_num(x, 0.0))
        if (pct_vals.fillna(0.0).abs().sum() == 0.0):
            need_recalc = True
        df_print[col_pct] = pct_vals

    if need_recalc:
        # base: total cupos por (piso,dia) incluyendo Cupos libres
        grp = df_print.groupby([col_piso, col_dia], dropna=False)[col_cup].sum().reset_index()
        grp = grp.rename(columns={col_cup: "_total_pd"})
        df_print = df_print.merge(grp, on=[col_piso, col_dia], how="left")
        df_print[col_pct] = df_print.apply(
            lambda r: (r[col_cup] / r["_total_pd"] * 100.0) if _to_num(r.get("_total_pd", 0), 0) > 0 else 0.0,
            axis=1
        )
        df_print.drop(columns=["_total_pd"], inplace=True, errors="ignore")

    # Clamp final (por seguridad visual)
    df_print[col_pct] = df_print[col_pct].apply(lambda v: max(0.0, min(100.0, float(_to_num(v, 0.0)))))

    # Orden
    try:
        # usa tu helper de ordenamiento si aplica
        tmp = df_print.rename(columns={col_piso: "piso", col_dia: "dia"})
        tmp = apply_sorting_to_df(tmp)
        # revierte nombres a lo que teníamos
        tmp = tmp.rename(columns={"piso": col_piso, "dia": col_dia})
        df_print = tmp
    except Exception:
        pass

    # -------------------------
    # Portada + Tabla diaria
    # -------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    if logo_path and Path(logo_path).exists():
        try:
            pdf.image(str(logo_path), x=10, y=8, w=30)
        except Exception:
            pass
    pdf.ln(25)
    pdf.cell(0, 10, _clean_text("Informe de Distribución"), ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _clean_text("1. Detalle de Distribución Diaria"), ln=True)
    pdf.ln(2)

    # Tabla diaria
    pdf.set_font("Arial", "B", 9)
    widths = [28, 78, 22, 18, 24]  # Piso, Equipo, Día, Cupos, %Distrib
    headers = ["Piso", "Equipo", "Día", "Cupos", "%Distrib Diario"]
    for w, h in zip(widths, headers):
        pdf.cell(w, 6, _clean_text(h), border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 9)

    for _, r in df_print.iterrows():
        piso_val = r.get(col_piso, "-")
        eq_val = r.get(col_equ, "")
        dia_val = r.get(col_dia, "")
        cup_val = r.get(col_cup, 0)
        pct_val = r.get(col_pct, 0.0)

        pdf.cell(widths[0], 6, _clean_text(piso_val)[:12], border=1)
        pdf.cell(widths[1], 6, _clean_text(eq_val)[:45], border=1)
        pdf.cell(widths[2], 6, _clean_text(dia_val)[:12], border=1)
        pdf.cell(widths[3], 6, _clean_text(str(int(cup_val))), border=1, align="R")
        pdf.cell(widths[4], 6, _clean_text(_pct_fmt(pct_val)), border=1, align="R")
        pdf.ln()

        # salto de página si está muy abajo
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.set_font("Arial", "B", 9)
            for w, h in zip(widths, headers):
                pdf.cell(w, 6, _clean_text(h), border=1)
            pdf.ln()
            pdf.set_font("Arial", "", 9)

    # -------------------------
    # 2) Resumen semanal por equipo (desde lo mismo)
    # -------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _clean_text("2. Resumen de Uso Semanal por Equipo"), ln=True)
    pdf.ln(2)

    # Totales semanales desde df_print (excluye Cupos libres para resumen)
    df_week = df_print.copy()
    df_week = df_week[df_week[col_equ].str.lower() != "cupos libres"].copy()

    # Total semanal = suma cupos
    wk = df_week.groupby(col_equ, dropna=False)[col_cup].sum().reset_index()
    wk = wk.rename(columns={col_equ: "Equipo", col_cup: "Total Semanal"})
    wk["Promedio Diario"] = (wk["Total Semanal"] / 5.0).round(2)

    # Dotación opcional desde semanal_df (hoja Equipos o similar)
    dot_map = infer_team_dotacion_map(semanal_df) if semanal_df is not None else {}
    wk["Dotación"] = wk["Equipo"].map(dot_map).fillna(0).astype(int)

    def _uso(row):
        dot = int(row["Dotación"])
        tot = float(row["Total Semanal"])
        return round((tot / (dot * 5.0) * 100.0), 2) if dot > 0 else 0.0

    wk["% Uso Semanal"] = wk.apply(_uso, axis=1)

    wk = wk.sort_values(["Promedio Diario", "Equipo"], ascending=[False, True])

    # tabla resumen
    pdf.set_font("Arial", "B", 9)
    w2 = [88, 32, 32, 28]  # Equipo, Prom, Total, %Uso
    h2 = ["Equipo", "Promedio", "Total Semanal", "% Uso Semanal"]
    for w, h in zip(w2, h2):
        pdf.cell(w, 6, _clean_text(h), border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for _, r in wk.iterrows():
        pdf.cell(w2[0], 6, _clean_text(r["Equipo"])[:45], border=1)
        pdf.cell(w2[1], 6, _clean_text(str(r["Promedio Diario"])), border=1, align="R")
        pdf.cell(w2[2], 6, _clean_text(str(int(r["Total Semanal"]))), border=1, align="R")
        pdf.cell(w2[3], 6, _clean_text(_pct_fmt(r["% Uso Semanal"])), border=1, align="R")
        pdf.ln()

        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.set_font("Arial", "B", 9)
            for w, h in zip(w2, h2):
                pdf.cell(w, 6, _clean_text(h), border=1)
            pdf.ln()
            pdf.set_font("Arial", "", 9)

    # -------------------------
    # Glosario (reducido)
    # -------------------------
    pdf.ln(6)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, _clean_text("Glosario de Métricas y Cálculos:"), ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(
        0, 6,
        _clean_text(
            "1. % Distribución Diario = (Cupos del equipo en el día / Total cupos del piso en ese día) * 100.\n"
            "2. Promedio Diario = (Total Semanal / 5)."
            "3. % Uso Semanal = (Cupos del equipo en la semana / (Dotación * 5)) * 100 (si hay dotación disponible).\n"
            "4. Déficit = Máximo(0, Mínimo requerido - Asignado)."
        )
    )

    # -------------------------
    # 3) Reporte de Déficit (si existe)
    # -------------------------
    deficits_ui = filter_minimum_deficits(deficit_data or [])
    if deficits_ui:
        pdf.add_page()
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 8, _clean_text("Reporte de Déficit de Cupos"), ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # columnas 
        pdf.set_font("Arial", "B", 9)
        wd = [14, 58, 20, 14, 14, 14, 56]  # Piso, Equipo, Día, Dot, Min, Falt, Causa
        hd = ["Piso", "Equipo", "Día", "Dot.", "Min.", "Falt.", "Causa Detallada"]
        for w, h in zip(wd, hd):
            pdf.cell(w, 6, _clean_text(h), border=1, align="C")
        pdf.ln()

        pdf.set_font("Arial", "", 9)
        for item in deficits_ui:
            piso = _norm_piso(item.get("piso", "-")).replace("Piso ", "")
            equipo = str(item.get("equipo", "")).strip()
            dia = str(item.get("dia", item.get("día", ""))).strip()
            dot = int(_to_num(item.get("dotacion", item.get("dot", 0)), 0))
            minimo = int(_to_num(item.get("minimo", 0), 0))
            falt = int(_to_num(item.get("deficit", 0), 0))
            causa = str(item.get("causa", "")).strip()

            # fila (con multiline en causa)
            y0 = pdf.get_y()
            x0 = pdf.get_x()

            pdf.cell(wd[0], 6, _clean_text(piso), border=1)
            pdf.cell(wd[1], 6, _clean_text(equipo)[:28], border=1)
            pdf.cell(wd[2], 6, _clean_text(dia)[:10], border=1, align="C")
            pdf.cell(wd[3], 6, _clean_text(str(dot)), border=1, align="R")
            pdf.cell(wd[4], 6, _clean_text(str(minimo)), border=1, align="R")

            # falt en rojo
            pdf.set_text_color(180, 0, 0)
            pdf.cell(wd[5], 6, _clean_text(str(falt)), border=1, align="R")
            pdf.set_text_color(0, 0, 0)

            # causa multiline: usar multi_cell, pero mantener bordes con truco simple
            x_causa = pdf.get_x()
            y_causa = pdf.get_y()
            pdf.multi_cell(wd[6], 6, _clean_text(causa), border=1)
            y1 = pdf.get_y()

            # volver a la derecha para seguir (multi_cell ya avanzó)
            pdf.set_xy(x0, y1)

            # salto si necesario
            if pdf.get_y() > 265:
                pdf.add_page()
                pdf.set_font("Arial", "B", 9)
                for w, h in zip(wd, hd):
                    pdf.cell(w, 6, _clean_text(h), border=1, align="C")
                pdf.ln()
                pdf.set_font("Arial", "", 9)

    # Footer fecha
    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(80, 80, 80)
    try:
        now = datetime.datetime.now()
        pdf.ln(2)
        pdf.cell(0, 6, _clean_text(f"Informe generado el {now.strftime('%d/%m/%Y %H:%M')} hrs"), ln=True, align="R")
    except Exception:
        pass
    pdf.set_text_color(0, 0, 0)

    return pdf.output(dest="S").encode("latin-1")

def confirm_delete_room_dialog(conn, usuario, fecha_str, sala, inicio):
    st.warning(f"¿Anular reserva de sala?\n\n👤 {usuario} | 📅 {fecha_str}\n🏢 {sala} ({inicio})")
    c1, c2 = st.columns(2)
    if c1.button("🔴 Sí, anular", type="primary", width="stretch", key="yes_s"):
        if delete_room_reservation_from_db(conn, usuario, fecha_str, sala, inicio): st.success("Eliminada"); st.rerun()
    if c2.button("Cancelar", width="stretch", key="no_s"): st.rerun()

# --- UTILS TOKENS ---
def generate_token(): return uuid.uuid4().hex[:8].upper()

def safe_clear_cache(fn):
    """Limpia cache de funciones @st.cache_data si existe; si no, no hace nada."""
    try:
        fn.clear()
    except Exception:
        pass

def clear_compute_cache():
    """Limpia cache del motor seats si está cacheado con @st.cache_data."""
    try:
        compute_distribution_from_excel.clear()
    except Exception:
        pass
        
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

# Obtener settings actualizados en cada carga (permitir que logo y estilos cambien al instante)
settings = get_all_settings(conn)

# Definir variables
site_title = settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios")
global_logo_path = settings.get("logo_path", "static/logo.png")
logo_base64 = settings.get("logo_base64")

logo_source = resolve_logo_source(global_logo_path, logo_base64)

if logo_source is not None:
    c1, c2 = st.columns([1, 5])
    c1.image(logo_source, width=150, use_container_width=False)
    c2.title(site_title)
else:
    st.title(site_title)
    st.caption("💡 No se pudo cargar el logo. Sube uno desde Apariencia o coloca un archivo en static/logo.png.")

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
        t1, t2, t3 = st.tabs(["Estadísticas", "Ver Planos", "Reservas Agendadas"])
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
                            lib = pd.concat([lib, pd.DataFrame([{"piso": piso, "dia": dia, "cupos": 2}])], ignore_index=True)
                        else:
                            # Asegurar mínimo 1, máximo 2
                            idx = lib[mask].index[0] if mask.any() else None
                            if idx is not None:
                                current_val = int(lib.loc[idx, "cupos"]) if pd.notna(lib.loc[idx, "cupos"]) else 1
                                lib.loc[idx, "cupos"] = 2
            
            lib = apply_sorting_to_df(lib)
            
            st.subheader("Distribución completa")
            # MODIFICADO: Fix use_container_width
            st.dataframe(df_view, hide_index=True, width=None, use_container_width=True)
        
        with t3:
            # ==========================
            # A) RESERVAS DE SALAS (con filtro por piso)
            # ==========================
            st.subheader("Reservas de Salas de Reuniones")

            df_salas = get_room_reservations_df(conn)

            # selector de piso para salas
            pisos_salas = []
            if not df_salas.empty and "piso" in df_salas.columns:
                pisos_salas = sort_floors(df_salas["piso"].dropna().unique())
            if not pisos_salas:
                pisos_salas = ["Todos"]

            piso_sala_sel = st.selectbox(
                "Filtrar por piso",
                ["Todos"] + [p for p in pisos_salas if p != "Todos"],
                key="pub_salas_piso_filter"
            )

            if df_salas.empty:
                st.info("No hay reservas de salas registradas.")
            else:
                df_tabla = df_salas[['user_name', 'room_name', 'reservation_date', 'start_time', 'end_time', 'piso']].copy()
                df_tabla.columns = ['Equipo', 'Sala', 'Fecha', 'Hora Inicio', 'Hora Fin', 'Piso']
                df_tabla = df_tabla.sort_values(['Fecha', 'Hora Inicio'])

                if piso_sala_sel != "Todos":
                    df_tabla = df_tabla[df_tabla["Piso"].astype(str) == str(piso_sala_sel)]

                st.dataframe(df_tabla, hide_index=True, width=None, use_container_width=True)

            st.markdown("---")

            # ==========================
            # B) CALENDARIO PISO FLEX (solo aquí)
            # ==========================
            st.subheader("Calendario de reserva para Piso Flex")

            all_res = list_reservations_df(conn)

            fecha_base = datetime.date.today().replace(day=1)
            mes_sel_date = st.date_input(
                "Mes a visualizar",
                value=fecha_base,
                key="cal_reserva_piso_flex_t3"
            )
            mes_sel = mes_sel_date.replace(day=1)

            # pisos disponibles (desde distribución si existe)
            pisos_cal = sort_floors(pisos_disponibles) if pisos_disponibles else ["Piso 1"]

            import calendar
            for piso_cal in pisos_cal:
                piso_label = str(piso_cal).strip()
                if piso_label.lower().startswith("piso piso"):
                    piso_label = piso_label[5:].strip()
                if not piso_label.lower().startswith("piso"):
                    piso_label = f"Piso {piso_label}"

                with st.expander(f"📅 {piso_label}", expanded=False):
                    reservas_piso = []
                    if all_res is not None and not all_res.empty:
                        def _norm_piso_local(x):
                            s = str(x).strip()
                            if s.lower().startswith("piso piso"):
                                s = s[5:].strip()
                            if not s.lower().startswith("piso"):
                                s = f"Piso {s}"
                            return s

                        mask_piso = (
                            all_res["piso"].astype(str).map(_norm_piso_local) == piso_label
                        ) & (all_res["team_area"] == "Cupos libres")

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

                    cal = calendar.monthcalendar(mes_sel.year, mes_sel.month)

                    html_cal = '<div style="margin: 20px 0; overflow-x: auto;">'
                    html_cal += '<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; table-layout: fixed;">'
                    html_cal += '<thead><tr style="background-color: #00A04A; color: white;">'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Lun</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Mar</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Mié</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Jue</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Vie</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Sáb</th>'
                    html_cal += '<th style="padding: 12px 8px; border: 2px solid #006B32; font-size: 13px; font-weight: bold; width: 14.28%;">Dom</th>'
                    html_cal += '</tr></thead><tbody>'

                    for week in cal:
                        html_cal += "<tr style='height: 120px;'>"
                        for day in week:
                            if day == 0:
                                html_cal += '<td style="padding: 0; border: 1px solid #ddd; background-color: #f5f5f5; vertical-align: top;"></td>'
                            else:
                                fecha_dia = datetime.date(mes_sel.year, mes_sel.month, day)
                                reservas_dia = [r for r in reservas_piso if r["fecha"].date() == fecha_dia]

                                if reservas_dia:
                                    equipos_lista = [r["equipo"] for r in reservas_dia]
                                    if len(equipos_lista) > 3:
                                        equipos_mostrar = equipos_lista[:3]
                                        equipos_restantes = len(equipos_lista) - 3
                                        equipos_str = "<br>".join([f"• {eq}" for eq in equipos_mostrar])
                                        equipos_str += f'<br><span style="color: #006B32; font-weight: bold;">+{equipos_restantes} más</span>'
                                    else:
                                        equipos_str = "<br>".join([f"• {eq}" for eq in equipos_lista])

                                    html_cal += f'<td style="padding: 8px 6px; border: 1px solid #ddd; background-color: #e8f5e9; vertical-align: top; min-height: 120px;">'
                                    html_cal += f'<div style="font-size: 14px; font-weight: bold; color: #006B32; margin-bottom: 4px; border-bottom: 1px solid #c8e6c9; padding-bottom: 2px;">{day}</div>'
                                    html_cal += f'<div style="font-size: 10px; color: #2e7d32; line-height: 1.4; word-wrap: break-word; overflow-wrap: break-word;">{equipos_str}</div>'
                                    html_cal += '</td>'
                                else:
                                    html_cal += f'<td style="padding: 8px 6px; border: 1px solid #ddd; vertical-align: top; min-height: 120px;">'
                                    html_cal += f'<div style="font-size: 14px; font-weight: bold; color: #666; margin-bottom: 4px;">{day}</div>'
                                    html_cal += '<div style="font-size: 9px; color: #999; font-style: italic;">Disponible</div>'
                                    html_cal += '</td>'
                        html_cal += "</tr>"

                    html_cal += '</tbody></table></div>'
                    html_cal += '<style>@media (max-width: 768px) { table { font-size: 10px; } td { padding: 4px 2px !important; min-height: 80px !important; } }</style>'

                    st.markdown(html_cal, unsafe_allow_html=True)
        
        with t2:
            st.subheader("Descarga de Planos")
            c1, c2 = st.columns(2)
            
            # Aseguramos que la lista no tenga valores nulos
            pisos_clean = [p for p in pisos_disponibles if p is not None]
            if not pisos_clean: pisos_clean = ["Piso 1"]
            
            p_sel = c1.selectbox("Selecciona Piso", pisos_clean)
            ds = c2.selectbox("Selecciona Día", ["Todos (Lunes a Viernes)"] + ORDER_DIAS)
            
            # Blindaje: Convertir a string antes de reemplazar
            if p_sel:
                pn = str(p_sel).replace("Piso ", "").strip()
            else:
                pn = "1" # Fallback por defecto
                
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
                disponibles = max(0, total_cupos - ocupados)
                
                if disponibles > 0:
                    st.success(f"✅ **Hay cupo: Quedan {disponibles} puestos disponibles**")
                else:
                    st.error(f"🔴 **AGOTADO: Se ocuparon los {total_cupos} puestos del día.**")
                
                st.markdown("### Datos del Solicitante")
                
                # Obtener lista de equipos para seleccionar área
                equipos_disponibles = sorted(df[df["piso"] == pi]["equipo"].unique().tolist())
                equipos_disponibles = [e for e in equipos_disponibles if e != "Cupos libres"]
                
                with st.form("form_puesto"):
                    cf1, cf2 = st.columns(2)
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
                        elif count_monthly_free_spots(conn, em, fe) >= 2:
                            st.error("Has alcanzado el límite de 2 reservas mensuales.")
                        elif disponibles <= 0:
                            st.error("Lo sentimos, el cupo se acaba de agotar.")
                        else:
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
            
            st.markdown("#### Horarios Disponibles")
            
            # Mostrar horarios ocupados (informativo)
            if reservas_hoy:
                horarios_ocupados = ', '.join([f"{r.get('start_time', '')} - {r.get('end_time', '')}" for r in reservas_hoy])
                st.warning(f"⚠️ Horarios ocupados: {horarios_ocupados}")
            
            slots_completos = generate_time_slots("08:00", "20:00", 15)
            if len(slots_completos) < 2:
                st.error("No hay horarios configurados para esta sala.")
            else:
                opciones_inicio = slots_completos[:-1]
                inicio_key = f"inicio_sala_{sl}"
                fin_key = f"fin_sala_{sl}"
                
                idx_inicio = 0
                hora_inicio_sel = st.selectbox("Hora de inicio", opciones_inicio, index=idx_inicio, key=inicio_key)
                
                if hora_inicio_sel not in slots_completos:
                    hora_inicio_sel = opciones_inicio[0]
                pos_inicio = slots_completos.index(hora_inicio_sel)
                opciones_fin = slots_completos[pos_inicio + 1:]
                
                if not opciones_fin:
                    st.warning("Selecciona una hora de inicio anterior a las 20:00 hrs.")
                else:
                    idx_fin = min(4, len(opciones_fin) - 1)  # Default: 1 hora
                    hora_fin_sel = st.selectbox("Hora de término", opciones_fin, index=idx_fin, key=fin_key)
                    
                    inicio_dt = datetime.datetime.strptime(hora_inicio_sel, "%H:%M")
                    fin_dt = datetime.datetime.strptime(hora_fin_sel, "%H:%M")
                    duracion_min = int((fin_dt - inicio_dt).total_seconds() / 60)
                    
                    if duracion_min < 15:
                        st.error("El intervalo debe ser de al menos 15 minutos.")
                    else:
                        conflicto_actual = check_room_conflict(
                            reservas_hoy, str(fe_s), sl, hora_inicio_sel, hora_fin_sel
                        )
                        if conflicto_actual:
                            st.error("❌ Ese intervalo ya está reservado. Elige otro horario.")
                        
                        st.markdown("---")
                        st.markdown(f"### Confirmar Reserva: {hora_inicio_sel} - {hora_fin_sel}")
                        
                        with st.form("form_sala"):
                            st.info(
                                f"**Equipo/Área:** {equipo_seleccionado}\n\n"
                                f"**Sala:** {sl}\n\n"
                                f"**Fecha:** {fe_s}\n\n"
                                f"**Horario:** {hora_inicio_sel} - {hora_fin_sel}"
                            )
                            
                            e_s = st.text_input("Correo Electrónico", key="email_sala")
                            
                            sub_sala = st.form_submit_button("✅ Confirmar Reserva", type="primary", disabled=conflicto_actual)
                        
                        if sub_sala:
                            if not e_s:
                                st.error("Por favor ingresa tu correo electrónico.")
                            elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', e_s):
                                st.error("Por favor ingresa un correo electrónico válido (ejemplo: usuario@ejemplo.com).")
                            elif check_room_conflict(get_room_reservations_df(conn).to_dict("records"), str(fe_s), sl, hora_inicio_sel, hora_fin_sel):
                                st.error("❌ Conflicto: La sala ya está ocupada en ese horario.")
                            else:
                                add_room_reservation(
                                    conn,
                                    equipo_seleccionado,
                                    e_s,
                                    pi_s,
                                    sl,
                                    str(fe_s),
                                    hora_inicio_sel,
                                    hora_fin_sel,
                                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                                )
                                msg = (
                                    f"✅ Sala Confirmada:\n\n"
                                    f"- Equipo/Área: {equipo_seleccionado}\n"
                                    f"- Sala: {sl}\n"
                                    f"- Fecha: {fe_s}\n"
                                    f"- Horario: {hora_inicio_sel} - {hora_fin_sel}"
                                )
                                st.success(msg)
                                
                                if e_s:
                                    try:
                                        email_sent = send_reservation_email(e_s, "Reserva Sala", msg.replace("\n","<br>"))
                                        if email_sent:
                                            st.info("📧 Correo de confirmación enviado")
                                        else:
                                            st.warning("⚠️ No se pudo enviar el correo. Verifica la configuración SMTP.")
                                    except Exception as email_error:
                                        st.warning(f"⚠️ Error al enviar correo: {email_error}")
                                
                                st.rerun()

    # ---------------------------------------------------------
    # OPCIÓN 3: GESTIONAR (ANULAR Y VER TODO)
    # ---------------------------------------------------------
    elif opcion_reserva == "📋 Mis Reservas y Listados":
    
        # --- SECCION 1: BUSCADOR PARA ANULAR ---
        st.subheader("Buscar y Cancelar mis reservas")
        q = st.text_input("Ingresa tu Correo o Nombre para buscar:")
    
        mp = pd.DataFrame()
        ms = pd.DataFrame()
    
        if q:
            dp = list_reservations_df(conn)
            ds = get_room_reservations_df(conn)
    
            def ensure_cols(df):
                if df is None or df.empty:
                    return df
                df = df.copy()
                df.columns = [str(c).strip() for c in df.columns]
    
                rename_map = {}
                if "user_name" not in df.columns:
                    for c in df.columns:
                        if c.lower() in ["nombre", "name", "usuario", "user", "user name", "user_name"]:
                            rename_map[c] = "user_name"
                            break
    
                if "user_email" not in df.columns:
                    for c in df.columns:
                        if c.lower() in ["correo", "email", "mail", "e-mail", "user email", "user_email"]:
                            rename_map[c] = "user_email"
                            break
    
                return df.rename(columns=rename_map)
    
            dp = ensure_cols(dp)
            ds = ensure_cols(ds)
    
            required = {"user_name", "user_email"}
            if (dp is not None and not dp.empty and not required.issubset(set(dp.columns))) or \
            (ds is not None and not ds.empty and not required.issubset(set(ds.columns))):
                st.error(f"Faltan columnas para buscar. Encontré en Puestos: {list(dp.columns)} | en Salas: {list(ds.columns)}")
                st.stop()
    
            ql = q.strip().lower()
    
            if dp is not None and not dp.empty:
                mp = dp[
                    dp["user_name"].fillna("").astype(str).str.lower().str.contains(ql, na=False) |
                    dp["user_email"].fillna("").astype(str).str.lower().str.contains(ql, na=False)
                ]
    
            if ds is not None and not ds.empty:
                ms = ds[
                    ds["user_name"].fillna("").astype(str).str.lower().str.contains(ql, na=False) |
                    ds["user_email"].fillna("").astype(str).str.lower().str.contains(ql, na=False)
                ]
    
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
                                confirm_delete_dialog(conn, r["user_name"], r["reservation_date"], r["team_area"], r["piso"])
    
                if not ms.empty:
                    st.markdown("#### 🏢 Tus Salas")
                    for idx, r in ms.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([5, 1])
                            c1.markdown(f"**{r['reservation_date']}** | {r['room_name']} | {r['start_time']} - {r['end_time']}")
                            if c2.button("Anular", key=f"del_s_{idx}", type="primary"):
                                confirm_delete_room_dialog(conn, r["user_name"], r["reservation_date"], r["room_name"], r["start_time"])
    
        st.markdown("---")
    
        # --- SECCION 2: VER TODO ---
        with st.expander("Ver Listado General de Reservas", expanded=True):
            st.subheader("Reserva de puestos")
            st.dataframe(clean_reservation_df(list_reservations_df(conn)), hide_index=True, use_container_width=True)
    
            st.markdown("<br>", unsafe_allow_html=True)
    
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
    # T1: GENERADOR DE DISTRIBUCIÓN
    # -----------------------------------------------------------
    with t1:
        st.subheader("Generador de Distribución Inteligente")
        st.markdown("Sube el archivo Excel para calcular la distribución de puestos.")
        
        # Carga de archivo
        up = st.file_uploader("Subir archivo Excel (Hojas: 'Equipos', 'Parámetros', 'Capacidades')", type=["xlsx"], key="file_uploader_t1")
        
        ignore_params = st.checkbox(
            "Ignorar reglas de días (Generar Ideal respetando solo capacidad)",
            value=False,
            help="Si marcas esto, se ignoran los días fijos y mínimos, PERO se respeta la capacidad total del piso.",
            key="chk_ignore_params"
        )

        # Botón Inicial
        if st.button("Procesar e Iniciar", type="primary", key="btn_process_init"):
            st.cache_data.clear()
            clear_compute_cache()
            
            if up:
                try:
                    # 1. Leer Excel - Equipos
                    df_eq = pd.read_excel(up, "Equipos", engine='openpyxl')
                    
                    # 2. Leer Parámetros (SIEMPRE, aunque ignoremos reglas)
                    try: df_pa = pd.read_excel(up, "Parámetros", engine='openpyxl')
                    except: df_pa = pd.DataFrame()
                    
                    # 3. Leer Capacidades (CRÍTICO para el límite total)
                    try: df_cap = pd.read_excel(up, "Capacidades", engine='openpyxl')
                    except:
                        df_cap = pd.DataFrame()
                        st.warning("⚠️ No se encontró la hoja 'Capacidades'. Se usará la suma de personas.")

                    # Guardar en sesión
                    st.session_state['excel_equipos'] = df_eq
                    st.session_state['excel_params'] = df_pa
                    st.session_state['excel_caps'] = df_cap
                    st.session_state['ignore_params'] = ignore_params
                    st.session_state['ideal_options'] = None
                    
                    # Calcular
                    if ignore_params:
                        # Ideal = Saint-Laguë + reserva (SIN mínimos / SIN full-day)
                        rows, deficit, audit, score = compute_distribution_from_excel(
                            equipos_df=df_eq,
                            parametros_df=df_pa,
                            df_capacidades=df_cap,
                            cupos_reserva=2,
                            ignore_params=True,
                            variant_seed=0,
                            variant_mode="holgura",
                        )
                        st.session_state['proposal_rows'] = rows
                        st.session_state['proposal_deficit'] = []  # por spec
                        st.session_state['variants'] = None
                        st.session_state['selected_variant_idx'] = None
                        st.toast("✅ Distribución ideal generada.")
                    else:
                        # Con parámetros => variantes (choice 'o' + desempates)
                        variants = compute_distribution_variants(
                            equipos_df=df_eq,
                            parametros_df=df_pa,
                            df_capacidades=df_cap,
                            cupos_reserva=2,
                            ignore_params=False,
                            n_variants=5,
                            variant_seed=random.randint(1, 10_000_000),
                            variant_mode="holgura",  # default
                        )

                        best = variants[0] if variants else None
                        st.session_state["variants"] = variants
                        st.session_state["selected_variant_idx"] = 0

                        st.session_state['proposal_rows'] = best["rows"] if best else []
                        st.session_state['proposal_deficit'] = best["deficit_report"] if best else []
                        st.session_state["last_variant_meta"] = {
                            "seed": best["seed"],
                            "mode": best["mode"],
                            "score": best["score"]["score"],
                            "details": best["score"].get("details", {})
                        } if best else None

                        st.success("✅ Distribución generada (con parámetros + variantes).")
                    
        # -----------------------------------------------------------
        # ZONA DE RESULTADOS
        # -----------------------------------------------------------
        if st.session_state.get("proposal_rows") is not None:

            st.markdown("### Panel de Control")
            c_regen, c_opt, c_save = st.columns([1, 1, 1])

            # Recuperar datos de sesión
            df_eq_s = st.session_state.get("excel_equipos", pd.DataFrame())
            df_pa_s = st.session_state.get("excel_params", pd.DataFrame())
            df_cap_s = st.session_state.get("excel_caps", pd.DataFrame())
            ign_s = st.session_state.get("ignore_params", False)

            # 1) REGENERAR (VARIANTES)
            if c_regen.button("🔄 Regenerar", key="btn_regen_variants"):
                st.cache_data.clear()
                clear_compute_cache()
                with st.spinner("Generando variantes..."):
                    if ign_s:
                        # ignore_params => no hay variantes reales (no hay 'o'/mínimos),
                        rows, deficit, audit, score = compute_distribution_from_excel(
                            equipos_df=df_eq_s,
                            parametros_df=df_pa_s,
                            df_capacidades=df_cap_s,
                            cupos_reserva=2,
                            ignore_params=True,
                            variant_seed=random.randint(1, 10_000_000),
                            variant_mode="holgura",
                        )
                        st.session_state["proposal_rows"] = rows
                        st.session_state["proposal_deficit"] = []
                        st.session_state["variants"] = None
                        st.session_state["selected_variant_idx"] = None
                        st.toast("✅ Regenerado.")
                    else:
                        variants = compute_distribution_variants(
                            equipos_df=df_eq_s,
                            parametros_df=df_pa_s,
                            df_capacidades=df_cap_s,
                            cupos_reserva=2,
                            ignore_params=False,
                            n_variants=8,  # más opciones
                            variant_seed=random.randint(1, 10_000_000),
                            variant_mode=st.session_state.get("variant_mode_ui", "holgura"),
                        )
                        st.session_state["variants"] = variants
                        st.session_state["selected_variant_idx"] = 0

                        best = variants[0] if variants else None
                        st.session_state["proposal_rows"] = best["rows"] if best else []
                        st.session_state["proposal_deficit"] = best["deficit_report"] if best else []
                        st.session_state["last_variant_meta"] = {
                            "seed": best["seed"],
                            "mode": best["mode"],
                            "score": best["score"]["score"],
                            "details": best["score"].get("details", {})
                        } if best else None

                st.rerun()

            # 2) GUARDAR
            if c_save.button("💾 Guardar", type="primary", key="btn_save_v3"):
                try:
                    rows_fixed = recompute_pct(st.session_state["proposal_rows"])
                    st.session_state["proposal_rows"] = rows_fixed

                    clear_distribution(conn)
                    insert_distribution(conn, rows_fixed)

                    if st.session_state.get("proposal_deficit"):
                        st.session_state["deficit_report"] = st.session_state["proposal_deficit"]
                    elif "deficit_report" in st.session_state:
                        del st.session_state["deficit_report"]

                    st.success("¡Guardado correctamente!")
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

            # Mostrar meta de la última equilibrada (si existe)
            meta = st.session_state.get("last_balance_meta")
            if meta:
                st.caption(
                    f"⚖️ Equidad score: {meta['score']:.4f} | intentos: {meta['attempts']} | seed: {meta['seed']}"
                )

            # --- TABLAS ---
            t_view, t_def = st.tabs(["📊 Tabla de Distribución", "🚨 Reporte de Conflictos"])

            with t_view:
                df_preview = pd.DataFrame(st.session_state.get("proposal_rows", []))

                # Mapa Equipo -> Personas desde hoja "Equipos"
                dot_map = _dot_map_from_equipos(df_eq_s)  # ya la tienes definida arriba

                if not df_preview.empty:
                    # normalizar columnas esperadas
                    for c in ["piso", "dia", "equipo", "cupos"]:
                        if c not in df_preview.columns:
                            df_preview[c] = ""

                    # ✅ NO mostrar/usar "Cupos libres" en el generador (tabla admin)
                    df_preview = df_preview[df_preview["equipo"].astype(str).str.strip().str.lower() != "cupos libres"].copy()

                    # ✅ % Uso diario = (Cupos / Personas) * 100
                    def _pct_uso(row):
                        equipo = str(row.get("equipo", "")).strip()
                        cupos = row.get("cupos", 0)
                        try:
                            cupos = int(float(str(cupos).replace(",", ".")))
                        except:
                            cupos = 0
                        personas = dot_map.get(equipo, 0)
                        if personas and personas > 0:
                            return round((cupos / personas) * 100, 2)
                        return 0.0

                    df_preview["% de Uso diario"] = df_preview.apply(_pct_uso, axis=1)

                    # ✅ dejar SOLO estas columnas
                    df_show = df_preview[["piso", "equipo", "dia", "cupos", "% de Uso diario"]].copy()
                    df_show = df_show.rename(columns={
                        "piso": "Piso",
                        "equipo": "Equipo",
                        "dia": "Día",
                        "cupos": "Cupos"
                    })

                    df_show = apply_sorting_to_df(df_show)
                    st.dataframe(df_show, hide_index=True, use_container_width=True)
                else:
                    st.warning("No hay datos.")

            with t_def:
                deficit_data = st.session_state.get("proposal_deficit", [])
                if deficit_data:
                    st.error(f"⚠️ {len(deficit_data)} conflictos detectados.")

                    df_def = pd.DataFrame(deficit_data)

                    cols_hide = [c for c in df_def.columns if str(c).strip().lower() in ("formula", "explicacion", "explicación", "causa")]
                    df_def = df_def.drop(columns=cols_hide, errors="ignore")

                    st.dataframe(df_def, hide_index=True, use_container_width=True)
                else:
                    st.success("✅ Distribución perfecta.")

        else:
            st.info("Sube y procesa un Excel para generar una propuesta.")

    with t2:
    st.subheader("Editor Visual de Zonas (Rectángulos sobre plano)")

    # -------------------------
    # Helpers
    # -------------------------
    def norm_piso(x):
        s = str(x).strip()
        if s.lower().startswith("piso piso"):
            s = s[5:].strip()
        if not s.lower().startswith("piso"):
            s = f"Piso {s}"
        return s

    def sort_floors_like(pisos):
        def keyfn(p):
            p = norm_piso(p)
            num = "".join(ch for ch in p if ch.isdigit())
            return (int(num) if num else 9999, p)
        return sorted([norm_piso(p) for p in pisos], key=keyfn)

    def find_plan_path(p_sel: str):
        p_sel = norm_piso(p_sel)
        p_num = "".join(ch for ch in p_sel if ch.isdigit()) or "1"
        candidates = [
            PLANOS_DIR / f"piso{p_num}.png",
            PLANOS_DIR / f"piso{p_num}.jpg",
            PLANOS_DIR / f"Piso{p_num}.png",
            PLANOS_DIR / f"Piso{p_num}.jpg",
            PLANOS_DIR / f"piso_{p_num}.png",
            PLANOS_DIR / f"piso_{p_num}.jpg",
            PLANOS_DIR / f"piso {p_num}.png",
            PLANOS_DIR / f"piso {p_num}.jpg",
        ]
        return next((p for p in candidates if p.exists()), None)

    def cupos_equipo_dia(df, piso, equipo, dia):
        if df is None or df.empty:
            return 0
        try:
            sub = df[
                (df["piso"].astype(str) == str(piso)) &
                (df["equipo"].astype(str) == str(equipo)) &
                (df["dia"].astype(str) == str(dia))
            ]
            if sub.empty:
                return 0
            return int(pd.to_numeric(sub["cupos"], errors="coerce").fillna(0).sum())
        except Exception:
            return 0

    def zones_for_floor(zonas_all: dict, piso: str):
        zs = (zonas_all.get(piso) or [])
        if not isinstance(zs, list):
            return []
        return zs

    def build_init_objects(existing_zones, scale: float):
        init = []
        for z in existing_zones:
            try:
                if str(z.get("type", "rect")).lower() != "rect":
                    continue
                z_color = z.get("color", "#00A04A") or "#00A04A"
                init.append({
                    "type": "rect",
                    "left": float(z.get("left", 0)) * scale,
                    "top": float(z.get("top", 0)) * scale,
                    "width": float(z.get("width", 0)) * scale,
                    "height": float(z.get("height", 0)) * scale,
                    "fill": z.get("fill", hex_to_rgba(z_color, 0.30)),
                    "stroke": z.get("stroke", z_color),
                    "strokeWidth": z.get("strokeWidth", 2),
                })
            except Exception:
                continue
        return init

    def unscale_rect(o, scale: float):
        def _u(v):
            try:
                return float(v) / float(scale) if scale and scale > 0 else float(v)
            except Exception:
                return 0.0
        return {
            "type": "rect",
            "left": _u(o.get("left", 0)),
            "top": _u(o.get("top", 0)),
            "width": _u(o.get("width", 0)),
            "height": _u(o.get("height", 0)),
            "fill": o.get("fill"),
            "stroke": o.get("stroke"),
            "strokeWidth": o.get("strokeWidth", 2),
        }

    # -------------------------
    # Data base
    # -------------------------
    zonas_all = load_zones() or {}
    df_d = read_distribution_df(conn)

    if df_d is None or df_d.empty or "piso" not in df_d.columns:
        st.warning("⚠️ No hay distribución cargada aún. Sube y procesa un Excel primero.")
        st.stop()

    pisos_list = sort_floors_like(df_d["piso"].dropna().unique().tolist())
    if not pisos_list:
        pisos_list = ["Piso 1"]

    ORDER_DIAS_UI = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

    # -------------------------
    # Session state defaults
    # -------------------------
    if "zones_piso" not in st.session_state:
        st.session_state["zones_piso"] = pisos_list[0]
    if "zones_dia" not in st.session_state:
        st.session_state["zones_dia"] = ORDER_DIAS_UI[0]
    if "zones_equipo" not in st.session_state:
        st.session_state["zones_equipo"] = ""
    if "zones_color" not in st.session_state:
        st.session_state["zones_color"] = "#00A04A"

    # Toolbar config (export)
    if "zones_show_legend" not in st.session_state:
        st.session_state["zones_show_legend"] = True
    if "zones_show_logo" not in st.session_state:
        st.session_state["zones_show_logo"] = True
    if "zones_logo_pos" not in st.session_state:
        st.session_state["zones_logo_pos"] = "Izquierda"
    if "zones_show_title" not in st.session_state:
        st.session_state["zones_show_title"] = True
    if "zones_title_text" not in st.session_state:
        st.session_state["zones_title_text"] = "Distribución de Puestos"
    if "zones_title_align" not in st.session_state:
        st.session_state["zones_title_align"] = "Centro"
    if "zones_title_size" not in st.session_state:
        st.session_state["zones_title_size"] = 18

    # -------------------------
    # Layout
    # -------------------------
    col_left, col_right = st.columns([2.4, 1])

    # =========================
    # Right panel: Piso -> Día -> Equipo -> Cupos
    # =========================
    with col_right:
        st.markdown("### 🎛️ Selección")

        piso_sel = st.selectbox(
            "Piso",
            pisos_list,
            index=pisos_list.index(st.session_state["zones_piso"]) if st.session_state["zones_piso"] in pisos_list else 0,
            key="zones_piso_sel"
        )
        piso_sel = norm_piso(piso_sel)
        st.session_state["zones_piso"] = piso_sel

        dia_sel = st.selectbox(
            "Día (Lunes a Viernes)",
            ORDER_DIAS_UI,
            index=ORDER_DIAS_UI.index(st.session_state["zones_dia"]) if st.session_state["zones_dia"] in ORDER_DIAS_UI else 0,
            key="zones_dia_sel"
        )
        st.session_state["zones_dia"] = dia_sel

        sub_p = df_d[df_d["piso"].astype(str) == str(piso_sel)]
        equipos = sorted([
            str(e) for e in sub_p["equipo"].dropna().unique().tolist()
            if str(e).strip().lower() != "cupos libres"
        ])
        if not equipos:
            equipos = ["(sin equipos cargados)"]

        if st.session_state["zones_equipo"] not in equipos:
            st.session_state["zones_equipo"] = equipos[0]

        equipo_sel = st.selectbox(
            "Equipo",
            equipos,
            index=equipos.index(st.session_state["zones_equipo"]) if st.session_state["zones_equipo"] in equipos else 0,
            key="zones_team_sel"
        )
        st.session_state["zones_equipo"] = equipo_sel

        cupos_val = cupos_equipo_dia(df_d, piso_sel, equipo_sel, dia_sel) if "(sin equipos" not in equipo_sel else 0
        st.caption(f"**Cupos:** {cupos_val}")

        st.markdown("---")
        st.markdown("### 🧾 Leyenda / Export")

        st.session_state["zones_show_legend"] = st.toggle("Incluir leyenda en export (PNG/PDF)", value=st.session_state["zones_show_legend"])
        st.session_state["zones_show_logo"] = st.toggle("Incluir logo", value=st.session_state["zones_show_logo"])
        st.session_state["zones_logo_pos"] = st.selectbox("Posición logo", ["Izquierda", "Centro", "Derecha"], index=["Izquierda","Centro","Derecha"].index(st.session_state["zones_logo_pos"]))

        st.session_state["zones_show_title"] = st.toggle("Incluir título", value=st.session_state["zones_show_title"])
        st.session_state["zones_title_text"] = st.text_input("Texto del título", value=st.session_state["zones_title_text"], disabled=(not st.session_state["zones_show_title"]))
        st.session_state["zones_title_align"] = st.selectbox("Alineación título", ["Izquierda", "Centro", "Derecha"], index=["Izquierda","Centro","Derecha"].index(st.session_state["zones_title_align"]), disabled=(not st.session_state["zones_show_title"]))
        st.session_state["zones_title_size"] = st.slider("Tamaño título", min_value=10, max_value=40, value=int(st.session_state["zones_title_size"]), disabled=(not st.session_state["zones_show_title"]))

    # =========================
    # Left panel: Toolbar + Canvas
    # =========================
    with col_left:
        # --- Load plan image
        plano_path = find_plan_path(piso_sel)
        if not plano_path:
            st.error(f"❌ No se encontró el plano para {piso_sel}")
            st.info("💡 Debe existir en /planos como piso1.png, piso2.png, piso3.png, etc.")
            st.stop()
    st.session_state["last_style_config"] = {
    "show_legend": st.session_state["zones_show_legend"],
    "show_logo": st.session_state["zones_show_logo"],
    "logo_position": st.session_state["zones_logo_pos"].lower(),  # izquierda/centro/derecha
    "show_title": st.session_state["zones_show_title"],
    "title_text": st.session_state["zones_title_text"],
    "title_align": st.session_state["zones_title_align"].lower(),
    "title_font_size": int(st.session_state["zones_title_size"]),
}

        # --- Build existing zones & init objects
        existing = zones_for_floor(zonas_all, piso_sel)

        # Canvas scale for easier editing (2x but with cap)
        pil_img = PILImage.open(plano_path).convert("RGB")
        img_w, img_h = pil_img.size

        target_w = 1400
        scale = min(target_w / img_w, 2.0)
        canvas_w = int(img_w * scale)
        canvas_h = int(img_h * scale)
        bg_img = pil_img.resize((canvas_w, canvas_h))

        init_objects = build_init_objects(existing, scale)

        # --- "Word-like" toolbar
        st.markdown("### 🧰 Herramientas")
        tb1, tb2, tb3, tb4, tb5 = st.columns([1.2, 1.2, 1.4, 1.6, 2.6])

        # Color picker (Word-ish palette + customize hex)
        with tb1:
            st.markdown("**Color**")
            PALETTE = [
                "#000000", "#1C1C1C", "#404040", "#808080", "#C0C0C0", "#FFFFFF",
                "#8B0000", "#FF0000", "#FF4500", "#FFA500", "#FFD700", "#FFFF00",
                "#006400", "#00A04A", "#00FF00", "#00FA9A", "#00FFFF", "#00BFFF",
                "#00008B", "#0000FF", "#1E90FF", "#4169E1", "#8000FF", "#8A2BE2",
                "#FF00FF", "#FF1493", "#DC143C", "#A0522D", "#8B4513", "#2F4F4F",
            ]

            current = st.session_state["zones_color"]
            st.color_picker(" ", value=current, key="zones_color_picker")  # compact native picker
            # keep session in sync
            st.session_state["zones_color"] = st.session_state["zones_color_picker"]

            with st.expander("Paleta (Word)", expanded=False):
                grid = st.columns(6)
                for i, hx in enumerate(PALETTE):
                    with grid[i % 6]:
                        if st.button(" ", key=f"pal_{piso_sel}_{hx}", help=hx):
                            st.session_state["zones_color"] = hx
                            st.session_state["zones_color_picker"] = hx
                            st.rerun()
                        st.markdown(
                            f"""<div style="width:28px;height:28px;border-radius:6px;border:1px solid #999;background:{hx};margin-top:-28px;margin-bottom:10px;"></div>""",
                            unsafe_allow_html=True
                        )

                custom_hex = st.text_input("Personalizar (#RRGGBB)", value=st.session_state["zones_color"], key="zones_hex")
                custom_hex = (custom_hex or "").strip()
                if custom_hex and not custom_hex.startswith("#"):
                    custom_hex = "#" + custom_hex
                if re.match(r"^#[0-9a-fA-F]{6}$", custom_hex):
                    st.session_state["zones_color"] = custom_hex
                    st.session_state["zones_color_picker"] = custom_hex

        # Undo / Redo (limitation note)
        # st_canvas doesn't provide a perfect undo stack API; we implement a practical approach:
        # - "Deshacer": vuelve al último guardado en zonas (desde JSON persistido)
        # - "Rehacer": recarga lo que estaba en el canvas antes del deshacer (RAM)
        if "zones_undo_snapshot" not in st.session_state:
            st.session_state["zones_undo_snapshot"] = None
        if "zones_redo_snapshot" not in st.session_state:
            st.session_state["zones_redo_snapshot"] = None

        with tb2:
            st.markdown("**Edición**")
            if st.button("↶ Deshacer", key="zones_undo_btn"):
                # Save current canvas snapshot as redo (if any)
                st.session_state["zones_redo_snapshot"] = st.session_state.get("zones_canvas_snapshot")
                # Restore from last saved zones state (init_objects)
                st.session_state["zones_undo_snapshot"] = init_objects
                # Force rerun; canvas will re-init from saved zones (since we don't mutate zones here)
                st.rerun()

            if st.button("↷ Rehacer", key="zones_redo_btn"):
                snap = st.session_state.get("zones_redo_snapshot")
                if snap is not None:
                    st.session_state["zones_force_init_objects"] = snap
                    st.rerun()

        # "Guardar zona" (adds the LAST drawn rectangle to stored zones keeping earlier ones)
        # This relies on reading canvas json and appending only new rects since last save.
        with tb3:
            st.markdown("**Zona**")
            save_one = st.button("💾 Guardar zona", key="zones_save_one")

        # "Guardar todo" with confirmation
        with tb4:
            st.markdown("**Guardar**")
            save_all = st.button("✅ Guardar todo", type="primary", key="zones_save_all")

        # Legend preview
        with tb5:
            st.markdown("**Leyenda (previsualización)**")
            # Build legend from zones currently stored (not necessarily current canvas draft)
            # We'll also show selected Equipo/Color/Cupos as "next zone"
            dot = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{st.session_state['zones_color']};margin-right:8px;'></span>"
            st.markdown(
                f"{dot} <b>{equipo_sel}</b> <span style='opacity:.8'>(Cupos: {cupos_val})</span>",
                unsafe_allow_html=True
            )
            st.caption("La leyenda definitiva que se exporta se toma desde las zonas guardadas del piso.")

        # --- Canvas
        fill_rgba = hex_to_rgba(st.session_state["zones_color"], 0.30)
        st_canvas_key = f"canvas_{piso_sel}"

        # If we have a forced init snapshot (redo), prefer it once.
        forced = st.session_state.pop("zones_force_init_objects", None)
        initial_objs = forced if forced is not None else init_objects

        canvas_result = st_canvas(
            fill_color=fill_rgba,
            stroke_width=2,
            stroke_color=st.session_state["zones_color"],
            background_image=bg_img,  # PIL.Image works
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            initial_drawing={"version": "4.4.0", "objects": initial_objs},
            key=st_canvas_key,
        )

        # Keep a snapshot of current canvas objects in RAM (for redo-ish behavior)
        if canvas_result and canvas_result.json_data:
            st.session_state["zones_canvas_snapshot"] = canvas_result.json_data.get("objects", [])

        # -------------------------
        # Save actions
        # -------------------------
        def _persist_from_canvas(objs):
            # converts all rects from canvas into our stored schema and overwrites floor zones
            new_zones = []
            for o in objs or []:
                if o.get("type") != "rect":
                    continue
                rect = unscale_rect(o, scale)

                # attach metadata from current selections where possible
                rect.update({
                    "equipo": equipo_sel,
                    "dia": dia_sel,
                    "color": st.session_state["zones_color"],
                    "fill": rect.get("fill") or hex_to_rgba(st.session_state["zones_color"], 0.30),
                    "stroke": rect.get("stroke") or st.session_state["zones_color"],
                    "strokeWidth": rect.get("strokeWidth", 2),
                })
                new_zones.append(rect)

            zonas_all[piso_sel] = new_zones
            return save_zones(zonas_all), len(new_zones)

        def _append_last_rect_only(objs):
            # appends only the last rect drawn, leaving saved zones intact
            rects = [o for o in (objs or []) if o.get("type") == "rect"]
            if not rects:
                return False, 0

            last = rects[-1]
            rect = unscale_rect(last, scale)
            rect.update({
                "equipo": equipo_sel,
                "dia": dia_sel,
                "color": st.session_state["zones_color"],
                "fill": rect.get("fill") or hex_to_rgba(st.session_state["zones_color"], 0.30),
                "stroke": rect.get("stroke") or st.session_state["zones_color"],
                "strokeWidth": rect.get("strokeWidth", 2),
            })

            current_saved = zones_for_floor(zonas_all, piso_sel)
            current_saved.append(rect)
            zonas_all[piso_sel] = current_saved
            ok = save_zones(zonas_all)
            return ok, (1 if ok else 0)

        # Save one zone (last rect)
        if save_one:
            objs = (canvas_result.json_data.get("objects", []) if canvas_result and canvas_result.json_data else [])
            ok, n = _append_last_rect_only(objs)
            if ok:
                st.success("✅ Zona guardada")
                st.rerun()
            else:
                st.error("❌ No se pudo guardar la zona (¿hay un rectángulo dibujado?)")

        # Save all (confirmation modal)
        if save_all:
            st.session_state["zones_confirm_save_all"] = True

        if st.session_state.get("zones_confirm_save_all"):
            with st.modal("Confirmar guardado"):
                st.write(f"Vas a guardar TODOS los rectángulos del canvas en **{piso_sel}**.")
                c_ok, c_no = st.columns(2)
                if c_ok.button("✅ Confirmar", type="primary", use_container_width=True, key="zones_confirm_yes"):
                    objs = (canvas_result.json_data.get("objects", []) if canvas_result and canvas_result.json_data else [])
                    ok, n = _persist_from_canvas(objs)
                    st.session_state["zones_confirm_save_all"] = False
                    if ok:
                        st.success(f"✅ Guardadas {n} zonas en {piso_sel}")
                        st.rerun()
                    else:
                        st.error("❌ No se pudieron guardar las zonas")
                if c_no.button("Cancelar", use_container_width=True, key="zones_confirm_no"):
                    st.session_state["zones_confirm_save_all"] = False
                    st.rerun()
    
    with t3:
        st.subheader("Descargas")

        # Separar informes de cupos y salas
        st.markdown("### 📊 Informes de Distribución")
        rf = st.selectbox("Formato Reporte", ["Excel (XLSX)", "PDF"], key="report_format")
        if st.button("Generar Reporte de Distribución", key="gen_dist_report"):
            df_raw = read_distribution_df(conn)
            df_raw = apply_sorting_to_df(df_raw)

            # ✅ calcular d_data ANTES de usarlo
            raw_deficits_pdf = st.session_state.get('deficit_report') or st.session_state.get('proposal_deficit') or []
            d_data = filter_minimum_deficits(raw_deficits_pdf)

            if "Excel" in rf:
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    df_raw.to_excel(w, index=False, sheet_name='Distribución')

                st.session_state['rd'] = b.getvalue()
                st.session_state['rn'] = "distribucion.xlsx"
                st.session_state['rm'] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            else:
                df = df_raw.rename(columns={"piso":"Piso","equipo":"Equipo","dia":"Día","cupos":"Cupos","pct":"%Distrib"})
                df_eq_pdf = st.session_state.get("excel_equipos", pd.DataFrame())

                st.session_state['rd'] = generate_full_pdf(
                    df,
                    df_eq_pdf,
                    logo_path=Path(global_logo_path),
                    deficit_data=d_data
                )
                st.session_state['rn'] = "reporte_distribucion.pdf"
                st.session_state['rm'] = "application/pdf"

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
            
            # Verificar si hay datos antes de crear el Excel
            if df_cupos.empty and df_salas.empty:
                st.warning("⚠️ No hay reservas registradas para generar el informe.")
            else:
                b = BytesIO()
                sheets_written = False
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
                        sheets_written = True
                    
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
                        sheets_written = True
                    
                    # Si por alguna razón no se escribió ninguna hoja, crear una vacía
                    if not sheets_written:
                        pd.DataFrame({'Mensaje': ['No hay datos disponibles']}).to_excel(w, index=False, sheet_name='Sin Datos')
                
                if sheets_written:
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
            
            # Asegurar que siempre haya al menos una fila
            if df_planos.empty:
                df_planos = pd.DataFrame({'Piso': ['Sin datos'], 'Día': ['-'], 'Zonas Configuradas': [0], 
                                         'PNG Generado': ['No'], 'PDF Generado': ['No'], 'Estado': ['Sin datos']})
            
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
        st.markdown("### 🗑️ Mantenimiento y Limpieza")
        
        opt = st.radio("Seleccionar categoría:", ["Reservas", "Distribución", "Planos/Zonas", "TODO"], key="delete_option")
        
        if opt == "Reservas":
            st.markdown("#### 📋 Reservas de Puestos Flex")
            df_cupos = list_reservations_df(conn)
            if not df_cupos.empty:
                # Agregar columna de selección
                df_cupos_display = df_cupos.copy()
                df_cupos_display['Seleccionar'] = False
                
                # Mostrar tabla con checkboxes
                st.dataframe(
                    df_cupos_display[['user_name', 'user_email', 'piso', 'reservation_date', 'team_area']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Crear checkboxes para cada fila
                selected_indices_cupos = []
                for idx in df_cupos.index:
                    if st.checkbox(
                        f"Eliminar: {df_cupos.loc[idx, 'user_name']} - {df_cupos.loc[idx, 'reservation_date']} - {df_cupos.loc[idx, 'piso']}",
                        key=f"del_cupo_{idx}"
                    ):
                        selected_indices_cupos.append(idx)
                
                if selected_indices_cupos:
                    if st.button("🗑️ Eliminar Seleccionadas", type="primary", key="delete_selected_cupos"):
                        deleted = 0
                        for idx in selected_indices_cupos:
                            r = df_cupos.loc[idx]
                            if delete_reservation_from_db(conn, r['user_name'], r['reservation_date'], r['team_area']):
                                deleted += 1
                        if deleted > 0:
                            st.success(f"✅ {deleted} reserva(s) eliminada(s)")
                            st.rerun()
                        else:
                            st.error("❌ Error al eliminar reservas")
                
                # Opción de borrar todo
                st.markdown("---")
                if st.button("🗑️ BORRAR TODAS LAS RESERVAS DE PUESTOS", type="primary", key="delete_all_cupos"):
                    ws = get_worksheet(conn, "reservations")
                    if ws:
                        ws.clear()
                        ws.append_row(["user_name", "user_email", "piso", "reservation_date", "team_area", "created_at"])
                        safe_clear_cache(list_reservations_df)
                        st.success("✅ Todas las reservas de puestos eliminadas")
                        st.rerun()
            else:
                st.info("No hay reservas de puestos registradas")
            
            st.markdown("---")
            st.markdown("#### 🏢 Reservas de Salas")
            df_salas = get_room_reservations_df(conn)
            if not df_salas.empty:
                st.dataframe(
                    df_salas[['user_name', 'user_email', 'room_name', 'reservation_date', 'start_time', 'end_time']],
                    hide_index=True,
                    use_container_width=True
                )
                
                selected_indices_salas = []
                for idx in df_salas.index:
                    if st.checkbox(
                        f"Eliminar: {df_salas.loc[idx, 'user_name']} - {df_salas.loc[idx, 'reservation_date']} - {df_salas.loc[idx, 'room_name']} ({df_salas.loc[idx, 'start_time']})",
                        key=f"del_sala_{idx}"
                    ):
                        selected_indices_salas.append(idx)
                
                if selected_indices_salas:
                    if st.button("🗑️ Eliminar Seleccionadas", type="primary", key="delete_selected_salas"):
                        deleted = 0
                        for idx in selected_indices_salas:
                            r = df_salas.loc[idx]
                            if delete_room_reservation_from_db(conn, r['user_name'], r['reservation_date'], r['room_name'], r['start_time']):
                                deleted += 1
                        if deleted > 0:
                            st.success(f"✅ {deleted} reserva(s) de sala eliminada(s)")
                            st.rerun()
                        else:
                            st.error("❌ Error al eliminar reservas")
                
                st.markdown("---")
                if st.button("🗑️ BORRAR TODAS LAS RESERVAS DE SALAS", type="primary", key="delete_all_salas"):
                    ws = get_worksheet(conn, "room_reservations")
                    if ws:
                        ws.clear()
                        ws.append_row(["user_name", "user_email", "piso", "room_name", "reservation_date", "start_time", "end_time", "created_at"])
                        safe_clear_cache(get_room_reservations_df)
                        st.success("✅ Todas las reservas de salas eliminadas")
                        st.rerun()
            else:
                st.info("No hay reservas de salas registradas")
        
        elif opt == "Distribución":
            st.markdown("#### 📊 Distribuciones Guardadas")
            df_dist = read_distribution_df(conn)
            if not df_dist.empty:
                st.dataframe(df_dist, hide_index=True, use_container_width=True)
                
                selected_indices_dist = []
                for idx in df_dist.index:
                    r = df_dist.loc[idx]
                    piso = r.get('piso', r.get('Piso', ''))
                    equipo = r.get('equipo', r.get('Equipo', ''))
                    dia = r.get('dia', r.get('Día', ''))
                    cupos = r.get('cupos', r.get('Cupos', ''))
                    
                    if st.checkbox(
                        f"Eliminar: {piso} - {equipo} - {dia} ({cupos} cupos)",
                        key=f"del_dist_{idx}"
                    ):
                        selected_indices_dist.append(idx)
                
                if selected_indices_dist:
                    if st.button("🗑️ Eliminar Seleccionadas", type="primary", key="delete_selected_dist"):
                        # Obtener los valores reales de las filas seleccionadas
                        rows_to_delete = []
                        for idx in selected_indices_dist:
                            r = df_dist.loc[idx]
                            rows_to_delete.append({
                                'piso': r.get('piso', r.get('Piso', '')),
                                'equipo': r.get('equipo', r.get('Equipo', '')),
                                'dia': r.get('dia', r.get('Día', ''))
                            })
                        
                        deleted = 0
                        for row in rows_to_delete:
                            if delete_distribution_row(conn, row['piso'], row['equipo'], row['dia']):
                                deleted += 1
                        
                        if deleted > 0:
                            st.success(f"✅ {deleted} distribución(es) eliminada(s)")
                            st.rerun()
                        else:
                            st.error("❌ Error al eliminar distribuciones")
                
                st.markdown("---")
                if st.button("🗑️ BORRAR TODA LA DISTRIBUCIÓN", type="primary", key="delete_all_dist"):
                    ws = get_worksheet(conn, "distribution")
                    if ws:
                        ws.clear()
                        ws.append_row(["piso", "equipo", "dia", "cupos", "pct", "created_at"])
                        safe_clear_cache(read_distribution_df)
                        st.success("✅ Toda la distribución eliminada")
                        st.rerun()
            else:
                st.info("No hay distribuciones guardadas")
        
        elif opt == "Planos/Zonas":
            st.markdown("#### 🗺️ Zonas Configuradas por Piso y Día")
            zonas = load_zones()
            
            if zonas:
                for piso in pisos_list:
                    if piso in zonas and zonas[piso]:
                        st.markdown(f"**{piso}**")
                        zonas_piso = zonas[piso]
                        
                        # Agrupar por día si es posible
                        zonas_por_dia = {}
                        for z in zonas_piso:
                            dia = z.get('dia', z.get('Día', 'N/A'))
                            if dia not in zonas_por_dia:
                                zonas_por_dia[dia] = []
                            zonas_por_dia[dia].append(z)
                        
                        for dia, zonas_dia in zonas_por_dia.items():
                            with st.expander(f"{dia} ({len(zonas_dia)} zona(s))"):
                                for i, z in enumerate(zonas_dia):
                                    equipo = z.get('team', z.get('equipo', 'N/A'))
                                    color = z.get('color', '#000000')
                                    c1, c2 = st.columns([4, 1])
                                    c1.markdown(f"**{equipo}** - Color: {color}")
                                    if c2.button("🗑️ Eliminar", key=f"del_zone_{piso}_{dia}_{i}"):
                                        # Encontrar y eliminar la zona
                                        zonas[piso] = [z2 for z2 in zonas[piso] if not (
                                            z2.get('team', z2.get('equipo', '')) == equipo and
                                            z2.get('dia', z2.get('Día', '')) == dia and
                                            z2.get('color', '') == color
                                        )]
                                        if save_zones(zonas):
                                            st.success("✅ Zona eliminada")
                                            st.rerun()
                                        else:
                                            st.error("❌ Error al eliminar zona")
                        
                        # Botón para eliminar todas las zonas del piso
                        if st.button(f"🗑️ Eliminar Todas las Zonas de {piso}", key=f"delete_all_zones_{piso}"):
                            zonas[piso] = []
                            if save_zones(zonas):
                                st.success(f"✅ Todas las zonas de {piso} eliminadas")
                                st.rerun()
                            else:
                                st.error("❌ Error al eliminar zonas")
                        st.markdown("---")
                
                # Botón para eliminar todas las zonas
                if st.button("🗑️ BORRAR TODAS LAS ZONAS", type="primary", key="delete_all_zones"):
                    if save_zones({}):
                        st.success("✅ Todas las zonas eliminadas")
                        st.rerun()
                    else:
                        st.error("❌ Error al eliminar zonas")
            else:
                st.info("No hay zonas configuradas")
        
        elif opt == "TODO":
            st.warning("⚠️ Esta acción eliminará TODAS las reservas, distribuciones y zonas. Esta acción no se puede deshacer.")
            if st.button("🗑️ BORRAR TODO", type="primary", key="delete_everything"):
                msg = perform_granular_delete(conn, "TODO")
                # También eliminar todas las zonas
                if save_zones({}):
                    st.success(f"✅ {msg} Todas las zonas eliminadas.")
                else:
                    st.success(f"✅ {msg} (Error al eliminar zonas)")
                st.rerun()














































