import pandas as pd
import math
import re

def normalize_text(text):
    """Limpia textos para comparaciones."""
    if pd.isna(text) or text == "": return ""
    text = str(text).strip().lower()
    replacements = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n', '/': ' ', '-': ' '}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return re.sub(r'\s+', ' ', text)

def parse_days_from_text(text):
    """Detecta días fijos y flexibles."""
    if pd.isna(text): return {'fijos': set(), 'flexibles': []}
    mapa = {"lunes":"Lunes", "martes":"Martes", "miercoles":"Miércoles", "jueves":"Jueves", "viernes":"Viernes"}
    options = re.split(r'\s+o\s+|,\s*o\s+', text, flags=re.IGNORECASE)
    all_days = set()
    for option_text in options:
        normalized_option = normalize_text(option_text)
        for key, val in mapa.items():
            if key in normalized_option: all_days.add(val)
    return {'fijos': all_days, 'flexibles': options}

def compute_distribution_from_excel(equipos_df, parametros_df, cupos_reserva=2, ignore_params=False):
    rows = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    deficit_report = [] 

    # 1. Normalizar columnas
    equipos_df.columns = [str(c).strip().lower() for c in equipos_df.columns]
    parametros_df.columns = [str(c).strip().lower() for c in parametros_df.columns]

    # 2. Buscar columnas clave
    col_piso = next((c for c in equipos_df.columns if 'piso' in normalize_text(c)), None)
    col_equipo = next((c for c in equipos_df.columns if 'equipo' in normalize_text(c)), None)
    col_personas = next((c for c in equipos_df.columns if 'personas' in normalize_text(c) or 'total' in normalize_text(c)), None)
    col_minimos = next((c for c in equipos_df.columns if 'minimo' in normalize_text(c) or 'mínimo' in normalize_text(c)), None)
    
    if not (col_piso and col_equipo and col_personas and col_minimos):
        return [], []

    # 3. Procesar Parámetros (LECTURA INTELIGENTE)
    capacidad_pisos = {}
    reglas_full_day = {}
    RESERVA_OBLIGATORIA = 2 

    col_param = next((c for c in parametros_df.columns if 'criterio' in normalize_text(c) or 'parametro' in normalize_text(c)), '')
    col_valor = next((c for c in parametros_df.columns if 'valor' in normalize_text(c)), '')

    # PASO A: Leer SIEMPRE la capacidad física (Cupos Totales), sin importar ignore_params
    if col_param and col_valor:
        for _, row in parametros_df.iterrows():
            p = str(row.get(col_param, '')).strip().lower()
            v = str(row.get(col_valor, '')).strip()
            # Esto se lee SIEMPRE
            if "cupos totales piso" in p:
                match_p = re.search(r'piso\s+(\d+)', p)
                match_c = re.search(r'(\d+)', v)
                if match_p and match_c: 
                    capacidad_pisos[match_p.group(1)] = int(match_c.group(1))

            # PASO B: Leer reglas específicas SOLO SI NO ignoramos parámetros
            if not ignore_params:
                if "dia completo" in p or "día completo" in p:
                    equipo_nombre = re.split(r'd[ií]a completo\s+', p, flags=re.IGNORECASE)[-1].strip()
                    if v: reglas_full_day[normalize_text(equipo_nombre)] = parse_days_from_text(v)

    # 4. Algoritmo de Distribución
    pisos_unicos = equipos_df[col_piso].dropna().unique()

    for piso_raw in pisos_unicos:
        piso_str = str(int(piso_raw)) if isinstance(piso_raw, (int, float)) else str(piso_raw)
        
        # --- DEFINICIÓN DE CAPACIDAD REAL ---
        # Si existe en el Excel (parametros), usamos eso.
        # Si NO existe en el Excel (porque el usuario no lo puso), usamos la suma de personas como fallback.
        if piso_str in capacidad_pisos:
            cap_total_real = capacidad_pisos[piso_str]
        else:
            # Fallback: Solo si no está definido en parametros, sumamos la gente del piso
            df_temp = equipos_df[equipos_df[col_piso] == piso_raw]
            cap_total_real = int(df_temp[col_personas].sum()) if col_personas else 50

        # --- TECHO DURO PARA EQUIPOS ---
        # Si Total Real es 38, el Techo es 36.
        # Los equipos JAMÁS recibirán más de 36.
        hard_cap_equipos = max(0, cap_total_real - RESERVA_OBLIGATORIA)
        
        df_piso = equipos_df[equipos_df[col_piso] == piso_raw].copy()

        # Pre-cálculo de flexibles
        full_day_asignacion = {} 
        capacidad_fija_por_dia = {d: 0 for d in dias_semana}
        equipos_flexibles = []

        for _, r in df_piso.iterrows():
            nm = str(r[col_equipo]).strip()
            per = int(r[col_personas]) if pd.notna(r[col_personas]) else 0
            reglas = reglas_full_day.get(normalize_text(nm))
            is_flexible = reglas and len(reglas['flexibles']) > 1
            is_fixed_full_day = reglas and not is_flexible

            if is_fixed_full_day:
                for dia in reglas['fijos']:
                    if dia in dias_semana: capacidad_fija_por_dia[dia] += per
            elif is_flexible:
                equipos_flexibles.append({'eq': nm, 'per': per, 'dias_opt': reglas['fijos']})
            else:
                min_req = int(r[col_minimos]) if col_minimos and pd.notna(r[col_minimos]) else 0
                base_demand = max(2, min_req) if per >= 2 else per
                for dia in dias_semana: capacidad_fija_por_dia[dia] += base_demand

        capacidad_libre_pre = {d: max(0, hard_cap_equipos - capacidad_fija_por_dia[d]) for d in dias_semana}
        
        for item_flex in equipos_flexibles:
            best_day = None; max_libre = -float('inf')
            for dia_opt in item_flex['dias_opt']:
                if dia_opt in dias_semana:
                    if capacidad_libre_pre[dia_opt] > max_libre:
                        max_libre = capacidad_libre_pre[dia_opt]; best_day = dia_opt
            if best_day:
                full_day_asignacion[normalize_text(item_flex['eq'])] = best_day
                capacidad_libre_pre[best_day] -= item_flex['per']

        # --- BUCLE DIARIO ---
        for dia_idx, dia in enumerate(dias_semana):
            fd_teams = []
            normal_teams = []
            
            for _, r in df_piso.iterrows():
                nm = str(r[col_equipo]).strip()
                per = int(r[col_personas]) if pd.notna(r[col_personas]) else 0
                min_excel = int(r[col_minimos]) if col_minimos and pd.notna(r[col_minimos]) else 0
                
                target_min = max(2, min_excel)
                if target_min > per: target_min = per
                
                reglas = reglas_full_day.get(normalize_text(nm))
                is_fd_today = False
                if reglas:
                    is_flex = len(reglas['flexibles']) > 1
                    if not is_flex and dia in reglas['fijos']: is_fd_today = True
                    elif is_flex and full_day_asignacion.get(normalize_text(nm)) == dia: is_fd_today = True
                
                t = {
                    'eq': nm, 'per': per, 'target_min': target_min, 
                    'asig': 0, 'deficit': 0
                }
                if is_fd_today: fd_teams.append(t)
                else: normal_teams.append(t)

            # CONTROL DE CAPACIDAD DEL DÍA
            used_cap = 0
            
            # 1. Asignar Full Day (Prioridad)
            for t in fd_teams:
                available = hard_cap_equipos - used_cap
                to_assign = min(t['per'], available)
                t['asig'] = to_assign
                used_cap += to_assign

            # 2. Rotación Equitativa
            if len(normal_teams) > 0:
                shift = dia_idx % len(normal_teams)
                normal_teams = normal_teams[shift:] + normal_teams[:shift]

            # 3. Reparto Round Robin
            keep_going = True
            while keep_going and used_cap < hard_cap_equipos:
                keep_going = False
                for t in normal_teams:
                    if used_cap < hard_cap_equipos and t['asig'] < t['per']:
                        t['asig'] += 1
                        used_cap += 1
                        keep_going = True

            # 4. Cálculo de Déficit
            for t in normal_teams + fd_teams:
                goal = t['target_min']
                if t['asig'] < t['per']:
                    t['deficit'] = t['per'] - t['asig']
                    reason = "Falta espacio físico (Reserva priorizada)"
                    if t['asig'] < goal: reason = "No alcanzó el mínimo requerido"
                    
                    deficit_report.append({
                        "piso": f"Piso {piso_str}", "equipo": t['eq'], "dia": dia, 
                        "dotacion": t['per'], "minimo": goal, "asignado": t['asig'],
                        "deficit": t['deficit'], "causa": reason
                    })

            # 5. INSERCIÓN DE CUPOS LIBRES
            # Aquí está la magia: Usamos el Total Real - Lo que gastaron los equipos
            # Si Total=38, Equipos gastaron max 36. Remanente = 38 - 36 = 2.
            # Si Total=38, Equipos gastaron 30 (porque son pocos). Remanente = 38 - 30 = 8.
            remanente_real = cap_total_real - used_cap
            
            # Guardar filas de equipos
            all_teams = fd_teams + normal_teams
            for t in all_teams:
                if t['asig'] > 0:
                    pct = round((t['asig'] / t['per']) * 100, 1) if t['per'] > 0 else 0.0
                    rows.append({"piso": f"Piso {piso_str}", "equipo": t['eq'], "dia": dia, "cupos": int(t['asig']), "pct": pct})
            
            # Guardar fila de Cupos Libres
            if remanente_real > 0:
                pct_libres = round((remanente_real / cap_total_real) * 100, 1)
                rows.append({"piso": f"Piso {piso_str}", "equipo": "Cupos libres", "dia": dia, "cupos": int(remanente_real), "pct": pct_libres})

    return rows, deficit_report
