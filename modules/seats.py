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

def extract_number(text):
    """Extrae el primer número entero de un texto o número. Ej: 'Piso 1' -> '1'."""
    if pd.isna(text): return None
    text_str = str(text)
    nums = re.findall(r'\d+', text_str)
    if nums:
        return str(int(nums[0])) # Retorna string sin ceros a la izquierda ('01' -> '1')
    return None

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

def compute_distribution_from_excel(equipos_df, parametros_df, df_capacidades, cupos_reserva=2, ignore_params=False):
    rows = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    deficit_report = [] 

    # 1. Normalizar columnas
    equipos_df.columns = [str(c).strip().lower() for c in equipos_df.columns]
    if not parametros_df.empty:
        parametros_df.columns = [str(c).strip().lower() for c in parametros_df.columns]

    # 2. Buscar columnas clave
    col_piso = next((c for c in equipos_df.columns if 'piso' in normalize_text(c)), None)
    col_equipo = next((c for c in equipos_df.columns if 'equipo' in normalize_text(c)), None)
    col_personas = next((c for c in equipos_df.columns if 'personas' in normalize_text(c) or 'total' in normalize_text(c)), None)
    col_minimos = next((c for c in equipos_df.columns if 'minimo' in normalize_text(c) or 'mínimo' in normalize_text(c)), None)
    
    if not (col_piso and col_equipo and col_personas and col_minimos): return [], []

    # 3. PROCESAR CAPACIDADES (NORMALIZACIÓN ROBUSTA)
    capacidad_pisos = {}
    RESERVA_OBLIGATORIA = 2 

    if not df_capacidades.empty:
        # Intentar detectar columnas de la hoja Capacidades (A y B generalmente)
        # Asumimos que la primera columna tiene el nombre del piso y la segunda el total
        for i, row in df_capacidades.iterrows():
            try:
                raw_piso = row.iloc[0] # Columna A (Ej: "Piso 1")
                raw_cap = row.iloc[1]  # Columna B (Ej: 38)
                
                piso_key = extract_number(raw_piso)
                if piso_key and pd.notna(raw_cap):
                    capacidad_pisos[piso_key] = int(raw_cap)
            except:
                continue

    # 4. PROCESAR REGLAS (Solo si no se ignoran)
    reglas_full_day = {}
    if not ignore_params and not parametros_df.empty:
        col_param = next((c for c in parametros_df.columns if 'criterio' in normalize_text(c) or 'parametro' in normalize_text(c)), '')
        col_valor = next((c for c in parametros_df.columns if 'valor' in normalize_text(c)), '')
        if col_param and col_valor:
            for _, row in parametros_df.iterrows():
                p = str(row.get(col_param, '')).strip().lower()
                v = str(row.get(col_valor, '')).strip()
                if "dia completo" in p:
                    nm = re.split(r'd[ií]a completo\s+', p)[-1].strip()
                    if v: reglas_full_day[normalize_text(nm)] = parse_days_from_text(v)

    # --- ALGORITMO ---
    pisos_unicos = equipos_df[col_piso].dropna().unique()

    for piso_raw in pisos_unicos:
        # Normalizar el piso actual del ciclo (Ej: 1.0 -> "1")
        piso_str = extract_number(piso_raw)
        if not piso_str: continue 

        # 1. Definir Techo Físico
        if piso_str in capacidad_pisos:
            cap_total_real = capacidad_pisos[piso_str]
        else:
            # Fallback: Suma de personas + reserva (solo si falla el excel)
            df_temp = equipos_df[equipos_df[col_piso] == piso_raw]
            cap_total_real = int(df_temp[col_personas].sum()) + RESERVA_OBLIGATORIA

        # 2. LA GUILLOTINA: Límite estricto para equipos
        hard_limit = max(0, cap_total_real - RESERVA_OBLIGATORIA)
        
        df_piso = equipos_df[equipos_df[col_piso] == piso_raw].copy()
        
        # Pre-cálculo flexibles
        full_day_asignacion = {} 
        capacidad_fija_por_dia = {d: 0 for d in dias_semana}
        equipos_flexibles = []

        for _, r in df_piso.iterrows():
            nm = str(r[col_equipo]).strip(); per = int(r[col_personas] or 0)
            reglas = reglas_full_day.get(normalize_text(nm))
            is_flex = reglas and len(reglas['flexibles']) > 1
            if reglas and not is_flex:
                for d in reglas['fijos']: 
                    if d in dias_semana: capacidad_fija_por_dia[d] += per
            elif is_flex:
                equipos_flexibles.append({'eq': nm, 'per': per, 'dias_opt': reglas['fijos']})
            else:
                base = max(2, int(r[col_minimos] or 0)) if per >= 2 else per
                for d in dias_semana: capacidad_fija_por_dia[d] += base

        cap_libre_pre = {d: max(0, hard_limit - capacidad_fija_por_dia[d]) for d in dias_semana}
        for item in equipos_flexibles:
            best_day = None; max_l = -999
            for d in item['dias_opt']:
                if d in dias_semana and cap_libre_pre[d] > max_l:
                    max_l = cap_libre_pre[d]; best_day = d
            if best_day:
                full_day_asignacion[normalize_text(item['eq'])] = best_day
                cap_libre_pre[best_day] -= item['per']

        # Loop Diario
        for dia_idx, dia in enumerate(dias_semana):
            teams = []
            for _, r in df_piso.iterrows():
                nm = str(r[col_equipo]).strip(); per = int(r[col_personas] or 0)
                mini = int(r[col_minimos] or 0)
                reglas = reglas_full_day.get(normalize_text(nm))
                is_fd = False
                if reglas:
                    is_flex = len(reglas['flexibles']) > 1
                    if not is_flex and dia in reglas['fijos']: is_fd = True
                    elif is_flex and full_day_asignacion.get(normalize_text(nm)) == dia: is_fd = True
                
                teams.append({
                    'eq': nm, 'per': per, 'min': max(2, mini) if max(2, mini) <= per else per,
                    'asig': 0, 'is_fd': is_fd
                })

            # A. Asignar Full Day
            used = 0
            fd_teams = [t for t in teams if t['is_fd']]
            norm_teams = [t for t in teams if not t['is_fd']]
            
            for t in fd_teams:
                t['asig'] = t['per']
                used += t['asig']

            # B. Round Robin Normales
            if norm_teams:
                shift = dia_idx % len(norm_teams)
                norm_teams = norm_teams[shift:] + norm_teams[:shift]
                keep = True
                while keep:
                    keep = False
                    for t in norm_teams:
                        # Solo asignamos si hay espacio en el hard_limit
                        if used < hard_limit and t['asig'] < t['per']:
                            t['asig'] += 1; used += 1; keep = True
                    
                    # Freno de emergencia si ya llenamos el hard_limit
                    if used >= hard_limit:
                        break

            # C. CORTE ESTRICTO FINAL (Por si acaso Full Day se pasó)
            total_asig = sum(t['asig'] for t in teams)
            exceso = total_asig - hard_limit
            
            if exceso > 0:
                while exceso > 0:
                    candidatos = [t for t in teams if t['asig'] > 0]
                    if not candidatos: break
                    # Prioridad de corte: Quitar al que más tiene asignado
                    candidatos.sort(key=lambda x: x['asig'], reverse=True)
                    candidatos[0]['asig'] -= 1
                    exceso -= 1
            
            # Recalcular total asignado tras el corte
            total_asig = sum(t['asig'] for t in teams)

            # D. Guardar Resultados
            for t in teams:
                if t['asig'] > 0:
                    pct = round(t['asig']/t['per']*100, 1) if t['per'] else 0
                    # Usamos piso_raw para que se vea igual que en el Excel ("Piso 1" o "1")
                    rows.append({"piso": str(piso_raw), "equipo": t['eq'], "dia": dia, "cupos": int(t['asig']), "pct": pct})
            
            # E. Insertar Cupos Libres (Relleno hasta el Total Real)
            remanente = cap_total_real - total_asig
            
            # Seguridad: Si el remanente es menor a 2, algo raro pasó, forzamos 2
            if remanente < RESERVA_OBLIGATORIA: remanente = RESERVA_OBLIGATORIA
            
            pct_lib = round(remanente/cap_total_real*100, 1)
            rows.append({"piso": str(piso_raw), "equipo": "Cupos libres", "dia": dia, "cupos": int(remanente), "pct": pct_lib})

            # F. Reporte Déficit
            for t in teams:
                if t['asig'] < t['per']:
                    cause = "Falta espacio físico (Reserva priorizada)"
                    if t['asig'] < t['min']: cause = "No alcanzó el mínimo requerido"
                    deficit_report.append({
                        "piso": str(piso_raw), "equipo": t['eq'], "dia": dia, 
                        "dotacion": t['per'], "minimo": t['min'], "asignado": t['asig'], 
                        "deficit": t['per'] - t['asig'], "causa": cause
                    })

    return rows, deficit_report
