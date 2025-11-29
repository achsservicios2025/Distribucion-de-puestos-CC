import pandas as pd
import math
import re
import numpy as np

# --- FUNCIONES AUXILIARES ---

def normalize_text(text):
    """Limpia textos para comparaciones (maneja tildes y normaliza espacios)."""
    if pd.isna(text) or text == "": return ""
    text = str(text).strip().lower()
    replacements = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n', '/': ' ', '-': ' '}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return re.sub(r'\s+', ' ', text)

def parse_days_from_text(text):
    """
    Detecta días fijos y días flexibles (opcionales) en el texto.
    Retorna un diccionario con 'fijos' (set) y 'flexibles' (list of options).
    """
    if pd.isna(text): return {'fijos': set(), 'flexibles': []}
    mapa = {"lunes":"Lunes", "martes":"Martes", "miercoles":"Miércoles", "jueves":"Jueves", "viernes":"Viernes"}
    
    options = re.split(r'\s+o\s+|,\s*o\s+', text, flags=re.IGNORECASE)
    
    flexible_options = []
    
    for option_text in options:
        current_set = set()
        normalized_option = normalize_text(option_text)
        
        for key, val in mapa.items():
            if key in normalized_option: current_set.add(val)
            
        if current_set: flexible_options.append(current_set)
    
    if len(flexible_options) == 1:
        fijos = flexible_options[0]
        flexibles_list = []
    else:
        fijos = set()
        for s in flexible_options: fijos.update(s)
        flexibles_list = options

    return {'fijos': fijos, 'flexibles': flexible_options}

# --- ALGORITMO ORIGINAL (MANTENIDO) ---

def compute_distribution_from_excel(equipos_df, parametros_df, cupos_reserva=2):
    rows = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    deficit_report = [] 

    # 1. Normalizar columnas del DataFrame (headers)
    equipos_df.columns = [str(c).strip().lower() for c in equipos_df.columns]
    parametros_df.columns = [str(c).strip().lower() for c in parametros_df.columns]

    # 2. Buscar columnas clave dinámicamente
    col_piso = next((c for c in equipos_df.columns if 'piso' in normalize_text(c)), None)
    col_equipo = next((c for c in equipos_df.columns if 'equipo' in normalize_text(c)), None)
    col_personas = next((c for c in equipos_df.columns if 'dotacion' in normalize_text(c) or 'personas' in normalize_text(c) or 'total' in normalize_text(c)), None)
    col_minimos = next((c for c in equipos_df.columns if 'minimo' in normalize_text(c) or 'mínimo' in normalize_text(c)), None)
    
    if not (col_piso and col_equipo and col_personas and col_minimos):
        return [], [{"piso": "Error", "equipo": "Error", "dia": "", "dotacion": 0, "minimo": 0, "deficit": 0, "causa": "Faltan columnas DOTACION o MINIMO en el Excel."}]

    # 3. Procesar Parámetros
    col_param = next((c for c in parametros_df.columns if 'criterio' in normalize_text(c) or 'parametro' in normalize_text(c)), '')
    col_valor = next((c for c in parametros_df.columns if 'valor' in normalize_text(c)), '')

    capacidad_pisos = {}
    reglas_full_day = {}
    cap_reserva_fija = 0 
    
    for _, row in parametros_df.iterrows():
        p = str(row.get(col_param, '')).strip().lower()
        v = str(row.get(col_valor, '')).strip()
        if "cupos totales piso" in p:
            match_p = re.search(r'piso\s+(\d+)', p)
            match_c = re.search(r'(\d+)', v)
            if match_p and match_c: capacidad_pisos[match_p.group(1)] = int(match_c.group(1))
        if "cupos libres por piso" in p:
            match_r = re.search(r'(\d+)', v)
            if match_r: cap_reserva_fija = int(match_r.group(1))
        if "dia completo" in p or "día completo" in p:
            equipo_nombre = re.split(r'd[ií]a completo\s+', p, flags=re.IGNORECASE)[-1].strip()
            if v: reglas_full_day[normalize_text(equipo_nombre)] = parse_days_from_text(v)

    # 4. Algoritmo de Distribución
    pisos_unicos = equipos_df[col_piso].dropna().unique()

    for piso_raw in pisos_unicos:
        piso_str = str(int(piso_raw)) if isinstance(piso_raw, (int, float)) else str(piso_raw)
        cap_total_piso = capacidad_pisos.get(piso_str, 50) 
        df_piso = equipos_df[equipos_df[col_piso] == piso_raw].copy()

        # --- Lógica de Flexibles (Pre-asignación de día óptimo) ---
        full_day_asignacion = {} 
        capacidad_fija_por_dia = {d: 0 for d in dias_semana}
        equipos_flexibles = []

        for _, r in df_piso.iterrows():
            nm = str(r[col_equipo]).strip()
            per = int(r[col_personas]) if pd.notna(r[col_personas]) else 0
            reglas = reglas_full_day.get(normalize_text(nm))
            is_flexible = reglas and len(reglas.get('flexibles', [])) > 1 
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

        capacidad_libre_pre = {d: max(0, cap_total_piso - capacidad_fija_por_dia[d]) for d in dias_semana}
        for item_flex in equipos_flexibles:
            best_day = None; max_libre = -float('inf')
            for dia_opt in item_flex['dias_opt']:
                if dia_opt in dias_semana:
                    if capacidad_libre_pre[dia_opt] > max_libre:
                        max_libre = capacidad_libre_pre[dia_opt]; best_day = dia_opt
            if best_day:
                full_day_asignacion[normalize_text(item_flex['eq'])] = best_day
                capacidad_libre_pre[best_day] -= item_flex['per']

        # --- BUCLE DE ASIGNACIÓN DIARIA (Algoritmo por Rondas) ---
        for dia_idx, dia in enumerate(dias_semana):
            fd_teams = []
            normal_teams = []
            
            # 1. Clasificación
            for _, r in df_piso.iterrows():
                nm = str(r[col_equipo]).strip()
                per = int(r[col_personas]) if pd.notna(r[col_personas]) else 0
                min_excel = int(r[col_minimos]) if col_minimos and pd.notna(r[col_minimos]) else 0
                
                target_min = max(2, min_excel)
                if target_min > per: target_min = per
                
                reglas = reglas_full_day.get(normalize_text(nm))
                is_fd_today = False
                if reglas:
                    is_flex = len(reglas.get('flexibles', [])) > 1
                    if not is_flex and dia in reglas['fijos']: is_fd_today = True
                    elif is_flex and full_day_asignacion.get(normalize_text(nm)) == dia: is_fd_today = True
                
                t = {
                    'eq': nm, 
                    'per': per, 
                    'min_excel': min_excel, 
                    'target_min': target_min, 
                    'asig': 0, 
                    'deficit': 0
                }
                
                if is_fd_today: fd_teams.append(t)
                else: normal_teams.append(t)

            # 2. Prioridad 0: Full Day
            current_cap = cap_total_piso
            for t in fd_teams:
                t['asig'] = t['per']
                current_cap -= t['asig']
            
            remaining_cap = max(0, current_cap)

            # --- Rotación Inicial ---
            if len(normal_teams) > 0:
                shift = dia_idx % len(normal_teams)
                normal_teams = normal_teams[shift:] + normal_teams[:shift]

            # 3. ALGORITMO DE RONDAS
            
            # RONDA 1: Supervivencia
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 2: Mínimo 2
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < 2 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 3: Mínimo Excel
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['min_excel'] and t['asig'] < t['per']:
                    needed = t['min_excel'] - t['asig']
                    give = min(needed, remaining_cap)
                    t['asig'] += give
                    remaining_cap -= give

            # RONDA 4: Proporcional
            if remaining_cap > 0:
                pool = [t for t in normal_teams if t['asig'] < t['per']]
                if pool:
                    total_gap = sum(t['per'] - t['asig'] for t in pool)
                    factor = remaining_cap / total_gap if total_gap > 0 else 0
                    
                    dist_round = 0
                    for t in pool:
                        gap = t['per'] - t['asig']
                        extra = min(math.floor(gap * factor), gap)
                        t['asig'] += extra
                        dist_round += extra
                        
                    remaining_cap -= dist_round
                    
                    # Saldo final
                    pool = [t for t in normal_teams if t['asig'] < t['per']]
                    if len(pool) > 0:
                        shift_pool = dia_idx % len(pool)
                        pool = pool[shift_pool:] + pool[:shift_pool]

                        for t in pool:
                            if remaining_cap > 0 and t['asig'] < t['per']:
                                t['asig'] += 1
                                remaining_cap -= 1

            # 4. Déficit
            for t in normal_teams:
                goal = t['target_min']
                if t['asig'] < goal:
                    t['deficit'] = goal - t['asig']
                    deficit_report.append({
                        "piso": f"Piso {piso_str}", 
                        "equipo": t['eq'], 
                        "dia": dia, 
                        "dotacion": t['per'],
                        "minimo": goal,
                        "asignado": t['asig'],
                        "deficit": t['deficit'],
                        "causa": "Capacidad crítica (Piso lleno)"
                    })

            # 5. Cupos Libres
            final_libres = 0
            alguien_falta = any(t['asig'] < t['per'] for t in normal_teams)
            
            if not alguien_falta and remaining_cap > 0:
                final_libres = min(remaining_cap, cap_reserva_fija) if cap_reserva_fija > 0 else remaining_cap

            # Guardar resultados
            all_teams = fd_teams + normal_teams
            for t in all_teams:
                if t['asig'] > 0:
                    pct = round((t['asig'] / t['per']) * 100, 1) if t['per'] > 0 else 0.0
                    rows.append({"piso": f"Piso {piso_str}", "equipo": t['eq'], "dia": dia, "cupos": int(t['asig']), "pct": pct})
            
            if final_libres > 0:
                pct = round((final_libres / cap_total_piso) * 100, 1) if cap_total_piso > 0 else 0.0
                rows.append({"piso": f"Piso {piso_str}", "equipo": "Cupos libres", "dia": dia, "cupos": int(final_libres), "pct": pct})

    return rows, deficit_report

# --- NUEVO ALGORITMO: DISTRIBUCIÓN IDEAL ---

def compute_ideal_distribution(equipos_df, variant=0, pisos_capacity=None):
    """
    Genera una distribución ideal ignorando la hoja de parámetros.
    Reglas:
    1. Mínimo 2 cupos por equipo.
    2. Equidad proporcional.
    3. Máximo 3 cupos libres (Relleno agresivo).
    4. Variación aleatoria controlada por 'variant' (seed).
    """
    if pisos_capacity is None:
        # Default o estimado si no se pasa config
        pisos_capacity = {"Piso 1": 50, "Piso 2": 50, "Piso 3": 50}
    
    # 1. Identificar columnas (tolerante a nombres)
    equipos_df.columns = [str(c).strip().lower() for c in equipos_df.columns]
    
    col_equipo = next((c for c in equipos_df.columns if 'equipo' in normalize_text(c)), equipos_df.columns[0])
    col_dotacion = next((c for c in equipos_df.columns if 'dotacion' in normalize_text(c) or 'total' in normalize_text(c)), None)
    
    if not col_dotacion:
        # Fallback: buscar primera numérica
        nums = equipos_df.select_dtypes(include=[np.number]).columns
        if len(nums) > 0: col_dotacion = nums[0]
        else: return [], [{"causa": "No se encontró columna de dotación"}]

    # 2. Preparar datos
    equipos = []
    for _, row in equipos_df.iterrows():
        dot = int(row[col_dotacion]) if pd.notna(row[col_dotacion]) else 0
        if dot > 0:
            equipos.append({
                "nombre": str(row[col_equipo]).strip(),
                "dotacion": dot,
                "asignado": 0
            })
    
    # Semilla para variación determinista pero distinta por opción
    rng = np.random.default_rng(seed=42 + variant)
    
    rows = []
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    deficit_report = []

    # Lista de pisos disponibles
    pisos_list = list(pisos_capacity.keys())
    
    # PROCESO INDEPENDIENTE POR DÍA
    for dia_idx, dia in enumerate(dias):
        # Reiniciar capacidades diarias
        cap_pisos_dia = pisos_capacity.copy()
        total_cap_dia = sum(cap_pisos_dia.values())
        
        # Mezclar equipos aleatoriamente para este día (afecta prioridad de llenado final)
        equipos_dia = [e.copy() for e in equipos]
        rng.shuffle(equipos_dia) 
        
        # A. ASIGNACIÓN MÍNIMA (Regla: Al menos 2 o la dotación si es < 2)
        for eq in equipos_dia:
            min_req = 2 if eq["dotacion"] >= 2 else eq["dotacion"]
            eq["asignado"] = min_req
            total_cap_dia -= min_req
        
        if total_cap_dia < 0:
            deficit_report.append({"piso": "General", "equipo": "Sistema", "dia": dia, "deficit": abs(total_cap_dia), "causa": "Capacidad insuficiente para mínimos (2)"})
            # Aquí se podría implementar lógica de recorte si fuera necesario, pero la regla dice asegurar 2.

        # B. REPARTO PROPORCIONAL DEL REMANENTE
        # Objetivo: Llenar hasta que queden máx 3 libres
        pendientes = [e for e in equipos_dia if e["asignado"] < e["dotacion"]]
        
        # Mientras sobre espacio y haya demanda
        while total_cap_dia > 3 and pendientes: 
            # Selección ponderada o aleatoria simple del pool
            eq = rng.choice(pendientes)
            if eq["asignado"] < eq["dotacion"]:
                eq["asignado"] += 1
                total_cap_dia -= 1
            else:
                pendientes.remove(eq)
            
            # Chequeo de seguridad para salir si todos están llenos
            if not pendientes and total_cap_dia > 3:
                break

        # C. ASIGNACIÓN A PISOS (Bin Packing)
        # Intentamos agrupar equipos enteros en pisos para no fragmentar demasiado
        mapa_pisos = {p: [] for p in pisos_list}
        
        # Ordenamos equipos por tamaño asignado (Best Fit Decreasing ayuda a empaquetar)
        equipos_dia.sort(key=lambda x: x["asignado"], reverse=True)
        
        for eq in equipos_dia:
            assigned_piso = None
            # Rotar el piso de inicio según día y variante para que no siempre se llene el Piso 1 primero
            start_piso_idx = (dia_idx + variant) % len(pisos_list)
            
            for i in range(len(pisos_list)):
                idx = (start_piso_idx + i) % len(pisos_list)
                piso_name = pisos_list[idx]
                if cap_pisos_dia[piso_name] >= eq["asignado"]:
                    cap_pisos_dia[piso_name] -= eq["asignado"]
                    mapa_pisos[piso_name].append(eq)
                    assigned_piso = piso_name
                    break
            
            if not assigned_piso:
                # Caso Split: El equipo no cabe entero, lo dividimos
                rem = eq["asignado"]
                for p_name in pisos_list:
                    if rem <= 0: break
                    available = cap_pisos_dia[p_name]
                    if available > 0:
                        take = min(available, rem)
                        cap_pisos_dia[p_name] -= take
                        rem -= take
                        # Crear entrada parcial
                        mapa_pisos[p_name].append({"nombre": eq["nombre"], "asignado": take, "dotacion": eq["dotacion"]})

        # D. GENERAR FILAS DE SALIDA
        for p_name, lista_eqs in mapa_pisos.items():
            for item in lista_eqs:
                if item["asignado"] > 0:
                    pct = round((item["asignado"] / item["dotacion"]) * 100, 1)
                    rows.append({
                        "piso": p_name,
                        "equipo": item["nombre"],
                        "dia": dia,
                        "cupos": int(item["asignado"]),
                        "pct": pct,
                        "dotacion_total": item["dotacion"] # Metadata útil para reportes
                    })
            
            # Cupos libres restantes en el piso
            libres = cap_pisos_dia[p_name]
            if libres > 0:
                 rows.append({
                        "piso": p_name,
                        "equipo": "Cupos libres",
                        "dia": dia,
                        "cupos": int(libres),
                        "pct": 0.0,
                        "dotacion_total": 0
                    })

    return rows, deficit_report
