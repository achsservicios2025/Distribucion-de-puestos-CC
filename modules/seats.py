import pandas as pd
import math
import re

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
    all_days = set()
    for s in flexible_options: all_days.update(s)
    return {'fijos': all_days, 'flexibles': options}

def compute_distribution_from_excel(equipos_df, parametros_df, cupos_reserva=2, ignore_params=False):
    rows = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    deficit_report = [] 

    # 1. Normalizar columnas del DataFrame (headers)
    equipos_df.columns = [str(c).strip().lower() for c in equipos_df.columns]
    parametros_df.columns = [str(c).strip().lower() for c in parametros_df.columns]

    # 2. Buscar columnas clave dinámicamente
    col_piso = next((c for c in equipos_df.columns if 'piso' in normalize_text(c)), None)
    col_equipo = next((c for c in equipos_df.columns if 'equipo' in normalize_text(c)), None)
    col_personas = next((c for c in equipos_df.columns if 'personas' in normalize_text(c) or 'total' in normalize_text(c)), None)
    col_minimos = next((c for c in equipos_df.columns if 'minimo' in normalize_text(c) or 'mínimo' in normalize_text(c)), None)
    
    if not (col_piso and col_equipo and col_personas and col_minimos):
        return [], []

    # 3. Procesar Parámetros (o ignorarlos si ignore_params=True)
    capacidad_pisos = {}
    reglas_full_day = {}
    cap_reserva_fija = 2  # Por defecto 2 cupos libres por día (máximo según requerimiento)
    
    if not ignore_params:
        col_param = next((c for c in parametros_df.columns if 'criterio' in normalize_text(c) or 'parametro' in normalize_text(c)), '')
        col_valor = next((c for c in parametros_df.columns if 'valor' in normalize_text(c)), '')
        
        for _, row in parametros_df.iterrows():
            p = str(row.get(col_param, '')).strip().lower()
            v = str(row.get(col_valor, '')).strip()
            if "cupos totales piso" in p:
                match_p = re.search(r'piso\s+(\d+)', p)
                match_c = re.search(r'(\d+)', v)
                if match_p and match_c: capacidad_pisos[match_p.group(1)] = int(match_c.group(1))
            if "cupos libres por piso" in p:
                match_r = re.search(r'(\d+)', v)
                if match_r: cap_reserva_fija = min(int(match_r.group(1)), 2)  # Máximo 2 cupos libres
            if "dia completo" in p or "día completo" in p:
                equipo_nombre = re.split(r'd[ií]a completo\s+', p, flags=re.IGNORECASE)[-1].strip()
                if v: reglas_full_day[normalize_text(equipo_nombre)] = parse_days_from_text(v)
    else:
        # Modo ideal: calcular capacidad automáticamente basada en equipos
        # Estimar capacidad por piso basada en equipos
        for piso_raw in equipos_df[col_piso].dropna().unique():
            piso_str = str(int(piso_raw)) if isinstance(piso_raw, (int, float)) else str(piso_raw)
            df_piso = equipos_df[equipos_df[col_piso] == piso_raw]
            personas_piso = df_piso[col_personas].sum() if col_personas else 0
            # Capacidad = personas + margen para flexibilidad (mínimo 10% más, pero al menos capacidad para 2 cupos libres)
            capacidad_pisos[piso_str] = int(personas_piso * 1.1) + 2

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
            is_flexible = reglas and len(reglas['flexibles']) > 1
            is_fixed_full_day = reglas and not is_flexible

            if is_fixed_full_day:
                for dia in reglas['fijos']:
                    if dia in dias_semana: capacidad_fija_por_dia[dia] += per
            elif is_flexible:
                equipos_flexibles.append({'eq': nm, 'per': per, 'dias_opt': reglas['fijos']})
            else:
                # Estimación base para balanceo (usamos la regla de 2 o el mínimo excel)
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
                
                # Regla de Negocio: Mínimo siempre es al menos 2 (si la dotación lo permite),
                # o lo que diga el Excel si es mayor a 2.
                target_min = max(2, min_excel)
                if target_min > per: target_min = per # No podemos inventar gente
                
                reglas = reglas_full_day.get(normalize_text(nm))
                is_fd_today = False
                if reglas:
                    is_flex = len(reglas['flexibles']) > 1
                    if not is_flex and dia in reglas['fijos']: is_fd_today = True
                    elif is_flex and full_day_asignacion.get(normalize_text(nm)) == dia: is_fd_today = True
                
                # Estructura de equipo
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

            # 2. Prioridad 0: Full Day (Entran completos sí o sí)
            current_cap = cap_total_piso
            for t in fd_teams:
                t['asig'] = t['per']
                current_cap -= t['asig']
            
            # Capacidad restante para equipos normales
            remaining_cap = max(0, current_cap)

            # --- CAMBIO 2: ROTACIÓN PARA JUSTICIA ---
            # Rotamos la lista según el día para que la suerte cambie.
            # IMPORTANTÍSIMO: No ordenamos por tamaño después de esto, 
            # para respetar la rotación y la estrategia elegida en app.py.
            if len(normal_teams) > 0:
                shift = dia_idx % len(normal_teams)
                normal_teams = normal_teams[shift:] + normal_teams[:shift]

            # 3. ALGORITMO DE RONDAS (Equidad Garantizada)
            
            # RONDA 1: Supervivencia (Asegurar 1 cupo a todos)
            # AQUÍ ESTABA EL BUG: Eliminamos el sort() para respetar el orden de rotación/estrategia
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 2: Regla del Mínimo 2 (Asegurar 2 cupos a todos)
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < 2 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 3: Mínimo del Excel (Si el Excel pedía > 2)
            # Aquí priorizamos cumplir el requerimiento formal del Excel
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['min_excel'] and t['asig'] < t['per']:
                    needed = t['min_excel'] - t['asig']
                    give = min(needed, remaining_cap)
                    t['asig'] += give
                    remaining_cap -= give

            # RONDA 4: Crecimiento Proporcional (Llenar con lo que sobra)
            # Ahora priorizamos a los equipos grandes que tienen mayor brecha
            if remaining_cap > 0:
                pool = [t for t in normal_teams if t['asig'] < t['per']]
                if pool:
                    # Calcular cuántos faltan en total
                    total_gap = sum(t['per'] - t['asig'] for t in pool)
                    factor = remaining_cap / total_gap if total_gap > 0 else 0
                    
                    dist_round = 0
                    for t in pool:
                        gap = t['per'] - t['asig']
                        extra = math.floor(gap * factor)
                        t['asig'] += extra
                        dist_round += extra
                    
                    remaining_cap -= dist_round
                    
                    # Saldo final (repartir cupos sueltos por redondeo)
                    # Aquí sí ordenamos por necesidad (quien tiene mayor brecha), pero respetamos rotación para empates
                    pool.sort(key=lambda x: (x['per'] - x['asig']), reverse=True)
                    
                    # --- CAMBIO 3: ROTACIÓN EN EL POOL ---
                    # También rotamos aquí para que el "cupo extra de la suerte" no se lo lleve siempre el mismo
                    if len(pool) > 0:
                         shift_pool = dia_idx % len(pool)
                         pool = pool[shift_pool:] + pool[:shift_pool]

                    for t in pool:
                        if remaining_cap > 0 and t['asig'] < t['per']:
                            t['asig'] += 1
                            remaining_cap -= 1

            # 4. Cálculo de Déficit y Reporte
            # Se considera déficit si no se cumplió la "Regla de 2" o el "Mínimo Excel"
            for t in normal_teams:
                # El objetivo real es el mayor entre 2 y el excel
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

            # 5. Cupos Libres (Solo si sobró después de satisfacer a TODOS al 100%)
            # O si sobró capacidad y ya nadie necesita más puestos
            final_libres = 0
            # Verificamos si alguien quedó con déficit (incluso de crecimiento)
            alguien_falta = any(t['asig'] < t['per'] for t in normal_teams)
            
            if not alguien_falta and remaining_cap > 0:
                 # Máximo 2 cupos libres por día (según requerimiento)
                 final_libres = min(remaining_cap, min(cap_reserva_fija, 2)) if cap_reserva_fija > 0 else min(remaining_cap, 2)

            # Guardar resultados
            all_teams = fd_teams + normal_teams
            for t in all_teams:
                if t['asig'] > 0:
                    pct = round((t['asig'] / t['per']) * 100, 1) if t['per'] > 0 else 0.0
                    rows.append({"piso": f"Piso {piso_str}", "equipo": t['eq'], "dia": dia, "cupos": int(t['asig']), "pct": pct})
            
            if final_libres > 0:
                pct = round((final_libres / cap_total_piso) * 100, 1)
                rows.append({"piso": f"Piso {piso_str}", "equipo": "Cupos libres", "dia": dia, "cupos": int(final_libres), "pct": pct})

    return rows, deficit_report
