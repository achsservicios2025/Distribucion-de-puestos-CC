# modules/seats.py
import pandas as pd
import math
import re
import numpy as np

# --- FUNCIONES AUXILIARES ---

def normalize_text(text):
    """Limpia textos para comparaciones (maneja tildes y normaliza espacios)."""
    if pd.isna(text) or text == "": 
        return ""
    text = str(text).strip().lower()
    
    # Manejo completo de tildes y caracteres especiales
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'ñ': 'n', 'ç': 'c',
        '/': ' ', '-': ' ', '_': ' ', '.': ' ', ',': ' '
    }
    
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

    return {'fijos': fijos, 'flexibles': flexible_options} # CORREGIDO: Usar flexible_options para la lista si existe

# --------------------------------------------------------------------------------
# NOTA: Todo el código de imagen y texto (PIL) ha sido eliminado para la estabilidad.
# --------------------------------------------------------------------------------

# --- ALGORITMO PRINCIPAL DE DISTRIBUCIÓN ---

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

            # 2. Prioridad 0: Full Day (Entran completos sí o sí)
            current_cap = cap_total_piso
            for t in fd_teams:
                t['asig'] = t['per']
                current_cap -= t['asig']
            
            remaining_cap = max(0, current_cap)

            # --- Rotación Inicial (Aplica solo a normal_teams) ---
            if len(normal_teams) > 0:
                shift = dia_idx % len(normal_teams)
                normal_teams = normal_teams[shift:] + normal_teams[:shift]

            # 3. ALGORITMO DE RONDAS (Equidad Garantizada)
            
            # RONDA 1: Supervivencia (Asegurar 1 cupo a todos que lo necesiten)
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 2: Regla del Mínimo 2 (Asegurar 2 cupos a todos que lo necesiten)
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < 2 and t['asig'] < t['per']:
                    t['asig'] += 1
                    remaining_cap -= 1
            
            # RONDA 3: Mínimo del Excel (Priorizar cumplir el requerimiento formal)
            for t in normal_teams:
                if remaining_cap > 0 and t['asig'] < t['min_excel'] and t['asig'] < t['per']:
                    needed = t['min_excel'] - t['asig']
                    give = min(needed, remaining_cap)
                    t['asig'] += give
                    remaining_cap -= give

            # RONDA 4: Crecimiento Proporcional (Llenar con lo que sobra)
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
                    
                    # Saldo final (repartir cupos sueltos por redondeo)
                    pool = [t for t in normal_teams if t['asig'] < t['per']]
                    if len(pool) > 0:
                        shift_pool = dia_idx % len(pool)
                        pool = pool[shift_pool:] + pool[:shift_pool]

                        for t in pool:
                            if remaining_cap > 0 and t['asig'] < t['per']:
                                t['asig'] += 1
                                remaining_cap -= 1

            # 4. Cálculo de Déficit y Reporte
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

# --- NUEVAS FUNCIONES DE DISTRIBUCIÓN IDEAL ---

def get_ideal_distribution_proposal(df_equipos, strategy="perfect_equity", variant=0):
    """
    Genera distribuciones ideales ignorando parámetros de capacidad
    """
    df_eq_proc = df_equipos.copy()
    
    # Identificar columnas automáticamente
    dotacion_col = None
    for col in df_eq_proc.columns:
        if col.lower() in ['dotacion', 'dotación', 'total', 'empleados']:
            dotacion_col = col
            break
    
    if dotacion_col is None:
        numeric_cols = df_eq_proc.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            dotacion_col = numeric_cols[0]
        else:
            dotacion_col = df_eq_proc.columns[1] if len(df_eq_proc.columns) > 1 else df_eq_proc.columns[0]
    
    equipo_col = None
    for col in df_eq_proc.columns:
        if col.lower() in ['equipo', 'team', 'departamento', 'área']:
            equipo_col = col
            break
    if equipo_col is None:
        equipo_col = df_eq_proc.columns[0]
    
    equipos = df_eq_proc[equipo_col].tolist()
    dotaciones = df_eq_proc[dotacion_col].tolist()
    
    if strategy == "perfect_equity":
        return perfect_equity_distribution(equipos, dotaciones, variant)
    elif strategy == "balanced_flex":
        return balanced_flex_distribution(equipos, dotaciones, variant)
    elif strategy == "controlled_random":
        return controlled_random_distribution(equipos, dotaciones, variant)
    else:
        return perfect_equity_distribution(equipos, dotaciones, variant)

def perfect_equity_distribution(equipos, dotaciones, variant=0):
    """
    Distribución perfectamente equitativa garantizando mínimo 2 cupos por equipo por día
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    # Calcular cupos mínimos garantizados (2 por equipo por día)
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # Garantizar mínimo 2 cupos por día (10 cupos semanales mínimos)
        cupos_minimos_semanales = min(dotacion, 10)  # Máximo 2 por día * 5 días
        base_cupos = cupos_minimos_semanales // 5
        resto = cupos_minimos_semanales % 5
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[(i + variant) % len(pisos)]
            cupos_dia = base_cupos + (1 if j < resto else 0)
            # Asegurar mínimo 2 cupos por equipo por día si la dotación lo permite
            if dotacion >= 2:
                cupos_dia = max(cupos_dia, 2)
            else:
                cupos_dia = dotacion
                
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia, 
                'dotacion_total': dotacion
            })
    
    # Cupos libres - máximo 2 por día
    cupos_libres = 2
    for piso in pisos:
        for dia in dias_semana:
            rows.append({
                'piso': piso, 
                'equipo': "Cupos libres", 
                'dia': dia, 
                'cupos': cupos_libres, 
                'dotacion_total': cupos_libres * 5
            })
    
    return rows, deficit_report

def balanced_flex_distribution(equipos, dotaciones, variant=0):
    """
    Distribución balanceada con flexibilidad controlada
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # 80% de cupos fijos distribuidos equitativamente
        cupos_fijos = int(dotacion * 0.8)
        # Garantizar mínimo 2 cupos por día
        cupos_minimos = min(cupos_fijos, 10)  # 2 por día * 5 días
        base_cupos = cupos_minimos // 5
        resto = cupos_minimos % 5
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[(i + variant) % len(pisos)]
            cupos_dia = base_cupos + (1 if j < resto else 0)
            # Asegurar mínimo 2
            cupos_dia = max(cupos_dia, 2) if dotacion >= 2 else dotacion
                
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia, 
                'dotacion_total': dotacion
            })
    
    # Cupos libres - máximo 2 por día
    cupos_libres = 2
    for piso in pisos:
        for dia in dias_semana:
            rows.append({
                'piso': piso, 
                'equipo': "Cupos libres", 
                'dia': dia, 
                'cupos': cupos_libres, 
                'dotacion_total': cupos_libres * 5
            })
    
    return rows, deficit_report

def controlled_random_distribution(equipos, dotaciones, variant=0):
    """
    Distribución aleatoria controlada garantizando mínimos
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    np.random.seed(variant * 1000)  # Diferente semilla para cada variante
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # Primero asignar mínimo 2 cupos por día
        cupos_minimos = min(dotacion, 10)  # 2 por día * 5 días máximo
        cupos_restantes = dotacion - cupos_minimos
        
        # Distribuir mínimos
        dist = [2] * 5  # Mínimo 2 por día
        
        # Distribuir el resto aleatoriamente pero balanceado
        if cupos_restantes > 0:
            for _ in range(cupos_restantes):
                dia_idx = np.random.randint(0, 5)
                dist[dia_idx] += 1
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[np.random.randint(0, len(pisos))]
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': dist[j], 
                'dotacion_total': dotacion
            })
    
    # Cupos libres - máximo 2 por día
    cupos_libres = 2
    for piso in pisos:
        for dia in dias_semana:
            rows.append({
                'piso': piso, 
                'equipo': "Cupos libres", 
                'dia': dia, 
                'cupos': cupos_libres, 
                'dotacion_total': cupos_libres * 5
            })
    
    return rows, deficit_report

def calculate_distribution_stats(rows, df_equipos):
    """
    Calcula métricas de calidad de la distribución
    """
    df = pd.DataFrame(rows)
    dotacion_map = {}
    
    # Mapear dotaciones
    equipo_col = None
    dotacion_col = None
    for col in df_equipos.columns:
        if col.lower() in ['equipo', 'team', 'departamento']:
            equipo_col = col
        elif col.lower() in ['dotacion', 'dotación', 'total']:
            dotacion_col = col
    
    if equipo_col and dotacion_col:
        for _, row in df_equipos.iterrows():
            dotacion_map[row[equipo_col]] = row[dotacion_col]
            
    stats = {
        'total_cupos_asignados': df['cupos'].sum(),
        'cupos_libres': df[df['equipo'] == 'Cupos libres']['cupos'].sum(),
        'equipos_con_deficit': 0,
        'distribucion_promedio': 0,
        'uniformidad': 0
    }
    
    # Calcular déficits
    for eq in df['equipo'].unique():
        if eq == 'Cupos libres':
            continue
        cupos_totales = df[df['equipo'] == eq]['cupos'].sum()
        dotacion = dotacion_map.get(eq, cupos_totales)
        if cupos_totales < dotacion:
            stats['equipos_con_deficit'] += 1
    
    # Calcular uniformidad (desviación estándar de cupos por día)
    stats['uniformidad'] = df.groupby('dia')['cupos'].sum().std()
    
    return stats
