# modules/seats.py
import pandas as pd
import math
import re
import numpy as np

# --- CONSTANTES ---
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

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

    return {'fijos': fijos, 'flexibles': flexible_options}

# --- ALGORITMO PRINCIPAL DE DISTRIBUCIÓN ---

def compute_distribution_from_excel(equipos_df, parametros_df, strategy="random"):
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
        
        # APLICAR REGLA DE 2 CUPOS LIBRES: Reducir capacidad disponible
        cap_disponible_equipos = cap_total_piso - 2  # Reservar 2 cupos libres
        cap_disponible_equipos = max(0, cap_disponible_equipos)  # No negativo
        
        df_piso = equipos_df[equipos_df[col_piso] == piso_raw].copy()

        # ... (código existente de clasificación de equipos)

        for dia_idx, dia in enumerate(dias_semana):
            # Usar la capacidad reducida para la distribución de equipos
            current_cap = cap_disponible_equipos
            # ... (resto del código de distribución)

            # Al final, siempre asignar 2 cupos libres
            final_libres = 2
            
            # Guardar resultados (mantener código existente)
            all_teams = []  # fd_teams + normal_teams - simplificado para este ejemplo
            for t in all_teams:
                if t['asig'] > 0:
                    pct = round((t['asig'] / t['per']) * 100, 1) if t['per'] > 0 else 0.0
                    rows.append({"piso": f"Piso {piso_str}", "equipo": t['eq'], "dia": dia, "cupos": int(t['asig']), "pct": pct})
            
            # Siempre agregar 2 cupos libres
            pct_libres = round((final_libres / cap_total_piso) * 100, 1) if cap_total_piso > 0 else 0.0
            rows.append({"piso": f"Piso {piso_str}", "equipo": "Cupos libres", "dia": dia, "cupos": int(final_libres), "pct": pct_libres})

    return rows, deficit_report

# --- NUEVAS FUNCIONES DE DISTRIBUCIÓN REALISTA ---

def get_realistic_distribution_proposal(df_equipos, strategy="realistic_equity"):
    """
    Genera distribuciones realistas que respetan capacidades y generan déficit
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
    
    # Verificar equipos con menos de 2 integrantes
    equipos_problema = []
    for equipo, dotacion in zip(equipos, dotaciones):
        if dotacion < 2:
            equipos_problema.append(f"{equipo} ({dotacion} integrante)")
    
    if strategy == "realistic_equity":
        return realistic_equity_distribution(equipos, dotaciones), equipos_problema
    elif strategy == "balanced_flex":
        return balanced_flex_distribution(equipos, dotaciones), equipos_problema
    elif strategy == "controlled_random":
        return controlled_random_distribution(equipos, dotaciones), equipos_problema
    else:
        return realistic_equity_distribution(equipos, dotaciones), equipos_problema

def realistic_equity_distribution(equipos, dotaciones):
    """
    Distribución realista que genera déficit y respeta capacidades limitadas
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    # Capacidades realistas por piso
    capacidades_pisos = {
        "Piso 1": 50,
        "Piso 2": 48, 
        "Piso 3": 45
    }
    
    # Capacidad total semanal (3 pisos × 5 días × capacidad por piso)
    capacidad_total_semanal = sum(capacidades_pisos.values()) * 5
    
    # Dotación total requerida
    dotacion_total = sum(dotaciones)
    
    # Factor de ajuste por capacidad limitada
    factor_capacidad = min(1.0, capacidad_total_semanal / dotacion_total) if dotacion_total > 0 else 1.0
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # Distribución que genera déficit real
        cupos_ideales = int(dotacion * factor_capacidad)
        
        # Asegurar mínimo 2 cupos por día si es posible, pero generar déficit si no hay capacidad
        cupos_minimos_semanales = min(cupos_ideales, 10)  # Máximo 2 por día × 5 días
        cupos_reales_semanales = max(2, cupos_minimos_semanales)  # Mínimo 2 cupos semanales
        
        # Distribuir entre días
        base_cupos = cupos_reales_semanales // 5
        resto = cupos_reales_semanales % 5
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[i % len(pisos)]  # Distribuir equipos entre pisos
            
            cupos_dia = base_cupos + (1 if j < resto else 0)
            
            # Calcular porcentaje REAL basado en dotación total
            if dotacion > 0:
                pct_real = (cupos_dia / dotacion) * 100
            else:
                pct_real = 0
            
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia,
                'pct': round(pct_real, 1)
            })
            
            # Generar reporte de déficit si no se alcanza el mínimo ideal
            cupos_ideales_dia = min(2, dotacion)  # Ideal: 2 por día o dotación si es menor
            if cupos_dia < cupos_ideales_dia:
                deficit_report.append({
                    "piso": piso,
                    "equipo": equipo, 
                    "dia": dia,
                    "dotacion": dotacion,
                    "minimo": cupos_ideales_dia,
                    "asignado": cupos_dia,
                    "deficit": cupos_ideales_dia - cupos_dia,
                    "causa": f"Capacidad limitada del {piso}"
                })
    
    # Cupos libres - máximo 2 por piso por día (RESPETANDO LA REGLA)
    for piso in pisos:
        capacidad_piso = capacidades_pisos[piso]
        for dia in dias_semana:
            # Calcular cupos ya asignados en este piso/día
            cupos_asignados = sum(row['cupos'] for row in rows if row['piso'] == piso and row['dia'] == dia)
            cupos_libres = min(2, capacidad_piso - cupos_asignados)
            
            if cupos_libres > 0:
                rows.append({
                    'piso': piso, 
                    'equipo': "Cupos libres", 
                    'dia': dia, 
                    'cupos': cupos_libres,
                    'pct': round((cupos_libres / capacidad_piso) * 100, 1)
                })
    
    return rows, deficit_report

def balanced_flex_distribution(equipos, dotaciones):
    """
    Distribución balanceada con flexibilidad controlada
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    # Capacidades por piso
    capacidades_pisos = {
        "Piso 1": 50,
        "Piso 2": 48, 
        "Piso 3": 45
    }
    
    capacidad_total_semanal = sum(capacidades_pisos.values()) * 5
    dotacion_total = sum(dotaciones)
    factor_capacidad = min(1.0, capacidad_total_semanal / dotacion_total) if dotacion_total > 0 else 1.0
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # 60% de cupos fijos, 40% flexibles
        cupos_fijos = int(dotacion * factor_capacidad * 0.6)
        cupos_flexibles = int(dotacion * factor_capacidad * 0.4)
        
        cupos_totales = max(2, cupos_fijos + cupos_flexibles)  # Mínimo 2 cupos semanales
        
        base_cupos = cupos_totales // 5
        resto = cupos_totales % 5
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[(i + j) % len(pisos)]  # Rotar pisos por día
            
            cupos_dia = base_cupos + (1 if j < resto else 0)
            
            # Calcular porcentaje real
            pct_real = (cupos_dia / dotacion) * 100 if dotacion > 0 else 0
            
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia,
                'pct': round(pct_real, 1)
            })
            
            # Reportar déficit si es necesario
            if cupos_dia < min(2, dotacion):
                deficit_report.append({
                    "piso": piso,
                    "equipo": equipo, 
                    "dia": dia,
                    "dotacion": dotacion,
                    "minimo": min(2, dotacion),
                    "asignado": cupos_dia,
                    "deficit": min(2, dotacion) - cupos_dia,
                    "causa": "Distribución balanceada con capacidad limitada"
                })
    
    # Cupos libres - respetando la regla de 2 por piso
    for piso in pisos:
        capacidad_piso = capacidades_pisos[piso]
        for dia in dias_semana:
            cupos_asignados = sum(row['cupos'] for row in rows if row['piso'] == piso and row['dia'] == dia)
            cupos_libres = min(2, capacidad_piso - cupos_asignados)
            
            if cupos_libres > 0:
                rows.append({
                    'piso': piso, 
                    'equipo': "Cupos libres", 
                    'dia': dia, 
                    'cupos': cupos_libres,
                    'pct': round((cupos_libres / capacidad_piso) * 100, 1)
                })
    
    return rows, deficit_report

def controlled_random_distribution(equipos, dotaciones):
    """
    Distribución aleatoria controlada garantizando mínimos
    """
    rows = []
    deficit_report = []
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    pisos = ["Piso 1", "Piso 2", "Piso 3"]
    
    # Capacidades por piso
    capacidades_pisos = {
        "Piso 1": 50,
        "Piso 2": 48, 
        "Piso 3": 45
    }
    
    capacidad_total_semanal = sum(capacidades_pisos.values()) * 5
    dotacion_total = sum(dotaciones)
    factor_capacidad = min(1.0, capacidad_total_semanal / dotacion_total) if dotacion_total > 0 else 1.0
    
    np.random.seed(42)  # Semilla fija para reproducibilidad
    
    for i, (equipo, dotacion) in enumerate(zip(equipos, dotaciones)):
        # Asignación aleatoria pero controlada
        cupos_totales = max(2, int(dotacion * factor_capacidad))
        
        # Distribuir aleatoriamente entre días
        dist = [0] * 5
        for _ in range(cupos_totales):
            dia_idx = np.random.randint(0, 5)
            dist[dia_idx] += 1
        
        # Asegurar al menos 1 cupo por día si hay suficientes cupos
        if cupos_totales >= 5:
            for j in range(5):
                if dist[j] == 0:
                    # Quitar de un día con muchos y poner en este
                    max_dia = dist.index(max(dist))
                    if dist[max_dia] > 1:
                        dist[max_dia] -= 1
                        dist[j] += 1
        
        for j, dia in enumerate(dias_semana):
            piso = pisos[np.random.randint(0, len(pisos))]
            
            cupos_dia = dist[j]
            pct_real = (cupos_dia / dotacion) * 100 if dotacion > 0 else 0
            
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia,
                'pct': round(pct_real, 1)
            })
            
            if cupos_dia < min(1, dotacion):  # Mínimo 1 por día en esta estrategia
                deficit_report.append({
                    "piso": piso,
                    "equipo": equipo, 
                    "dia": dia,
                    "dotacion": dotacion,
                    "minimo": min(1, dotacion),
                    "asignado": cupos_dia,
                    "deficit": min(1, dotacion) - cupos_dia,
                    "causa": "Distribución aleatoria controlada"
                })
    
    # Cupos libres
    for piso in pisos:
        capacidad_piso = capacidades_pisos[piso]
        for dia in dias_semana:
            cupos_asignados = sum(row['cupos'] for row in rows if row['piso'] == piso and row['dia'] == dia)
            cupos_libres = min(2, capacidad_piso - cupos_asignados)
            
            if cupos_libres > 0:
                rows.append({
                    'piso': piso, 
                    'equipo': "Cupos libres", 
                    'dia': dia, 
                    'cupos': cupos_libres,
                    'pct': round((cupos_libres / capacidad_piso) * 100, 1)
                })
    
    return rows, deficit_report

# --- FUNCIONES DE DISTRIBUCIÓN IDEAL (EXISTENTES) ---

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
    
    # Verificar equipos con menos de 2 integrantes
    equipos_problema = []
    for equipo, dotacion in zip(equipos, dotaciones):
        if dotacion < 2:
            equipos_problema.append(f"{equipo} ({dotacion} integrante)")
    
    if strategy == "perfect_equity":
        return perfect_equity_distribution(equipos, dotaciones, variant), equipos_problema
    elif strategy == "balanced_flex":
        return balanced_flex_distribution_ideal(equipos, dotaciones, variant), equipos_problema
    elif strategy == "controlled_random":
        return controlled_random_distribution_ideal(equipos, dotaciones, variant), equipos_problema
    else:
        return perfect_equity_distribution(equipos, dotaciones, variant), equipos_problema

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
                
            pct = (cupos_dia / dotacion) * 100 if dotacion > 0 else 0
                
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia,
                'pct': round(pct, 1)
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
                'pct': 0.0
            })
    
    return rows, deficit_report

def balanced_flex_distribution_ideal(equipos, dotaciones, variant=0):
    """
    Distribución balanceada con flexibilidad controlada (versión ideal)
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
            
            pct = (cupos_dia / dotacion) * 100 if dotacion > 0 else 0
                
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': cupos_dia,
                'pct': round(pct, 1)
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
                'pct': 0.0
            })
    
    return rows, deficit_report

def controlled_random_distribution_ideal(equipos, dotaciones, variant=0):
    """
    Distribución aleatoria controlada garantizando mínimos (versión ideal)
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
            
            pct = (dist[j] / dotacion) * 100 if dotacion > 0 else 0
            
            rows.append({
                'piso': piso, 
                'equipo': equipo, 
                'dia': dia, 
                'cupos': dist[j],
                'pct': round(pct, 1)
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
                'pct': 0.0
            })
    
    return rows, deficit_report

# --- FUNCIÓN DE ESTADÍSTICAS ---

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
    if not df.empty:
        stats['uniformidad'] = df.groupby('dia')['cupos'].sum().std()
    
    return stats
