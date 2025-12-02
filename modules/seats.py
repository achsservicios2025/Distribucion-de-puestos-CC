# EN modules/seats.py

# ... (dentro del bucle for piso_raw in pisos_unicos) ...

        # D. Guardar Resultados (Equipos)
        final_asig_sum = 0
        for t in teams:
            if t['asig'] > 0:
                pct = round(t['asig']/t['per']*100, 1) if t['per'] else 0
                
                # CORRECCIÓN AQUÍ: Usar piso_str en lugar de str(piso_raw)
                rows.append({
                    "piso": piso_str,  # <--- ANTES DECÍA: str(piso_raw)
                    "equipo": t['eq'], 
                    "dia": dia, 
                    "cupos": int(t['asig']), 
                    "pct": pct
                })
                final_asig_sum += t['asig']

        # E. Insertar Cupos Libres
        remanente = cap_total_real - final_asig_sum
        if remanente < RESERVA_OBLIGATORIA: 
            remanente = RESERVA_OBLIGATORIA
        
        pct_lib = round(remanente/cap_total_real*100, 1)
        
        # CORRECCIÓN AQUÍ TAMBIÉN
        rows.append({
            "piso": piso_str, # <--- ANTES DECÍA: str(piso_raw)
            "equipo": "Cupos libres", 
            "dia": dia, 
            "cupos": int(remanente), 
            "pct": pct_lib
        })

        # F. Reporte de Déficit
        for t in teams:
            if t['asig'] < t['per']:
                cause = "Falta capacidad física (Prioridad Reserva)"
                if t['asig'] < t['min']: cause = "No alcanzó el mínimo requerido"
                deficit_report.append({
                    "piso": piso_str, # <--- ANTES DECÍA: str(piso_raw)
                    "equipo": t['eq'], "dia": dia, 
                    "dotacion": t['per'], "minimo": t['min'], "asignado": t['asig'], 
                    "deficit": t['per'] - t['asig'], "causa": cause
                })
