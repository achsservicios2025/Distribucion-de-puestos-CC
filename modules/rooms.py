from datetime import datetime, timedelta

def generate_time_slots(start_str, end_str, interval_minutes):
    """Genera lista de horas ej: ['08:00', '08:15', ...]"""
    try:
        start = datetime.strptime(start_str, "%H:%M")
        end = datetime.strptime(end_str, "%H:%M")
        slots = []
        while start <= end:
            slots.append(start.strftime("%H:%M"))
            start += timedelta(minutes=interval_minutes)
        return slots
    except:
        # Fallback manual si falla la importación interna
        return ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00"]

def check_room_conflict(reservations, check_date, check_room, check_start, check_end):
    """
    Verifica si hay traslape de horario para una sala.
    Maneja nombres de columnas de BD (reservation_date) o diccionario (fecha).
    """
    try:
        new_start = datetime.strptime(check_start, "%H:%M")
        new_end = datetime.strptime(check_end, "%H:%M")
    except:
        return False # Error en formato de hora input

    for r in reservations:
        # Adaptador inteligente de claves (para evitar KeyErrors)
        # Intenta leer 'reservation_date' (BD), si no existe, lee 'fecha' (Legacy)
        r_date = r.get("reservation_date") or r.get("fecha")
        r_room = r.get("room_name") or r.get("sala")
        r_start = r.get("start_time") or r.get("inicio")
        r_end = r.get("end_time") or r.get("fin")

        # Si faltan datos en el registro, saltarlo
        if not (r_date and r_room and r_start and r_end):
            continue

        if str(r_date) != str(check_date) or r_room != check_room:
            continue
            
        # Verificar traslape
        try:
            curr_start = datetime.strptime(r_start, "%H:%M")
            curr_end = datetime.strptime(r_end, "%H:%M")
            
            # Lógica de colisión: (InicioA < FinB) y (FinA > InicioB)
            if (new_start < curr_end) and (new_end > curr_start):
                return True # Hay conflicto
        except:
            continue

    return False
