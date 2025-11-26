from datetime import datetime, timedelta # CORREGIDO

def generate_time_slots(start_str, end_str, interval_minutes):
# ... (sin cambios de funcionalidad, solo limpieza) ...
    try:
        start = datetime.strptime(start_str, "%H:%M")
        end = datetime.strptime(end_str, "%H:%M")
        slots = []
        while start <= end:
            slots.append(start.strftime("%H:%M"))
            start += timedelta(minutes=interval_minutes)
        return slots
    except:
        return ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00"]

def check_room_conflict(reservations, check_date, check_room, check_start, check_end):
# ... (sin cambios de funcionalidad, solo limpieza) ...
    try:
        new_start = datetime.strptime(check_start, "%H:%M")
        new_end = datetime.strptime(check_end, "%H:%M")
    except:
        return False
        
    for r in reservations:
        r_date = r.get("reservation_date") or r.get("fecha")
        r_room = r.get("room_name") or r.get("sala")
        r_start = r.get("start_time") or r.get("inicio")
        r_end = r.get("end_time") or r.get("fin")

        if not (r_date and r_room and r_start and r_end):
            continue

        if str(r_date) != str(check_date) or r_room != check_room:
            continue
            
        try:
            curr_start = datetime.strptime(r_start, "%H:%M")
            curr_end = datetime.strptime(r_end, "%H:%M")
            
            if (new_start < curr_end) and (new_end > curr_start):
                return True
        except:
            continue

    return False
