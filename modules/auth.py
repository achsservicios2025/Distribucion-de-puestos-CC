# modules/auth.py
import streamlit as st
from modules.database import get_all_settings

def get_admin_credentials(conn):
    """
    Recupera las credenciales de administrador: 
    1. Base de datos (settings table). 
    2. Fallback a st.secrets["admin"].
    3. Fallback final a valores por defecto ('admin', 'admin123').
    """
    
    # 1. Intentar obtener credenciales de la base de datos (cacheada)
    settings = get_all_settings(conn)
    
    # Si get_all_settings falla (ej. error de conexión), devuelve un diccionario vacío {}
    if settings is None:
        settings = {}
        
    user = settings.get("admin_user", None)
    pwd = settings.get("admin_pass", None)
    
    # 2. Fallback a st.secrets si la DB no tiene los valores
    if not user or not pwd:
        try:
            sec = st.secrets.get("admin", {}) # Usamos .get para evitar KeyError si [admin] no existe
            # Usamos el valor de la DB si existe; de lo contrario, usamos el secreto.
            user = user or sec.get("username", None)
            pwd = pwd or sec.get("password", None)
        except Exception:
            # st.secrets no está cargado o falla, no hacemos nada
            pass
            
    # 3. Fallback final a valores por defecto si no se encontró nada
    user = user or "admin"
    pwd = pwd or "admin123"
    
    return user, pwd

