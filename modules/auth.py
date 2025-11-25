# modules/auth.py
import streamlit as st
from modules.database import get_all_settings

def get_admin_credentials(conn):
    # Prefer settings table; fallback to st.secrets
    settings = get_all_settings(conn)
    user = settings.get("admin_user", None)
    pwd = settings.get("admin_pass", None)
    if not user or not pwd:
        # try secrets
        try:
            sec = st.secrets["admin"]
            user = user or sec.get("username","admin")
            pwd = pwd or sec.get("password","admin123")
        except Exception:
            user = user or "admin"
            pwd = pwd or "admin123"
    return user, pwd
