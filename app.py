import streamlit as st
import pandas as pd
import datetime
import os
import uuid
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image as PILImage
from PIL import Image
from io import BytesIO
import numpy as np

# ---------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------
st.set_page_config(page_title="Distribución de Puestos", layout="wide")

# ---------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------
try:
    from modules.database import (
        get_conn, init_db, insert_distribution, clear_distribution,
        read_distribution_df, save_setting, get_all_settings,
        add_reservation, user_has_reservation, list_reservations_df,
        add_room_reservation, get_room_reservations_df,
        count_monthly_free_spots, delete_reservation_from_db, 
        delete_room_reservation_from_db, perform_granular_delete,
        ensure_reset_table, save_reset_token, validate_and_consume_token
    )
    from modules.seats import compute_distribution_from_excel, get_ideal_distribution_proposal, calculate_distribution_stats
    from modules.rooms import generate_time_slots, check_room_conflict
    from modules.zones import generate_colored_plan, load_zones, save_zones
    
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"⚠️ Error importando módulos: {e}")
    MODULES_LOADED = False

# ---------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
PLANOS_DIR = Path("planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

# Crear directorios necesarios
for directory in [DATA_DIR, PLANOS_DIR, COLORED_DIR]:
    directory.mkdir(exist_ok=True)

# ---------------------------------------------------------
# INICIALIZACIÓN
# ---------------------------------------------------------
st.title("🏢 Sistema de Gestión de Espacios - ACHS Servicios")

# Verificar módulos
if not MODULES_LOADED:
    st.error("❌ Error crítico: No se pudieron cargar los módulos necesarios")
    st.stop()

# Intentar conexión a base de datos
try:
    conn = get_conn()
    if conn is None:
        st.error("❌ No se pudo establecer conexión con la base de datos")
        st.stop()
    
    if "db_initialized" not in st.session_state:
        with st.spinner('Inicializando base de datos...'):
            init_db(conn)
        st.session_state["db_initialized"] = True
        
except Exception as e:
    st.error(f"❌ Error de conexión: {e}")
    st.stop()

# Cargar configuración
try:
    settings = get_all_settings(conn)
except:
    settings = {}

# ---------------------------------------------------------
# FUNCIONES MEJORADAS
# ---------------------------------------------------------
def apply_sorting_to_df(df):
    """Ordenar DataFrame correctamente"""
    if df.empty: 
        return df
    
    df = df.copy()
    
    # Identificar columnas
    col_piso = None
    col_dia = None
    for col in df.columns:
        col_str = str(col).lower()  # Convertir a string para evitar el error
        if 'piso' in col_str:
            col_piso = col
        elif 'dia' in col_str or 'día' in col_str:
            col_dia = col
    
    # Ordenar por piso y día
    if col_piso and col_dia:
        # Convertir a categórico para orden personalizado
        df[col_piso] = pd.Categorical(df[col_piso], 
                                    categories=sorted(df[col_piso].unique(), key=lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 0),
                                    ordered=True)
        df[col_dia] = pd.Categorical(df[col_dia], 
                                    categories=ORDER_DIAS,
                                    ordered=True)
        df = df.sort_values([col_piso, col_dia])
    
    return df

def calculate_weekly_usage_summary(distrib_df):
    """Calcular resumen semanal mejorado con porcentajes"""
    if distrib_df.empty: 
        return pd.DataFrame()
    
    try:
        # Filtrar solo equipos (excluir cupos libres)
        equipos_df = distrib_df[distrib_df['equipo'] != "Cupos libres"]
        if equipos_df.empty:
            return pd.DataFrame()
        
        # Calcular total semanal por equipo
        weekly = equipos_df.groupby('equipo').agg({
            'cupos': 'sum'
        }).reset_index()
        weekly.columns = ['Equipo', 'Total Cupos Semanales']
        
        # Calcular porcentaje de distribución semanal
        total_semanal = weekly['Total Cupos Semanales'].sum()
        if total_semanal > 0:
            weekly['% Distribución Semanal'] = (weekly['Total Cupos Semanales'] / total_semanal * 100).round(1)
        else:
            weekly['% Distribución Semanal'] = 0
        
        # Calcular porcentaje mensual (aproximado)
        weekly['% Mensual'] = (weekly['% Distribución Semanal'] * 4).round(1)
            
        return weekly[['Equipo', '% Distribución Semanal', '% Mensual']].sort_values('% Distribución Semanal', ascending=False)
    except Exception as e:
        st.error(f"Error calculando resumen: {e}")
        return pd.DataFrame()

def clean_reservation_df(df, tipo="puesto"):
    """Limpiar DataFrame de reservas"""
    if df.empty: 
        return df
    
    try:
        cols_drop = [c for c in df.columns if c.lower() in ['id', 'created_at', 'registro', 'id.1']]
        df = df.drop(columns=cols_drop, errors='ignore')
        
        if tipo == "puesto":
            rename_map = {
                'user_name': 'Nombre', 
                'user_email': 'Correo', 
                'piso': 'Piso', 
                'reservation_date': 'Fecha Reserva', 
                'team_area': 'Ubicación'
            }
            df = df.rename(columns=rename_map)
            desired_cols = ['Fecha Reserva', 'Piso', 'Ubicación', 'Nombre', 'Correo']
            existing_cols = [c for c in desired_cols if c in df.columns]
            return df[existing_cols]
            
        elif tipo == "sala":
            rename_map = {
                'user_name': 'Nombre', 
                'user_email': 'Correo', 
                'piso': 'Piso', 
                'room_name': 'Sala', 
                'reservation_date': 'Fecha', 
                'start_time': 'Inicio', 
                'end_time': 'Fin'
            }
            df = df.rename(columns=rename_map)
            desired_cols = ['Fecha', 'Inicio', 'Fin', 'Sala', 'Piso', 'Nombre', 'Correo']
            existing_cols = [c for c in desired_cols if c in df.columns]
            return df[existing_cols]
        return df
    except:
        return df

def generate_full_pdf_report(distrib_df, logo_path, deficit_data=None):
    """Generar reporte PDF completo"""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(True, 15)
        
        # Página 1: Distribución diaria
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        
        # Logo
        try:
            if Path(logo_path).exists():
                pdf.image(str(logo_path), x=10, y=8, w=30)
        except:
            pass
            
        pdf.ln(25)
        pdf.cell(0, 10, "Informe de Distribución", ln=True, align='C')
        pdf.ln(6)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, "1. Detalle de Distribución Diaria", ln=True)

        # Tabla Diaria
        pdf.set_font("Arial", 'B', 9)
        widths = [30, 60, 25, 25, 25]
        headers = ["Piso", "Equipo", "Día", "Cupos", "%Distrib"]    
        for w, h in zip(widths, headers): 
            pdf.cell(w, 6, h, 1)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        distrib_df_sorted = apply_sorting_to_df(distrib_df)
        
        for _, r in distrib_df_sorted.iterrows():
            piso = str(r.get('piso', ''))
            equipo = str(r.get('equipo', ''))[:35]
            dia = str(r.get('dia', ''))
            cupos = str(r.get('cupos', ''))
            pct = str(r.get('pct', '0'))
            
            pdf.cell(widths[0], 6, piso, 1)
            pdf.cell(widths[1], 6, equipo, 1)
            pdf.cell(widths[2], 6, dia, 1)
            pdf.cell(widths[3], 6, cupos, 1)
            pdf.cell(widths[4], 6, f"{pct}%", 1)
            pdf.ln()

        # Resumen semanal
        pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, "2. Resumen de Uso Semanal por Equipo", ln=True)
        
        weekly_summary = calculate_weekly_usage_summary(distrib_df)
        if not weekly_summary.empty:
            pdf.set_font("Arial", 'B', 9)
            w_wk = [80, 40, 40]
            h_wk = ["Equipo", "% Distrib Semanal", "% Mensual"]
            start_x = 25
            pdf.set_x(start_x)
            for w, h in zip(w_wk, h_wk): 
                pdf.cell(w, 6, h, 1)
            pdf.ln()
            
            pdf.set_font("Arial", '', 9)
            for _, row in weekly_summary.iterrows():
                pdf.set_x(start_x)
                pdf.cell(w_wk[0], 6, str(row["Equipo"])[:30], 1)
                pdf.cell(w_wk[1], 6, f"{row['% Distribución Semanal']}%", 1)
                pdf.cell(w_wk[2], 6, f"{row['% Mensual']}%", 1)
                pdf.ln()
        
        return pdf.output(dest='S').encode('latin-1')
        
    except Exception as e:
        st.error(f"Error generando PDF: {e}")
        return None

def get_available_slots(conn, piso, fecha):
    """Obtener cupos disponibles en tiempo real para un piso y fecha"""
    try:
        # Obtener distribución base
        df_distrib = read_distribution_df(conn)
        dia_nombre = ORDER_DIAS[fecha.weekday()]
        
        # Cupos totales para ese piso y día
        distrib_dia = df_distrib[(df_distrib['piso'] == piso) & 
                                (df_distrib['dia'] == dia_nombre) &
                                (df_distrib['equipo'] == 'Cupos libres')]
        
        if distrib_dia.empty:
            return 0, 0
        
        cupos_totales = int(distrib_dia.iloc[0]['cupos'])
        
        # Cupos ya reservados
        reservas_existentes = list_reservations_df(conn)
        if not reservas_existentes.empty:
            reservas_dia = reservas_existentes[
                (reservas_existentes['piso'] == piso) &
                (reservas_existentes['reservation_date'] == str(fecha)) &
                (reservas_existentes['team_area'] == 'Cupos libres')
            ]
            cupos_ocupados = len(reservas_dia)
        else:
            cupos_ocupados = 0
            
        cupos_disponibles = max(0, cupos_totales - cupos_ocupados)
        return cupos_disponibles, cupos_totales
        
    except Exception as e:
        st.error(f"Error calculando disponibilidad: {e}")
        return 0, 0

# ---------------------------------------------------------
# DIÁLOGOS DE CONFIRMACIÓN
# ---------------------------------------------------------
@st.dialog("Confirmar Distribución con Equipos Problemáticos")
def confirm_problematic_teams_dialog(equipos_problema):
    st.warning("⚠️ **Atención: Equipos con menos de 2 integrantes detectados**")
    
    for equipo in equipos_problema:
        st.error(f"• {equipo}")
    
    st.write("Estos equipos no podrán recibir el mínimo de 2 cupos por día.")
    st.write("¿Desea continuar con la distribución?")
    
    col1, col2 = st.columns(2)
    if col1.button("✅ Sí, continuar", type="primary", use_container_width=True):
        st.session_state.confirm_distribution = True
        st.rerun()
    if col2.button("❌ Cancelar", use_container_width=True):
        st.session_state.confirm_distribution = False
        st.rerun()

@st.dialog("Confirmar Borrado Masivo")
def confirm_mass_delete_dialog(opcion):
    st.warning(f"⚠️ **¿Está seguro que desea borrar {opcion}?**")
    st.error("⚠️ **Esta acción no se puede deshacer**")
    
    col1, col2 = st.columns(2)
    if col1.button("✅ Sí, borrar", type="primary", use_container_width=True):
        st.session_state.confirm_delete = True
        st.rerun()
    if col2.button("❌ Cancelar", use_container_width=True):
        st.session_state.confirm_delete = False
        st.rerun()

@st.dialog("Confirmar Guardar Distribución")
def confirm_save_distribution_dialog():
    st.info("💾 **¿Guardar esta distribución como definitiva?**")
    
    col1, col2 = st.columns(2)
    if col1.button("✅ Sí, guardar", type="primary", use_container_width=True):
        st.session_state.confirm_save = True
        st.rerun()
    if col2.button("❌ Cancelar", use_container_width=True):
        st.session_state.confirm_save = False
        st.rerun()

# ---------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------
menu = st.sidebar.selectbox("Navegación", ["Vista Pública", "Reservas", "Administrador"])

# ==========================================
# VISTA PÚBLICA
# ==========================================
if menu == "Vista Pública":
    st.header("📊 Distribución de Cupos")
    
    try:
        df = read_distribution_df(conn)
        
        if not df.empty:
            st.success(f"✅ Datos cargados: {len(df)} registros de distribución")
            
            # Mostrar datos principales con menús desplegables
            st.subheader("Distribución Completa por Piso")
            
            # Agrupar por piso
            pisos = sorted(df['piso'].unique())
            
            for piso in pisos:
                with st.expander(f"🏢 {piso} - Ver equipos y reservas", expanded=False):
                    df_piso = df[df['piso'] == piso]
                    
                    # Mostrar equipos del piso
                    st.write(f"**Equipos en {piso}:**")
                    equipos_piso = df_piso[df_piso['equipo'] != 'Cupos libres']['equipo'].unique()
                    
                    for equipo in equipos_piso:
                        with st.expander(f"👥 {equipo}", expanded=False):
                            df_equipo = df_piso[df_piso['equipo'] == equipo]
                            st.dataframe(df_equipo[['dia', 'cupos', 'pct']], use_container_width=True)
                    
                    # Mostrar cupos libres del piso
                    cupos_libres_piso = df_piso[df_piso['equipo'] == 'Cupos libres']
                    if not cupos_libres_piso.empty:
                        st.write(f"**📊 Cupos libres en {piso}:**")
                        st.dataframe(cupos_libres_piso[['dia', 'cupos']], use_container_width=True)
            
            # Resumen semanal mejorado con porcentajes
            st.subheader("📈 Resumen Semanal y Mensual por Equipo")
            weekly_summary = calculate_weekly_usage_summary(df)
            if not weekly_summary.empty:
                st.dataframe(weekly_summary, use_container_width=True)
                
                # Métricas resumen
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_equipos = len(weekly_summary)
                    st.metric("Total Equipos", total_equipos)
                with col2:
                    avg_weekly = weekly_summary['% Distribución Semanal'].mean()
                    st.metric("Promedio % Semanal", f"{avg_weekly:.1f}%")
                with col3:
                    avg_monthly = weekly_summary['% Mensual'].mean()
                    st.metric("Promedio % Mensual", f"{avg_monthly:.1f}%")
                with col4:
                    max_team = weekly_summary.loc[weekly_summary['% Distribución Semanal'].idxmax()]
                    st.metric("Mayor % Semanal", f"{max_team['% Distribución Semanal']}%")
            else:
                st.info("No hay datos suficientes para el resumen semanal")
                
        else:
            st.info("📝 No hay datos de distribución cargados. Por favor, genere una distribución en el panel de Administrador.")
            
    except Exception as e:
        st.error(f"Error cargando datos: {e}")

# ==========================================
# RESERVAS
# ==========================================
elif menu == "Reservas":
    st.header("🎯 Sistema de Reservas")
    
    # Obtener equipos de la distribución
    df_distrib = read_distribution_df(conn)
    equipos = []
    if not df_distrib.empty:
        equipos = sorted(df_distrib[df_distrib['equipo'] != 'Cupos libres']['equipo'].unique())
    
    opcion_reserva = st.selectbox(
        "Tipo de Reserva",
        ["🪑 Reservar Puesto Flex", "🏢 Reservar Sala de Reuniones", "📋 Mis Reservas Activas"],
        key="reserva_type"
    )
    
    st.markdown("---")

    if opcion_reserva == "🪑 Reservar Puesto Flex":
        st.subheader("Reserva de Puesto Flex")
        st.info("💡 Máximo 2 reservas por mes por equipo")
        
        if not equipos:
            st.warning("❌ No hay equipos cargados en el sistema. Genere una distribución primero.")
        else:
            # Formulario de reserva mejorado
            with st.form("form_puesto_flex"):
                col1, col2 = st.columns(2)
                
                with col1:
                    equipo = st.selectbox("Seleccione su Equipo", equipos, key="equipo_puesto")
                    fecha = st.date_input("Fecha de Reserva", 
                                        min_value=datetime.date.today(),
                                        key="fecha_puesto")
                    piso = st.selectbox("Piso", ["Piso 1", "Piso 2", "Piso 3"], key="piso_puesto")
                
                with col2:
                    email = st.text_input("Correo Electrónico del Equipo", 
                                        placeholder="equipo@empresa.com",
                                        key="email_puesto")
                    
                    # Mostrar información del día
                    dia_semana = ORDER_DIAS[fecha.weekday()] if fecha.weekday() < 5 else "Fin de Semana"
                    st.info(f"📅 Día seleccionado: **{dia_semana}**")
                
                # Mostrar disponibilidad en tiempo real
                if fecha and piso and fecha.weekday() < 5:
                    cupos_disponibles, cupos_totales = get_available_slots(conn, piso, fecha)
                    
                    # Mostrar indicador de disponibilidad
                    st.subheader("📊 Disponibilidad")
                    
                    col_avail1, col_avail2 = st.columns(2)
                    with col_avail1:
                        st.metric("Cupos Disponibles", cupos_disponibles)
                    with col_avail2:
                        st.metric("Capacidad Total", cupos_totales)
                    
                    # Barra de progreso
                    if cupos_totales > 0:
                        porcentaje_ocupado = ((cupos_totales - cupos_disponibles) / cupos_totales) * 100
                        st.progress(porcentaje_ocupado / 100)
                        st.caption(f"🟢 {cupos_disponibles} disponibles de {cupos_totales} totales")
                    
                    if cupos_disponibles == 0:
                        st.warning("⚠️ No hay cupos disponibles para esta fecha y piso")
                
                submitted = st.form_submit_button("📅 Confirmar Reserva", type="primary")
                
                if submitted:
                    if not equipo or not email:
                        st.error("❌ Complete todos los campos obligatorios")
                    elif fecha.weekday() >= 5:
                        st.error("❌ No se pueden realizar reservas los fines de semana")
                    elif cupos_disponibles <= 0:
                        st.error("❌ No hay cupos disponibles para esta fecha")
                    else:
                        # Verificar límite mensual MEJORADO
                        reservas_mes = count_monthly_free_spots(conn, email, fecha)
                        st.info(f"📊 Usted tiene {reservas_mes} reservas flex este mes")
                        
                        if reservas_mes >= 2:
                            st.error(f"❌ Ha alcanzado el límite máximo de 2 reservas flex por mes")
                        else:
                            # Verificar si ya tiene reserva para esta fecha
                            if user_has_reservation(conn, email, str(fecha)):
                                st.error("❌ Ya tiene una reserva registrada para esta fecha")
                            else:
                                try:
                                    add_reservation(conn, equipo, email, piso, str(fecha), "Cupos libres", 
                                                  datetime.datetime.now().isoformat())
                                    st.success(f"✅ Reserva confirmada para **{equipo}** el {fecha}")
                                    st.balloons()
                                except Exception as e:
                                    st.error(f"❌ Error al guardar reserva: {e}")

    elif opcion_reserva == "🏢 Reservar Sala de Reuniones":
        st.subheader("Reserva de Sala de Reuniones")
        
        if not equipos:
            st.warning("❌ No hay equipos cargados en el sistema. Genere una distribución primero.")
        else:
            with st.form("form_sala_reuniones"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sala = st.selectbox("Sala", [
                        "Sala Reuniones Pequeña Piso 1",
                        "Sala Reuniones Grande Piso 1", 
                        "Sala Reuniones Piso 2",
                        "Sala Reuniones Piso 3"
                    ], key="sala_select")
                    
                    fecha_sala = st.date_input("Fecha", 
                                             min_value=datetime.date.today(),
                                             key="fecha_sala")
                
                with col2:
                    equipo_sala = st.selectbox("Equipo Solicitante", equipos, key="equipo_sala")
                    email_sala = st.text_input("Correo Electrónico", 
                                             placeholder="equipo@empresa.com",
                                             key="email_sala")
                
                # Horarios
                st.subheader("Horario")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    hora_inicio = st.selectbox("Hora Inicio", 
                                             generate_time_slots("08:00", "20:00", 30),
                                             key="hora_inicio")
                with col_t2:
                    hora_fin = st.selectbox("Hora Fin", 
                                          generate_time_slots("08:30", "20:30", 30),
                                          index=1, key="hora_fin")
                
                submitted_sala = st.form_submit_button("🏢 Confirmar Reserva de Sala", type="primary")
                
                if submitted_sala:
                    if not equipo_sala or not email_sala:
                        st.error("❌ Complete todos los campos obligatorios")
                    else:
                        # Verificar conflicto
                        if check_room_conflict(get_room_reservations_df(conn).to_dict("records"), 
                                             str(fecha_sala), sala, hora_inicio, hora_fin):
                            st.error("❌ La sala ya está reservada en ese horario")
                        else:
                            try:
                                # Determinar piso basado en la sala
                                piso_sala = "Piso 1"
                                if "Piso 2" in sala:
                                    piso_sala = "Piso 2"
                                elif "Piso 3" in sala:
                                    piso_sala = "Piso 3"
                                
                                add_room_reservation(conn, equipo_sala, email_sala, piso_sala, sala, 
                                                   str(fecha_sala), hora_inicio, hora_fin,
                                                   datetime.datetime.now().isoformat())
                                st.success(f"✅ Sala **{sala}** reservada para **{equipo_sala}**")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Error al reservar sala: {e}")

    elif opcion_reserva == "📋 Mis Reservas Activas":
        st.subheader("Mis Reservas Activas")
        
        email_busqueda = st.text_input("Ingrese su correo electrónico para buscar reservas:",
                                     placeholder="equipo@empresa.com")
        
        if email_busqueda:
            # Buscar reservas de puestos
            reservas_puestos = clean_reservation_df(list_reservations_df(conn), "puesto")
            mis_puestos = pd.DataFrame()
            if not reservas_puestos.empty and 'Correo' in reservas_puestos.columns:
                mis_puestos = reservas_puestos[reservas_puestos['Correo'].str.contains(email_busqueda, case=False, na=False)]
            
            # Buscar reservas de salas
            reservas_salas = clean_reservation_df(get_room_reservations_df(conn), "sala")
            mis_salas = pd.DataFrame()
            if not reservas_salas.empty and 'Correo' in reservas_salas.columns:
                mis_salas = reservas_salas[reservas_salas['Correo'].str.contains(email_busqueda, case=False, na=False)]
            
            if mis_puestos.empty and mis_salas.empty:
                st.info("📝 No se encontraron reservas activas para este correo")
            else:
                if not mis_puestos.empty:
                    st.subheader("🪑 Mis Reservas de Puestos")
                    for idx, reserva in mis_puestos.iterrows():
                        with st.container(border=True):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"**{reserva['Nombre']}**")
                                st.write(f"📅 {reserva['Fecha Reserva']} | 📍 {reserva['Piso']} | 🏷️ {reserva['Ubicación']}")
                            with col2:
                                if st.button("❌ Cancelar", key=f"cancel_p_{idx}"):
                                    if delete_reservation_from_db(conn, reserva['Nombre'], reserva['Fecha Reserva'], reserva['Ubicación']):
                                        st.success("Reserva cancelada")
                                        st.rerun()
                
                if not mis_salas.empty:
                    st.subheader("🏢 Mis Reservas de Salas")
                    for idx, reserva in mis_salas.iterrows():
                        with st.container(border=True):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"**{reserva['Nombre']}**")
                                st.write(f"📅 {reserva['Fecha']} | 🕒 {reserva['Inicio']} - {reserva['Fin']}")
                                st.write(f"📍 {reserva['Sala']} | 🏢 {reserva['Piso']}")
                            with col2:
                                if st.button("❌ Cancelar", key=f"cancel_s_{idx}"):
                                    if delete_room_reservation_from_db(conn, reserva['Nombre'], reserva['Fecha'], reserva['Sala'], reserva['Inicio']):
                                        st.success("Reserva cancelada")
                                        st.rerun()

# ==========================================
# ADMINISTRADOR
# ==========================================
elif menu == "Administrador":
    st.header("⚙️ Panel de Administración")
    
    # Verificar credenciales
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
    
    if not st.session_state.admin_logged_in:
        st.subheader("🔐 Autenticación Requerida")
        
        col1, col2 = st.columns(2)
        with col1:
            usuario = st.text_input("Usuario", value="admin")
        with col2:
            password = st.text_input("Contraseña", type="password", value="admin123")
        
        if st.button("🔓 Ingresar", type="primary"):
            if usuario == "admin" and password == "admin123":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Credenciales incorrectas")
        st.stop()
    
    # Barra superior de administrador
    col_top1, col_top2, col_top3 = st.columns([3, 1, 1])
    with col_top1:
        st.success(f"🔓 Sesión activa: **Administrador**")
    with col_top2:
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()
    with col_top3:
        if st.button("🚪 Cerrar Sesión"):
            st.session_state.admin_logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # Pestañas de administrador
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distribución", "🎨 Editor Visual", "📋 Informes", "⚙️ Configuración", "🛠️ Mantenimiento"])
    
    with tab1:
        st.subheader("Generador de Distribución")
        
        uploaded_file = st.file_uploader("Subir archivo Excel", type=["xlsx"], 
                                       help="El archivo debe contener hojas 'Equipos' y 'Parámetros'")
        
        col_strat1, col_strat2 = st.columns(2)
        with col_strat1:
            ignore_params = st.checkbox("🎯 Usar distribución ideal (ignorar parámetros)", 
                                      value=True,
                                      help="Genera distribución optimizada sin restricciones de capacidad")
        with col_strat2:
            if ignore_params:
                estrategia = st.selectbox("Estrategia", 
                                        ["⚖️ Equitativa Perfecta", "🔄 Balanceada con Flex", "🎲 Aleatoria Controlada"])
            else:
                estrategia = st.selectbox("Estrategia Base", 
                                        ["🎲 Aleatorio", "🧩 Tetris", "🐜 Relleno"])
        
        strat_map = {
            "🎲 Aleatorio": "random",
            "🧩 Tetris": "size_desc", 
            "🐜 Relleno": "size_asc",
            "⚖️ Equitativa Perfecta": "perfect_equity",
            "🔄 Balanceada con Flex": "balanced_flex",
            "🎲 Aleatoria Controlada": "controlled_random"
        }
        
        if uploaded_file and st.button("🚀 Generar Distribución", type="primary"):
            try:
                df_equipos = pd.read_excel(uploaded_file, "Equipos")
                st.success(f"✅ Equipos cargados: {len(df_equipos)} equipos")
                
                if ignore_params:
                    # Generar distribución ideal
                    rows, equipos_problema = get_ideal_distribution_proposal(df_equipos, strategy=strat_map[estrategia])
                    deficit = []
                else:
                    df_parametros = pd.read_excel(uploaded_file, "Parámetros")
                    rows, deficit = compute_distribution_from_excel(df_equipos, df_parametros, strategy=strat_map[estrategia])
                    equipos_problema = []
                
                # Mostrar equipos problemáticos si los hay
                if equipos_problema:
                    st.session_state.problematic_teams = equipos_problema
                    st.session_state.generated_rows = rows
                    st.session_state.generated_deficit = deficit
                    confirm_problematic_teams_dialog(equipos_problema)
                else:
                    st.session_state.generated_rows = rows
                    st.session_state.generated_deficit = deficit
                    st.success("✅ Distribución generada exitosamente")
                
            except Exception as e:
                st.error(f"❌ Error procesando archivo: {e}")
        
        # Manejar confirmación de distribución
        if hasattr(st.session_state, 'confirm_distribution'):
            if st.session_state.confirm_distribution:
                st.success("✅ Distribución confirmada y generada")
                st.session_state.confirm_distribution = None
            else:
                st.info("❌ Distribución cancelada")
                st.session_state.confirm_distribution = None
        
        # Mostrar distribución generada
        if hasattr(st.session_state, 'generated_rows') and st.session_state.generated_rows:
            st.subheader("Vista Previa de Distribución")
            
            df_preview = pd.DataFrame(st.session_state.generated_rows)
            st.dataframe(apply_sorting_to_df(df_preview), use_container_width=True)
            
            # Estadísticas
            if st.session_state.generated_rows:
                stats = calculate_distribution_stats(st.session_state.generated_rows, df_equipos)
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                col_stat1.metric("Total Cupos", stats['total_cupos_asignados'])
                col_stat2.metric("Cupos Libres", stats['cupos_libres'])
                col_stat3.metric("Equipos con Déficit", stats['equipos_con_deficit'])
                col_stat4.metric("Uniformidad", f"{stats['uniformidad']:.1f}")
            
            # Botón para guardar
            if st.button("💾 Guardar Distribución Definitiva", type="primary"):
                confirm_save_distribution_dialog()
        
        # Manejar confirmación de guardado
        if hasattr(st.session_state, 'confirm_save'):
            if st.session_state.confirm_save:
                try:
                    clear_distribution(conn)
                    insert_distribution(conn, st.session_state.generated_rows)
                    st.success("✅ Distribución guardada exitosamente en la base de datos")
                    st.session_state.confirm_save = None
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error guardando distribución: {e}")
            else:
                st.info("❌ Guardado cancelado")
                st.session_state.confirm_save = None
    
    with tab2:
        st.subheader("Editor Visual de Planos")
        st.info("🎨 Esta funcionalidad permite editar zonas en los planos de los pisos")
        
        # Cargar zonas existentes
        zonas = load_zones()
        
        col_edit1, col_edit2 = st.columns(2)
        with col_edit1:
            piso_editor = st.selectbox("Piso", ["Piso 1", "Piso 2", "Piso 3"], key="editor_piso")
        with col_edit2:
            dia_editor = st.selectbox("Día de Referencia", ORDER_DIAS, key="editor_dia")
        
        # Mostrar plano base si existe
        piso_num = piso_editor.replace("Piso ", "").strip()
        plano_path = PLANOS_DIR / f"piso{piso_num}.png"
        
        if plano_path.exists():
            st.image(str(plano_path), caption=f"Plano base - {piso_editor}", use_container_width=True)
            
            # Editor simple de zonas
            with st.expander("✏️ Agregar/Editar Zonas"):
                st.info("Seleccione un equipo y defina su zona en el plano")
                
                equipos_disponibles = []
                df_distrib_editor = read_distribution_df(conn)
                if not df_distrib_editor.empty:
                    equipos_disponibles = sorted(df_distrib_editor['equipo'].unique())
                
                if equipos_disponibles:
                    equipo_zona = st.selectbox("Equipo", equipos_disponibles)
                    color_zona = st.color_picker("Color", "#00A04A")
                    
                    col_coord1, col_coord2 = st.columns(2)
                    with col_coord1:
                        x_pos = st.slider("Posición X", 0, 800, 100)
                        ancho = st.slider("Ancho", 10, 400, 100)
                    with col_coord2:
                        y_pos = st.slider("Posición Y", 0, 600, 100)
                        alto = st.slider("Alto", 10, 400, 80)
                    
                    if st.button("💾 Guardar Zona"):
                        if piso_editor not in zonas:
                            zonas[piso_editor] = []
                        
                        # Actualizar zona existente o agregar nueva
                        zona_existente = False
                        for i, zona in enumerate(zonas[piso_editor]):
                            if zona['team'] == equipo_zona:
                                zonas[piso_editor][i] = {
                                    'team': equipo_zona,
                                    'x': x_pos,
                                    'y': y_pos, 
                                    'w': ancho,
                                    'h': alto,
                                    'color': color_zona
                                }
                                zona_existente = True
                                break
                        
                        if not zona_existente:
                            zonas[piso_editor].append({
                                'team': equipo_zona,
                                'x': x_pos,
                                'y': y_pos,
                                'w': ancho,
                                'h': alto,
                                'color': color_zona
                            })
                        
                        save_zones(zonas)
                        st.success("✅ Zona guardada exitosamente")
                else:
                    st.warning("No hay equipos cargados en la distribución")
        else:
            st.warning(f"❌ No se encontró el plano para {piso_editor}")
            st.info("💡 Suba los planos en formato PNG a la carpeta 'planos/' con nombres: piso1.png, piso2.png, piso3.png")
    
    with tab3:
        st.subheader("Generador de Informes")
        
        # Informes de uso
        with st.expander("📈 Informes de Uso", expanded=True):
            st.subheader("Resumen de Reservas")
            
            reservas_puestos = clean_reservation_df(list_reservations_df(conn), "puesto")
            if not reservas_puestos.empty:
                st.write("**Reservas de Puestos por Equipo**")
                uso_equipos = reservas_puestos.groupby('Nombre').agg({
                    'Fecha Reserva': 'count',
                    'Correo': 'first'
                }).reset_index()
                uso_equipos = uso_equipos.rename(columns={'Fecha Reserva': 'Total Reservas'})
                uso_equipos = uso_equipos.sort_values('Total Reservas', ascending=False)
                st.dataframe(uso_equipos, use_container_width=True)
            
            reservas_salas = clean_reservation_df(get_room_reservations_df(conn), "sala")
            if not reservas_salas.empty:
                st.write("**Reservas de Salas por Equipo**")
                uso_salas = reservas_salas.groupby('Nombre').agg({
                    'Fecha': 'count',
                    'Correo': 'first',
                    'Sala': lambda x: ', '.join(x.unique())
                }).reset_index()
                uso_salas = uso_salas.rename(columns={'Fecha': 'Total Reservas'})
                uso_salas = uso_salas.sort_values('Total Reservas', ascending=False)
                st.dataframe(uso_salas, use_container_width=True)
        
        # Generar reportes
        st.subheader("Generar Reportes Completos")
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            formato_reporte = st.selectbox("Formato", ["PDF", "Excel"])
        
        with col_report2:
            if st.button("📊 Generar Reporte Completo", type="primary"):
                df_reporte = read_distribution_df(conn)
                if not df_reporte.empty:
                    if formato_reporte == "PDF":
                        pdf_bytes = generate_full_pdf_report(df_reporte, "static/logo.png")
                        if pdf_bytes:
                            st.download_button("📥 Descargar PDF", pdf_bytes, "reporte_distribucion.pdf", "application/pdf")
                        else:
                            st.error("❌ Error generando PDF")
                    else:
                        # Generar Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_reporte.to_excel(writer, sheet_name='Distribución', index=False)
                            weekly = calculate_weekly_usage_summary(df_reporte)
                            if not weekly.empty:
                                weekly.to_excel(writer, sheet_name='Resumen Semanal', index=False)
                        st.download_button("📥 Descargar Excel", output.getvalue(), "reporte_distribucion.xlsx", 
                                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("❌ No hay datos de distribución para generar reportes")
    
    with tab4:
        st.subheader("Configuración del Sistema")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.write("**Credenciales de Administrador**")
            nuevo_user = st.text_input("Nuevo Usuario", value="admin")
            nueva_pass = st.text_input("Nueva Contraseña", type="password", value="admin123")
            
            if st.button("💾 Guardar Credenciales"):
                save_setting(conn, "admin_user", nuevo_user)
                save_setting(conn, "admin_pass", nueva_pass)
                st.success("✅ Credenciales actualizadas")
        
        with col_config2:
            st.write("**Configuración General**")
            titulo_sitio = st.text_input("Título del Sitio", value=settings.get("site_title", "Gestor de Puestos y Salas — ACHS Servicios"))
            color_primario = st.color_picker("Color Primario", value=settings.get("primary", "#00A04A"))
            
            if st.button("🎨 Aplicar Configuración"):
                save_setting(conn, "site_title", titulo_sitio)
                save_setting(conn, "primary", color_primario)
                st.success("✅ Configuración aplicada")
    
    with tab5:
        st.subheader("Herramientas de Mantenimiento")
        st.warning("⚠️ **ADVERTENCIA**: Estas operaciones son irreversibles")
        
        opcion_borrado = st.selectbox(
            "Seleccione qué desea borrar:",
            ["Reservas", "Distribución", "Planos/Zonas", "TODO (Todo el contenido)"],
            key="delete_option"
        )
        
        if st.button("🗑️ Ejecutar Borrado Masivo", type="primary"):
            st.session_state.pending_delete = opcion_borrado
            confirm_mass_delete_dialog(opcion_borrado)
        
        # Manejar confirmación de borrado
        if hasattr(st.session_state, 'confirm_delete'):
            if st.session_state.confirm_delete:
                try:
                    resultado = perform_granular_delete(conn, st.session_state.pending_delete)
                    st.success(f"✅ {resultado}")
                    st.session_state.confirm_delete = None
                    st.session_state.pending_delete = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error durante el borrado: {e}")
            else:
                st.info("❌ Borrado cancelado")
                st.session_state.confirm_delete = None
                st.session_state.pending_delete = None
        
        # Información del sistema
        st.subheader("Información del Sistema")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            df_dist = read_distribution_df(conn)
            st.metric("Registros de Distribución", len(df_dist))
        
        with col_info2:
            df_reservas = list_reservations_df(conn)
            st.metric("Reservas de Puestos", len(df_reservas))
        
        with col_info3:
            df_salas = get_room_reservations_df(conn)
            st.metric("Reservas de Salas", len(df_salas))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sistema de Gestión de Espacios - ACHS Servicios v2.0 | "
    "Desarrollado para optimización de espacios de trabajo"
    "</div>", 
    unsafe_allow_html=True
)
