import streamlit.components.v1 as components
import os
from pathlib import Path

# Obtener la ruta del directorio del componente
_component_dir = Path(__file__).parent
_frontend_dir = _component_dir / "frontend"

# Declarar el componente
_component_func = components.declare_component(
    "zone_editor",
    path=str(_frontend_dir)
)

def zone_editor(
    img_path,
    existing_zones,
    selected_team="",
    selected_color="#00A04A",
    width=700,
    key=None
):
    """
    Componente de editor de zonas con guardado automático.
    
    Parameters:
    -----------
    img_path : str
        Ruta a la imagen del plano
    existing_zones : list
        Lista de zonas existentes
    selected_team : str
        Equipo seleccionado
    selected_color : str
        Color seleccionado (hex)
    width : int
        Ancho del componente
    key : str
        Clave única para el componente
    
    Returns:
    --------
    dict o None
        Diccionario con las zonas actualizadas cuando se guarda, None en otros casos
    """
    # Convertir imagen a base64
    import base64
    try:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
    except Exception as e:
        return None
    
    # Preparar datos para el componente
    # Asegurarse de que existing_zones sea una lista
    if not isinstance(existing_zones, list):
        existing_zones = []
    
    component_value = _component_func(
        img_data=img_data,
        existing_zones=existing_zones,
        selected_team=selected_team,
        selected_color=selected_color,
        width=width,
        key=key,
        default=None
    )
    
    # Retornar el valor del componente (None si no hay cambios)
    return component_value

