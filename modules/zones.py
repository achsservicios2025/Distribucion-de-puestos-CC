import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor
import textwrap
import math

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ZONES_FILE = DATA_DIR / "zones.json"
PLANOS_DIR = Path("planos")
COLORED_DIR = Path("planos_coloreados")
ZONES_PATH = ZONES_FILE 

PLANOS_DIR.mkdir(exist_ok=True)
COLORED_DIR.mkdir(exist_ok=True)

# --- HELPERS ---
def _hex_to_rgba(hex_color, alpha=100):
    try:
        rgb = ImageColor.getrgb(hex_color)
        return (rgb[0], rgb[1], rgb[2], alpha)
    except: return (0, 255, 0, 100)

def load_zones():
    if not ZONES_FILE.exists(): return {}
    try:
        with open(ZONES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_zones(data):
    with open(ZONES_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)

def get_custom_font(font_name, size):
    """Carga fuentes del sistema disponibles."""
    font_files = {
        "Arial": "arial.ttf", "Arial Black": "ariblk.ttf", 
        "Calibri": "calibri.ttf", "Comic Sans MS": "comic.ttf",
        "Courier New": "cour.ttf", "Georgia": "georgia.ttf",
        "Impact": "impact.ttf", "Lucida Console": "lucon.ttf",
        "Roboto": "Roboto-Regular.ttf", "Segoe UI": "segoeui.ttf",
        "Tahoma": "tahoma.ttf", "Times New Roman": "times.ttf",
        "Trebuchet MS": "trebuc.ttf", "Verdana": "verdana.ttf"
    }
    filename = font_files.get(font_name, "arial.ttf")
    try:
        # Buscar en el directorio actual o en el sistema (dependiendo de la instalación PIL/sistema)
        return ImageFont.truetype(filename, size)
    except:
        try: return ImageFont.truetype("arial.ttf", size)
        except: return ImageFont.load_default()

# --- HEADER CON TÍTULO Y SUBTÍTULO (POSICIÓN CENTRAL) ---
def create_header_image(width, config, logo_path=None):
    bg_color = config.get("bg_color", "#FFFFFF")
    use_logo = config.get("use_logo", False)
    logo_w_req = config.get("logo_width", 150)
    logo_align = config.get("logo_align", "Izquierda") # OBTENEMOS LA NUEVA ALINEACIÓN
    
    # Título
    t_text = config.get("title_text", "")
    t_font = config.get("title_font", "Arial")
    t_size = config.get("title_size", 36)
    t_color = config.get("title_color", "#000000")
    t_align = config.get("alignment", "Centro") 
    
    # Subtítulo
    s_text = config.get("subtitle_text", "")
    s_font = config.get("subtitle_font", "Arial") 
    s_size = config.get("subtitle_size", 24)     
    s_color = config.get("subtitle_color", "#666666")
    s_align = config.get("subtitle_align", "Centro") 
    
    # Calcular alturas
    font_t = get_custom_font(t_font, t_size)
    font_s = get_custom_font(s_font, s_size)
    
    try: h_t = font_t.getbbox(t_text)[3] - font_t.getbbox(t_text)[1] if t_text else 0
    except: h_t = t_size # Fallback
    try: h_s = font_s.getbbox(s_text)[3] - font_s.getbbox(s_text)[1] if s_text else 0
    except: h_s = s_size # Fallback
    
    gap_px = 20 if s_text else 0 
    padding_top = 40
    padding_bottom = 40
    
    logo_h = 0
    logo_w = 0
    
    if use_logo and logo_path and os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        asp = logo_img.height / logo_img.width
        logo_w = logo_w_req
        logo_h = int(logo_w * asp)
    
    # Altura requerida para logo + textos
    content_height = logo_h + (padding_top if logo_h else 0) + h_t + gap_px + h_s
    
    # Altura final (mínimo 150)
    header_height = max(150, content_height + padding_top + padding_bottom)
    
    img = Image.new("RGB", (width, header_height), bg_color)
    draw = ImageDraw.Draw(img)

    # 1. Posición del Bloque (Logo + Textos)
    block_y_start = (header_height - content_height) // 2
    
    current_y = block_y_start
    
    # 2. Logo
    if use_logo and logo_path and os.path.exists(logo_path):
        try:
            logo_img = logo_img.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
            
            # Cálculo de la posición X del logo basada en logo_align
            if logo_align == "Izquierda":
                x_pos_logo = 30
            elif logo_align == "Derecha":
                x_pos_logo = width - logo_w - 30
            else: # Centro
                x_pos_logo = (width - logo_w) // 2 
            
            img.paste(logo_img, (x_pos_logo, current_y), logo_img)
            current_y += logo_h + padding_top # Mover Y debajo del logo
        except Exception as e: 
            print(f"Error al cargar/posicionar logo: {e}")
            pass
            
    # Helper alineación
    # Esta función determina la posición X del texto (Título/Subtítulo) en el ancho total
    def get_x(txt, fnt, align):
        try:
            w = fnt.getbbox(txt)[2] - fnt.getbbox(txt)[0]
        except: 
            w = 0 
            
        if align == "Izquierda": return 30
        elif align == "Derecha": return width - w - 30
        else: return (width - w) // 2

    # 3. Dibujar Textos (Debajo del logo si existe, o en la parte superior del bloque)
    
    if t_text:
        tx = get_x(t_text, font_t, t_align)
        draw.text((tx, current_y), t_text, font=font_t, fill=t_color)
        current_y += h_t + gap_px
    
    if s_text:
        sx = get_x(s_text, font_s, s_align)
        draw.text((sx, current_y), s_text, font=font_s, fill=s_color)

    return img

    # 3. Dibujar Textos (Debajo del logo si existe, o en la parte superior del bloque)
    
    if t_text:
        tx = get_x(t_text, font_t, t_align)
        draw.text((tx, current_y), t_text, font=font_t, fill=t_color)
        current_y += h_t + gap_px
    
    if s_text:
        sx = get_x(s_text, font_s, s_align)
        draw.text((sx, current_y), s_text, font=font_s, fill=s_color)

    return img

# --- LEYENDA (Añadido estilo configurable y alineación) ---
def create_legend_image(zones_list, width, seat_counts, config, bg_color="white"):
    unique_teams = {}
    for z in zones_list: unique_teams[z['team']] = z['color']
    if not unique_teams: return None

    # Configuración de Leyenda (Título y Elementos)
    leg_font = config.get("legend_font", "Arial")
    leg_size = config.get("legend_size", 14)
    leg_align = config.get("legend_align", "Izquierda") # USANDO NUEVO PARÁMETRO
    
    font = get_custom_font(leg_font, leg_size)
    title_font = get_custom_font(leg_font, int(leg_size * 1.5)) # Título un poco más grande
    
    padding = 30; row_h = int(leg_size * 2.5); circ = int(leg_size * 0.7)
    if row_h < 40: row_h = 40
    
    n = len(unique_teams)
    cols = 1
    if n > 8: cols = 2
    if n > 16: cols = 3
    rows = math.ceil(n / cols)
    
    total_h = (rows * row_h) + (padding * 2) + row_h
    img = Image.new("RGB", (width, total_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Helper de alineación para la leyenda
    def get_x_legend(txt, fnt, align):
        try: w = fnt.getbbox(txt)[2] - fnt.getbbox(txt)[0]
        except: w = 0
        if align == "Izquierda": return padding
        elif align == "Derecha": return width - w - padding
        else: return (width - w) // 2
        
    # 1. Título de Leyenda
    title_text = "Referencias / Equipos:"
    title_x = get_x_legend(title_text, title_font, leg_align)
    draw.text((title_x, 20), title_text, font=title_font, fill="black")
    
    sy = padding + row_h
    cw = (width - (padding*2)) / cols
    
    # 2. Elementos de la Leyenda
    for i, (tm, clr) in enumerate(unique_teams.items()):
        r_idx = i // cols; c_idx = i % cols
        py = sy + (r_idx * row_h)
        
        cp = seat_counts.get(tm, 0)
        lbl = f"{tm} ({cp} cupos)" if "Sala" not in tm else tm

        # Calcular posición X para el elemento
        if leg_align == "Izquierda":
            # Si es izquierda, se respeta el margen y las columnas
            px_start = padding + (c_idx * cw)
            px_text = px_start + circ*2 + 20
        else:
            # Para Centro/Derecha, centramos el texto dentro del ancho disponible de la columna
            col_start = padding + (c_idx * cw)
            col_end = col_start + cw
            
            w_lbl = font.getbbox(lbl)[2] - font.getbbox(lbl)[0]
            w_item = (circ * 2) + 20 + w_lbl
            
            if leg_align == "Centro":
                px_start = col_start + (cw - w_item) // 2
            else: # Derecha
                px_start = col_end - w_item
                
            px_text = px_start + circ*2 + 20

        # Dibujar Círculo
        draw.ellipse([px_start, py, px_start + circ*2, py + circ*2], fill=clr, outline="black", width=2)
        
        # Dibujar Texto
        draw.text((px_text, py + (circ/2)), lbl, font=font, fill="black")
        
    return img

# --- GENERADOR PRINCIPAL ---
def generate_colored_plan(piso_name, dia_name, seat_counts_dict, output_format="PNG", header_config=None, logo_path=None):
    zones_data = load_zones()
    if piso_name not in zones_data or not zones_data[piso_name]: return None

    piso_num = piso_name.replace("Piso ", "").strip()
    img_path = PLANOS_DIR / f"piso {piso_num}.png"
    if not img_path.exists(): img_path = PLANOS_DIR / f"piso {piso_num}.jpg"
    if not img_path.exists(): return None

    try:
        # 1. Mapa
        base = Image.open(img_path).convert("RGBA")
        ov = Image.new("RGBA", base.size, (255,255,255,0))
        d_ov = ImageDraw.Draw(ov)
        
        current = zones_data[piso_name]
        for z in current:
            x,y,w,h = z['x'], z['y'], z['w'], z['h']
            rgb = _hex_to_rgba(z['color'], 100)
            # Solo borde negro, sin texto
            d_ov.rectangle([x,y,x+w,y+h], fill=rgb, outline="black", width=2)

        map_img = Image.alpha_composite(base, ov).convert("RGB")
        fw = map_img.width

        # 2. Header & Leyenda
        if header_config is None: header_config = {}
        head_img = create_header_image(fw, header_config, logo_path)
        
        # LÓGICA DE VISIBILIDAD DE LEYENDA AGREGADA AQUÍ
        use_legend = header_config.get("use_legend", True) 
        leg_img = None
        if use_legend:
            leg_img = create_legend_image(current, fw, seat_counts_dict, header_config)

        # 3. Unir (Verticalmente)
        parts = [head_img, map_img]
        if leg_img: parts.append(leg_img)
        
        th = sum(p.height for p in parts)
        final = Image.new("RGB", (fw, th), "white")
        cy = 0
        for p in parts:
            final.paste(p, (0, cy))
            cy += p.height

        # 4. Guardar
        ext = "pdf" if output_format == "PDF" else "png"
        ds = dia_name.lower().replace("é","e").replace("á","a").replace("í","i").replace("ó","o").replace("ú","u")
        out_n = f"piso_{piso_num}_{ds}_combined.{ext}"
        out_p = COLORED_DIR / out_n
        
        # Guardar con calidad máxima. 
        # Para PDF, PIL ajusta automáticamente el tamaño de la hoja a la imagen
        final.save(out_p, quality=95)
            
        return out_p

    except Exception as e:
        print(f"Error: {e}")
        return None