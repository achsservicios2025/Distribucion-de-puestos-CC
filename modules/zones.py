import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ZONES_FILE = DATA_DIR / "zones.json"

PLANOS_DIR = Path("planos")
COLORED_DIR = Path("planos_coloreados")
PLANOS_DIR.mkdir(exist_ok=True)
COLORED_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# IO
# ---------------------------------------------------------
def load_zones():
    if not ZONES_FILE.exists():
        return {}
    try:
        with open(ZONES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_zones(data):
    try:
        with open(ZONES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _safe_int(x, default=0):
    try:
        return int(round(float(str(x).replace(",", "."))))
    except Exception:
        return default


def _hex_to_rgba(hex_color: str, alpha=100):
    try:
        r, g, b = ImageColor.getrgb(hex_color)
        return (r, g, b, alpha)
    except Exception:
        return (0, 160, 74, alpha)


def _normalize_piso_num(piso_name: str) -> str:
    s = str(piso_name or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or "1"


def _normalize_day_slug(dia_name: str) -> str:
    s = (dia_name or "").strip().lower()
    trans = str.maketrans({"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u"})
    return s.translate(trans).replace(" ", "")


def _find_plan_path(piso_num: str) -> Path | None:
    # acepta: piso1.png / piso 1.png / piso_1.png / Piso1.png + jpg
    candidates = [
        PLANOS_DIR / f"piso{piso_num}.png",
        PLANOS_DIR / f"piso{piso_num}.jpg",
        PLANOS_DIR / f"piso {piso_num}.png",
        PLANOS_DIR / f"piso {piso_num}.jpg",
        PLANOS_DIR / f"piso_{piso_num}.png",
        PLANOS_DIR / f"piso_{piso_num}.jpg",
        PLANOS_DIR / f"Piso{piso_num}.png",
        PLANOS_DIR / f"Piso{piso_num}.jpg",
        PLANOS_DIR / f"Piso {piso_num}.png",
        PLANOS_DIR / f"Piso {piso_num}.jpg",
    ]
    return next((p for p in candidates if p.exists()), None)


def _get_font(font_name: str, size: int):
    # Muy tolerante: en servers linux muchas fuentes no están.
    # Usa DejaVuSans (suele venir), o default.
    size = max(8, int(size or 12))
    font_candidates = []
    if font_name:
        font_candidates.append(font_name)
    font_candidates.extend([
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
    ])
    for fn in font_candidates:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if not text:
        return (0, 0)
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return (box[2] - box[0], box[3] - box[1])
    except Exception:
        # fallback
        return (len(text) * 7, 12)


def _x_by_align(total_w: int, item_w: int, align: str, pad: int):
    a = (align or "").strip().lower()
    if a in ("izquierda", "left"):
        return pad
    if a in ("derecha", "right"):
        return max(pad, total_w - item_w - pad)
    return max(pad, (total_w - item_w) // 2)


def _draw_header(width: int, header_config: dict, logo_bytes_or_path=None):
    """
    header_config soporta:
      - show_title (bool) / title_text / title_align / title_font_size / title_color
      - show_logo (bool) / logo_position (Izquierda/Centro/Derecha) / logo_width
      - bg_color
      - subtitle_text (opcional)
    """
    cfg = header_config or {}
    bg = cfg.get("bg_color", "#FFFFFF")

    show_logo = bool(cfg.get("show_logo", cfg.get("use_logo", False)))
    logo_pos = cfg.get("logo_position", cfg.get("logo_align", "Izquierda"))
    logo_w = _safe_int(cfg.get("logo_width", 140), 140)

    show_title = bool(cfg.get("show_title", True))
    title_text = str(cfg.get("title_text", "") or "")
    title_align = cfg.get("title_align", cfg.get("alignment", "Centro"))
    title_size = _safe_int(cfg.get("title_font_size", cfg.get("title_size", 22)), 22)
    title_color = cfg.get("title_color", "#000000")
    title_font = cfg.get("title_font", "DejaVuSans.ttf")

    subtitle_text = str(cfg.get("subtitle_text", "") or "")
    subtitle_align = cfg.get("subtitle_align", "Centro")
    subtitle_size = _safe_int(cfg.get("subtitle_size", 14), 14)
    subtitle_color = cfg.get("subtitle_color", "#666666")
    subtitle_font = cfg.get("subtitle_font", "DejaVuSans.ttf")

    pad_x = 30
    pad_y = 18
    gap = 10

    # pre-cálculo tamaños texto
    tmp = Image.new("RGB", (width, 10), bg)
    d = ImageDraw.Draw(tmp)

    font_t = _get_font(title_font, title_size)
    font_s = _get_font(subtitle_font, subtitle_size)

    tw, th = _text_size(d, title_text, font_t) if (show_title and title_text) else (0, 0)
    sw, sh = _text_size(d, subtitle_text, font_s) if subtitle_text else (0, 0)

    # cargar logo (si aplica)
    logo_img = None
    if show_logo and logo_bytes_or_path:
        try:
            if isinstance(logo_bytes_or_path, (bytes, bytearray)):
                from io import BytesIO
                logo_img = Image.open(BytesIO(logo_bytes_or_path)).convert("RGBA")
            else:
                if os.path.exists(str(logo_bytes_or_path)):
                    logo_img = Image.open(str(logo_bytes_or_path)).convert("RGBA")
        except Exception:
            logo_img = None

    logo_h = 0
    if logo_img:
        try:
            aspect = logo_img.height / max(1, logo_img.width)
            logo_h = int(logo_w * aspect)
        except Exception:
            logo_img = None
            logo_h = 0

    # altura header: logo + textos
    content_h = 0
    if logo_img:
        content_h += logo_h
        content_h += gap
    if show_title and title_text:
        content_h += th
        content_h += (gap if subtitle_text else 0)
    if subtitle_text:
        content_h += sh

    header_h = max(120, content_h + pad_y * 2)
    header = Image.new("RGB", (width, header_h), bg)
    draw = ImageDraw.Draw(header)

    y = (header_h - content_h) // 2

    # logo
    if logo_img:
        try:
            logo_img = logo_img.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
            lx = _x_by_align(width, logo_w, logo_pos, pad_x)
            header.paste(logo_img, (lx, y), logo_img)
            y += logo_h + gap
        except Exception:
            pass

    # título
    if show_title and title_text:
        x = _x_by_align(width, tw, title_align, pad_x)
        draw.text((x, y), title_text, font=font_t, fill=title_color)
        y += th + (gap if subtitle_text else 0)

    # subtítulo
    if subtitle_text:
        x = _x_by_align(width, sw, subtitle_align, pad_x)
        draw.text((x, y), subtitle_text, font=font_s, fill=subtitle_color)

    return header


def _draw_legend(width: int, legend_items: list[tuple[str, str, int]], header_config: dict):
    """
    legend_items: [(equipo, color_hex, cupos_int), ...]
    header_config:
      - show_legend / use_legend
      - legend_align, legend_font, legend_size
    """
    cfg = header_config or {}
    show_legend = bool(cfg.get("show_legend", cfg.get("use_legend", True)))
    if not show_legend:
        return None

    if not legend_items:
        return None

    bg = cfg.get("bg_color", "#FFFFFF")
    legend_align = cfg.get("legend_align", "Izquierda")
    legend_font = cfg.get("legend_font", "DejaVuSans.ttf")
    legend_size = _safe_int(cfg.get("legend_size", 14), 14)

    pad = 24
    row_h = max(36, int(legend_size * 2.4))
    circ = max(10, int(legend_size * 0.9))
    title_font = _get_font(legend_font, int(legend_size * 1.3))
    item_font = _get_font(legend_font, legend_size)

    # columnas dinámicas
    n = len(legend_items)
    cols = 1
    if n > 8: cols = 2
    if n > 16: cols = 3
    rows = (n + cols - 1) // cols
    col_w = int((width - 2 * pad) / cols)

    title_text = "Leyenda"
    tmp = Image.new("RGB", (width, 10), bg)
    d = ImageDraw.Draw(tmp)
    t_w, t_h = _text_size(d, title_text, title_font)

    total_h = pad + t_h + pad + rows * row_h + pad
    img = Image.new("RGB", (width, total_h), bg)
    draw = ImageDraw.Draw(img)

    tx = _x_by_align(width, t_w, legend_align, pad)
    draw.text((tx, pad), title_text, font=title_font, fill="#000000")

    start_y = pad + t_h + pad

    for i, (team, color, cupos) in enumerate(legend_items):
        r = i // cols
        c = i % cols
        x0 = pad + c * col_w
        y0 = start_y + r * row_h

        label = f"{team} ({cupos})"

        # alineación dentro de columna
        tmp2 = Image.new("RGB", (10, 10), bg)
        d2 = ImageDraw.Draw(tmp2)
        lbl_w, lbl_h = _text_size(d2, label, item_font)
        item_w = (circ * 2) + 12 + lbl_w

        if str(legend_align).lower() in ("derecha", "right"):
            x_item = x0 + col_w - item_w
        elif str(legend_align).lower() in ("centro", "center"):
            x_item = x0 + (col_w - item_w) // 2
        else:
            x_item = x0

        # círculo
        draw.ellipse(
            [x_item, y0 + 4, x_item + circ * 2, y0 + 4 + circ * 2],
            fill=color,
            outline="#000000",
            width=2,
        )
        draw.text((x_item + circ * 2 + 12, y0 + 2), label, font=item_font, fill="#000000")

    return img


def _zone_day(z):
    # editor guarda "dia" o "Día" o "day"
    for k in ("dia", "Día", "day"):
        if k in z and str(z[k]).strip():
            return str(z[k]).strip()
    return ""


def _zone_team(z):
    # editor guarda "equipo" o "team"
    for k in ("equipo", "team", "Equipo"):
        if k in z and str(z[k]).strip():
            return str(z[k]).strip()
    return ""


def _zone_color(z):
    return str(z.get("color") or z.get("stroke") or "#00A04A")


def _zone_rect(z):
    # soporta left/top/width/height y legacy x/y/w/h
    if "left" in z or "top" in z:
        x = float(z.get("left", 0))
        y = float(z.get("top", 0))
        w = float(z.get("width", 0))
        h = float(z.get("height", 0))
        return x, y, w, h
    x = float(z.get("x", 0))
    y = float(z.get("y", 0))
    w = float(z.get("w", 0))
    h = float(z.get("h", 0))
    return x, y, w, h


# ---------------------------------------------------------
# Main render
# ---------------------------------------------------------
def generate_colored_plan(
    piso_name: str,
    dia_name: str,
    seat_counts_dict: dict,
    output_format: str = "PNG",
    header_config: dict | None = None,
    logo_source=None
):
    """
    Genera un PNG/PDF combinado:
      Header (título/logo opcional) + Plano con rectángulos + Leyenda (opcional)

    - Filtra zonas por día (dia_name) si la zona tiene dia.
    - seat_counts_dict se usa para cupos en la leyenda:
        seat_counts_dict[equipo] = cupos
    - logo_source: bytes (logo cargado en DB) o path str
    """
    zones_data = load_zones()
    if not zones_data:
        return None

    piso_key = str(piso_name).strip()
    floor_zones = zones_data.get(piso_key) or []
    if not floor_zones:
        return None

    piso_num = _normalize_piso_num(piso_key)
    plan_path = _find_plan_path(piso_num)
    if not plan_path:
        return None

    cfg = header_config or {}

    # Filtrar por día (si existe el campo en la zona)
    dia_target = str(dia_name).strip()
    filtered = []
    for z in floor_zones:
        z_dia = _zone_day(z)
        if z_dia and dia_target and z_dia != dia_target:
            continue
        filtered.append(z)

    if not filtered:
        # sin zonas para ese día, igual puedes exportar solo header+map,
        # pero tu app normalmente espera zonas; devolvemos None para avisar.
        return None

    # Abrir plano
    base = Image.open(plan_path).convert("RGBA")

    # Overlay
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)

    for z in filtered:
        x, y, w, h = _zone_rect(z)
        color = _zone_color(z)
        fill = _hex_to_rgba(color, alpha=90)

        x2 = x + w
        y2 = y + h

        # seguridad
        x = max(0, min(base.width, int(round(x))))
        y = max(0, min(base.height, int(round(y))))
        x2 = max(0, min(base.width, int(round(x2))))
        y2 = max(0, min(base.height, int(round(y2))))

        if x2 <= x or y2 <= y:
            continue

        d.rectangle([x, y, x2, y2], fill=fill, outline="#000000", width=2)

    map_img = Image.alpha_composite(base, overlay).convert("RGB")
    fw = map_img.width

    # Header
    header_img = _draw_header(fw, cfg, logo_source)

    # Legend items (únicos por equipo)
    # color = el de la primera zona del equipo en ese día
    uniq = {}
    for z in filtered:
        team = _zone_team(z)
        if not team:
            continue
        if team not in uniq:
            uniq[team] = _zone_color(z)

    legend_items = []
    for team, color in uniq.items():
        cupos = _safe_int(seat_counts_dict.get(team, 0), 0)
        legend_items.append((team, color, cupos))

    # orden por nombre equipo
    legend_items.sort(key=lambda x: x[0].lower())

    legend_img = _draw_legend(fw, legend_items, cfg)

    # Merge vertical
    parts = [header_img, map_img]
    if legend_img is not None:
        parts.append(legend_img)

    total_h = sum(p.height for p in parts)
    final = Image.new("RGB", (fw, total_h), cfg.get("bg_color", "#FFFFFF"))
    y = 0
    for p in parts:
        final.paste(p, (0, y))
        y += p.height

    # Save
    ext = "pdf" if str(output_format).upper() == "PDF" else "png"
    ds = _normalize_day_slug(dia_name)
    out_name = f"piso_{piso_num}_{ds}_combined.{ext}"
    out_path = COLORED_DIR / out_name

    try:
        if ext == "pdf":
            final.save(out_path, format="PDF", resolution=150.0)
        else:
            final.save(out_path, format="PNG", optimize=True)
        return out_path
    except Exception:
        return None
