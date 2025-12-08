import streamlit as st
import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Optional
import random
import datetime
import numpy as np
from io import BytesIO
from PIL import Image
from fpdf import FPDF

# ---------------------------------------------------------
# 1) CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gestor de Puestos y Salas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2) IMPORTS MÓDULOS
# ---------------------------------------------------------
from modules.database import (
    get_conn, init_db, insert_distribution, clear_distribution,
    read_distribution_df, save_setting, get_all_settings,
    add_reservation, user_has_reservation, list_reservations_df,
    add_room_reservation, get_room_reservations_df,
    count_monthly_free_spots, delete_reservation_from_db,
    delete_room_reservation_from_db, perform_granular_delete,
    ensure_reset_table, save_reset_token, validate_and_consume_token,
    get_worksheet
)

try:
    from modules.database import delete_distribution_row, delete_distribution_rows_by_indices
except ImportError:
    def delete_distribution_row(conn, piso, equipo, dia):
        return False

    def delete_distribution_rows_by_indices(conn, indices):
        return False

from modules.auth import get_admin_credentials
from modules.layout import admin_appearance_ui, apply_appearance_styles
from modules.seats import compute_distribution_from_excel, compute_distribution_variants
from modules.emailer import send_reservation_email
from modules.rooms import generate_time_slots, check_room_conflict
from modules.zones import generate_colored_plan, load_zones, save_zones

from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 3) CONSTANTES / DIRS
# ---------------------------------------------------------
ORDER_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

PLANOS_DIR = Path("modules/planos")
DATA_DIR = Path("data")
COLORED_DIR = Path("planos_coloreados")

for d in (PLANOS_DIR, DATA_DIR, COLORED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 4) SESSION STATE UI
# ---------------------------------------------------------
st.session_state.setdefault("ui", {
    "app_title": "Gestor de Puestos y Salas",
    "bg_color": "#ffffff",
    "logo_path": "assets/logo.png",
    "title_font_size": 64,
    "logo_width": 420,
})

# Inicio = Administrador (pantalla principal)
st.session_state.setdefault("screen", "Administrador")
st.session_state.setdefault("forgot_mode", False)

# ✅ sesión admin
st.session_state.setdefault("is_admin", False)

# ---------------------------------------------------------
# 4.5) DB + SETTINGS
# ---------------------------------------------------------
def _conn_debug_panel(err: Exception):
    st.error("❌ No pude conectar a Google Sheets.")
    st.write("Checklist rápida:")
    st.write("- En Streamlit Cloud → App → Settings → Secrets, existe **[sheets] sheet_name**")
    st.write("- El Google Sheet está **compartido** con el service account (client_email)")
    st.write("- El sheet_name coincide exacto con el nombre del archivo en Drive (o usa el ID)")
    st.divider()
    st.write("Error capturado:")
    st.exception(err)

try:
    conn = get_conn()
except Exception as e:
    _conn_debug_panel(e)
    st.stop()

if conn is None:
    st.error("❌ get_conn() devolvió None (conexión inválida).")
    st.stop()

if "db_initialized" not in st.session_state:
    with st.spinner("Conectando a Google Sheets..."):
        init_db(conn)
    st.session_state["db_initialized"] = True

apply_appearance_styles(conn)

settings = get_all_settings(conn) or {}
st.session_state["ui"]["app_title"] = settings.get("site_title", st.session_state["ui"]["app_title"])
st.session_state["ui"]["logo_path"] = settings.get("logo_path", st.session_state["ui"]["logo_path"])

# ---------------------------------------------------------
# 5) CSS
# ---------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
  background: {st.session_state.ui["bg_color"]};
}}
header {{
  visibility: hidden;
  height: 0px;
}}

div[data-testid="stAppViewContainer"] > .main {{
  padding-top: 0rem !important;
}}
section.main > div {{
  padding-top: 0rem !important;
}}

.block-container {{
  max-width: 100% !important;
  padding-top: 0.75rem !important;
  padding-left: 5cm !important;
  padding-right: 5cm !important;
}}

.mk-content {{
  width: 100%;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}}

html, body, [class*="css"] {{
  font-size: 20px !important;
}}
h1 {{ font-size: 48px !important; }}
h2 {{ font-size: 40px !important; }}
h3 {{ font-size: 32px !important; }}
p, li, label, span {{ font-size: 20px !important; }}

div[data-baseweb="input"] input {{
  font-size: 20px !important;
  padding-top: 14px !important;
  padding-bottom: 14px !important;
}}

div[data-baseweb="select"] > div {{
  font-size: 20px !important;
  min-height: 56px !important;
  border-radius: 18px !important;
}}

.stButton button {{
  font-size: 20px !important;
  font-weight: 900 !important;
  padding: 12px 18px !important;
  border-radius: 16px !important;
}}

.mk-title {{
  text-align: center;
  font-weight: 900;
  margin: 0;
  line-height: 1.05;
}}

/* mismo ancho para ambos botones del login */
button[kind="primary"][data-testid="baseButton-primary"] {{
  width: 320px !important;
}}
button[data-testid="baseButton-secondary"] {{
  width: 320px !important;
}}

/* ✅ Botón-logo invisible pero clickeable */
.mk-logo-btn button {{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}}
.mk-logo-btn button:hover {{
  filter: brightness(0.98);
}}
.mk-logo-btn button:focus {{
  outline: none !important;
}}

/* --------- mini UI "cuadros" para editor ---------- */
.mk-box {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 12px;
  background: rgba(255,255,255,0.85);
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}}
.mk-box h4 {{
  margin: 0 0 10px 0;
  font-size: 18px !important;
  font-weight: 900;
}}
.mk-muted {{
  opacity: 0.75;
  font-size: 16px !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def clean_pdf_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\r", "")
           .replace("\t", " ")
           .replace("–", "-")
           .replace("—", "-")
           .replace("−", "-")
           .replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("‘", "'")
           .replace("•", "-")
           .replace("\u00a0", " "))
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s

def go(screen: str):
    st.session_state["screen"] = screen

def _safe_sheet_lookup(sheets: dict, want: list[str]) -> Optional[pd.DataFrame]:
    """Busca una hoja por nombres posibles, case-insensitive, con contains."""
    if not sheets:
        return None
    norm = {str(k).strip().lower(): k for k in sheets.keys()}
    for w in want:
        w0 = w.strip().lower()
        if w0 in norm:
            return sheets[norm[w0]]
    for w in want:
        w0 = w.strip().lower()
        hit = next((orig for low, orig in norm.items() if w0 in low), None)
        if hit:
            return sheets[hit]
    return None

def _piso_to_label(piso_any) -> str:
    """
    Tu seats devuelve piso como string numérico "1".
    Tu app/DB suele usar "Piso 1".
    """
    if piso_any is None:
        return "Piso 1"
    s = str(piso_any).strip()
    if not s:
        return "Piso 1"
    if s.lower().startswith("piso"):
        return s
    nums = re.findall(r"\d+", s)
    return f"Piso {nums[0]}" if nums else f"Piso {s}"

def admin_logout():
    st.session_state["is_admin"] = False
    st.session_state["forgot_mode"] = False
    go("Administrador")
    st.rerun()

def _round_half_up(x: float) -> int:
    """4.5->5, 4.4->4"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(np.floor(float(x) + 0.5))

def _list_plan_images() -> list[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"]
    imgs: list[Path] = []
    for pat in patterns:
        imgs.extend(sorted(PLANOS_DIR.glob(pat)))
    # sacamos duplicados por name si hay
    seen = set()
    out = []
    for p in imgs:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    return out

def _pick_floor_image(piso_label: str) -> Optional[Path]:
    """
    FIX: soporta piso1.png (sin word-boundary).
    """
    imgs = _list_plan_images()
    if not imgs:
        return None

    m = re.findall(r"\d+", str(piso_label or ""))
    piso_num = m[0] if m else None
    if piso_num:
        # token por no-dígitos o inicio/fin
        token_re = re.compile(rf"(^|[^0-9]){re.escape(piso_num)}([^0-9]|$)")
        hit = next((p for p in imgs if token_re.search(p.stem)), None)
        if hit:
            return hit
        # fallback substring
        hit2 = next((p for p in imgs if piso_num in p.stem), None)
        if hit2:
            return hit2

    return imgs[0]

def _ensure_canvas_state():
    st.session_state.setdefault("zone_editor", {
        "shape": "rect",
        "fill": "rgba(255, 99, 71, 0.25)",  # transparente por defecto
        "stroke": "rgba(30,30,30,0.55)",
        "stroke_width": 2,
        "show_title": True,
        "title_text": "",
        "title_size": 28,
        "title_font": "DejaVuSans",
        "undo_stack": [],
        "redo_stack": [],
        "committed_json": None,
    })

def _push_undo(current_json):
    ze = st.session_state["zone_editor"]
    if current_json is not None:
        ze["undo_stack"].append(current_json)
        ze["redo_stack"] = ze.get("redo_stack", [])

def _pop_undo():
    ze = st.session_state["zone_editor"]
    if not ze.get("undo_stack"):
        return None
    last = ze["undo_stack"].pop()
    ze["redo_stack"].append(ze.get("committed_json"))
    return last

def _pop_redo():
    ze = st.session_state["zone_editor"]
    if not ze.get("redo_stack"):
        return None
    last = ze["redo_stack"].pop()
    ze["undo_stack"].append(ze.get("committed_json"))
    return last

def _save_canvas_outputs(piso_label: str, base_image_path: Optional[Path], canvas_json: dict, out_prefix: str, title_text: str):
    """
    Guarda:
      - PNG (plano con overlay)
      - PDF (simple: PNG dentro del PDF)
    """
    if base_image_path is None or not base_image_path.exists():
        raise RuntimeError("No hay imagen de plano para guardar.")

    # 1) Generar PNG con overlay usando tu módulo zones.generate_colored_plan
    try:
        out_img: Image.Image = generate_colored_plan(
            base_image_path=str(base_image_path),
            zones_json=canvas_json,
            title=title_text if title_text else None
        )
    except TypeError:
        out_img = generate_colored_plan(
            base_image_path=str(base_image_path),
            zones_json=canvas_json
        )

    COLORED_DIR.mkdir(parents=True, exist_ok=True)
    png_path = COLORED_DIR / f"{out_prefix}_{piso_label.replace(' ', '_')}.png"
    out_img.save(png_path)

    # 2) PDF básico con la imagen
    pdf = FPDF(unit="pt", format="A4")
    pdf.add_page()
    max_w = 540
    max_h = 780

    w, h = out_img.size
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    tmp_png = DATA_DIR / f"__tmp_{out_prefix}.png"
    # Pillow resample compat
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS
    out_img.resize((new_w, new_h), resample=resample).save(tmp_png)

    x = int((595 - new_w) / 2)
    y = 40
    pdf.image(str(tmp_png), x=x, y=y, w=new_w, h=new_h)

    pdf_path = DATA_DIR / f"{out_prefix}_{piso_label.replace(' ', '_')}.pdf"
    pdf.output(str(pdf_path))

    try:
        tmp_png.unlink(missing_ok=True)
    except Exception:
        pass

    return png_path, pdf_path

# ---------------------------------------------------------
# TOPBAR
# ---------------------------------------------------------
def render_topbar_and_menu():
    logo_path = Path(st.session_state.ui["logo_path"])
    size = int(st.session_state.ui.get("title_font_size", 64))
    title = st.session_state.ui.get("app_title", "Gestor de Puestos y Salas")
    logo_w = int(st.session_state.ui.get("logo_width", 420))

    c1, c2, c3 = st.columns([1.2, 3.6, 1.2], vertical_alignment="center")

    with c1:
        if logo_path.exists():
            st.markdown("<div class='mk-logo-btn'>", unsafe_allow_html=True)
            if st.button(" ", key="tb_logo_home_btn"):
                go("Administrador")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.image(str(logo_path), width=logo_w)
        else:
            if st.button("🧩 Inicio", key="tb_logo_home_fallback"):
                go("Administrador")
                st.rerun()

    with c2:
        st.markdown(
            f"<div class='mk-title' style='font-size:{size}px;'>{title}</div>",
            unsafe_allow_html=True
        )

    with c3:
        menu_choice = st.selectbox(
            "Menú",
            ["—", "Inicio", "Reservas", "Ver Distribución y Planos"],
            index=0,
            key="tb_top_menu_select",
        )
        if menu_choice == "Inicio":
            go("Administrador")
        elif menu_choice == "Reservas":
            go("Reservas")
        elif menu_choice == "Ver Distribución y Planos":
            go("Planos")

# ---------------------------------------------------------
# ADMIN (LOGIN + PANEL)
# ---------------------------------------------------------
def _validate_admin_login(email: str, password: str) -> bool:
    try:
        creds = get_admin_credentials(conn)
    except Exception:
        creds = None

    if not creds:
        return True

    e0, p0 = "", ""

    if isinstance(creds, dict):
        e0 = (creds.get("email") or creds.get("admin_email") or "").strip().lower()
        p0 = (creds.get("password") or creds.get("admin_password") or "").strip()
    elif isinstance(creds, (tuple, list)) and len(creds) >= 2:
        e0 = str(creds[0] or "").strip().lower()
        p0 = str(creds[1] or "").strip()
    else:
        return True

    if not e0 or not p0:
        return True

    return (email.strip().lower() == e0) and (password == p0)

def admin_panel(conn):
    st.subheader("Administrador")

    top = st.columns([1, 1], vertical_alignment="center")
    with top[0]:
        st.caption("Sesión de administrador activa.")
    with top[1]:
        _, b = st.columns([1, 1])
        with b:
            if st.button("Cerrar sesión", key="ap_btn_admin_logout", use_container_width=True):
                admin_logout()

    tabs = st.tabs(["Cargar Datos", "Editor de Planos"])

    # =====================================================
    # TAB 1: Cargar Datos
    # =====================================================
    with tabs[0]:
        st.markdown("### Cargar Excel y generar distribución")
        st.caption("Tu motor seats espera hojas tipo: Equipos, Parámetros y Capacidades (nombres pueden variar).")

        up = st.file_uploader("Subir archivo Excel", type=["xlsx", "xls"], key="ap_admin_excel_upload")

        colA, colB = st.columns([1, 1], vertical_alignment="center")
        with colA:
            cupos_reserva = st.number_input(
                "Cupos libres (reserva diaria)",
                min_value=0, max_value=50, value=2, step=1,
                key="ap_cupos_reserva"
            )
        with colB:
            ignore_params = st.toggle(
                "Ignorar parámetros (solo reparto proporcional)",
                value=False,
                key="ap_ignore_params"
            )

        st.session_state.setdefault("regen_counter", 0)
        st.session_state.setdefault("variant_seed", 42)

        st.session_state.setdefault("pending_distribution_rows", [])
        st.session_state.setdefault("pending_distribution_deficit", [])
        st.session_state.setdefault("pending_distribution_audit", {})
        st.session_state.setdefault("pending_distribution_score", {})

        def _run_generation(df_equipos, df_param, df_cap, seed_val: Optional[int]) -> bool:
            if df_equipos is None or df_equipos.empty:
                st.error("Falta hoja Equipos (o está vacía).")
                return False

            _df_param = df_param if df_param is not None else pd.DataFrame()
            _df_cap = df_cap if df_cap is not None else pd.DataFrame()

            if not bool(ignore_params):
                variants = compute_distribution_variants(
                    equipos_df=df_equipos,
                    parametros_df=_df_param,
                    df_capacidades=_df_cap,
                    cupos_reserva=int(cupos_reserva),
                    ignore_params=False,
                    n_variants=10,
                    variant_seed=int(seed_val or 42),
                    variant_mode="holgura",
                )
                best = variants[0] if variants else None
                if not best or not best.get("rows"):
                    st.error("No se generaron filas. Revisa que el Excel tenga columnas clave.")
                    return False
                rows = best["rows"]
                deficit_report = best.get("deficit_report", [])
                audit = best.get("audit", {})
                score_obj = best.get("score", {})
            else:
                rows, deficit_report, audit, score_obj = compute_distribution_from_excel(
                    equipos_df=df_equipos,
                    parametros_df=_df_param,
                    df_capacidades=_df_cap,
                    cupos_reserva=int(cupos_reserva),
                    ignore_params=True,
                    variant_seed=int(seed_val or 42),
                    variant_mode="holgura",
                )
                if not rows:
                    st.error("No se generaron filas (rows vacías). Revisa que el Excel tenga columnas clave.")
                    return False

            st.session_state["pending_distribution_rows"] = rows
            st.session_state["pending_distribution_deficit"] = deficit_report
            st.session_state["pending_distribution_audit"] = audit
            st.session_state["pending_distribution_score"] = score_obj
            return True

        if up is not None:
            try:
                xls = pd.ExcelFile(up)
                sheets = {name: xls.parse(name) for name in xls.sheet_names}

                st.success(f"✅ Archivo leído. Hojas: {', '.join(list(sheets.keys()))}")

                df_equipos = _safe_sheet_lookup(sheets, ["equipos", "equipo"])
                df_param = _safe_sheet_lookup(sheets, ["parametros", "parámetros", "parametro", "parámetro"])
                df_cap = _safe_sheet_lookup(sheets, ["capacidades", "capacidad", "aforo", "cupos"])

                b1, b2, b3 = st.columns([1, 1, 1], vertical_alignment="center")
                gen = b1.button("Generar distribución", type="primary", key="ap_btn_gen_dist")
                regen = b2.button("Regenerar", key="ap_btn_regen_dist")
                save_btn = b3.button("Guardar Distribución", key="ap_btn_save_dist")

                if gen:
                    st.session_state["regen_counter"] = 0
                    seed_val = int(st.session_state.get("variant_seed", 42)) + int(st.session_state["regen_counter"])
                    ok = _run_generation(df_equipos, df_param, df_cap, seed_val=seed_val)
                    if ok:
                        st.rerun()

                if regen:
                    st.session_state["regen_counter"] = int(st.session_state.get("regen_counter", 0)) + 1
                    seed_val = int(st.session_state.get("variant_seed", 42)) + int(st.session_state["regen_counter"])
                    ok = _run_generation(df_equipos, df_param, df_cap, seed_val=seed_val)
                    if ok:
                        st.rerun()

                if save_btn:
                    rows = st.session_state.get("pending_distribution_rows", [])
                    if not rows:
                        st.warning("Primero genera una distribución para poder guardarla.")
                    else:
                        try:
                            rows = st.session_state.get("pending_distribution_rows", [])
                            if not rows:
                                st.warning("Primero genera una distribución para poder guardarla.")
                            else:
                                # Opción A: guardar directo (tu database.py espera lista de dicts)
                                insert_distribution(conn, rows)

                                st.success("✅ Distribución guardada en Google Sheets (DB).")
                                st.session_state["last_distribution_rows"] = rows
                                st.session_state["last_distribution_deficit"] = st.session_state.get("pending_distribution_deficit", [])
                                st.session_state["last_distribution_audit"] = st.session_state.get("pending_distribution_audit", {})
                                st.session_state["last_distribution_score"] = st.session_state.get("pending_distribution_score", {})
                        except Exception as e:
                            st.error(f"No pude guardar en DB: {e}")
                            return

            except Exception as e:
                st.error(f"No se pudo leer el Excel: {e}")

    # =====================================================
    # TAB 2: Editor de Planos (tu contenido actual)
    # =====================================================
    with tabs[1]:
        st.markdown("### Editor de Planos por Piso")
        st.caption("Elige piso → equipo → día para ver cupos. Dibuja zonas sobre el plano y guarda en PNG/PDF.")

        _ensure_canvas_state()

        left, right = st.columns([1.1, 2.2], vertical_alignment="top")

        with left:
            pisos_opts = ["Piso 1", "Piso 2", "Piso 3"]
            sel_piso = st.selectbox("Selecciona Piso", pisos_opts, key="zp_sel_piso")

            rows_src = st.session_state.get("pending_distribution_rows") or st.session_state.get("last_distribution_rows")
            if not rows_src:
                df_db = read_distribution_df(conn)
                if df_db is not None and not df_db.empty:
                    rows_src = df_db.to_dict("records")

            teams = []
            if rows_src:
                df_r = pd.DataFrame(rows_src)
                if "piso" in df_r.columns:
                    df_r["__piso_label"] = df_r["piso"].apply(_piso_to_label)
                elif "Piso" in df_r.columns:
                    df_r["__piso_label"] = df_r["Piso"].apply(_piso_to_label)
                else:
                    df_r["__piso_label"] = ""

                eq_col = "equipo" if "equipo" in df_r.columns else ("Equipo" if "Equipo" in df_r.columns else None)
                if eq_col:
                    df_r[eq_col] = df_r[eq_col].astype(str).str.strip()
                    df_r = df_r[df_r[eq_col].str.lower() != "cupos libres"]
                    teams = sorted(df_r[df_r["__piso_label"] == sel_piso][eq_col].dropna().unique().tolist())

            if not teams:
                st.info("No hay equipos para este piso todavía (genera una distribución primero).")
                sel_team = st.selectbox("Equipo", ["—"], key="zp_sel_team_disabled")
            else:
                sel_team = st.selectbox("Equipo", teams, key="zp_sel_team")

            dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
            sel_dia = st.selectbox("Día", dias, key="zp_sel_dia")

            cupos_msg = None
            if rows_src and teams and sel_team and sel_team != "—":
                df_r = pd.DataFrame(rows_src)
                pcol = "piso" if "piso" in df_r.columns else ("Piso" if "Piso" in df_r.columns else None)
                ecol = "equipo" if "equipo" in df_r.columns else ("Equipo" if "Equipo" in df_r.columns else None)
                dcol = "dia" if "dia" in df_r.columns else ("Día" if "Día" in df_r.columns else None)
                ccol = "cupos" if "cupos" in df_r.columns else ("Cupos" if "Cupos" in df_r.columns else None)

                if pcol and ecol and dcol and ccol:
                    df_r["__piso_label"] = df_r[pcol].apply(_piso_to_label)
                    df_r[ecol] = df_r[ecol].astype(str).str.strip()
                    df_r[dcol] = df_r[dcol].astype(str).str.strip()
                    hit = df_r[(df_r["__piso_label"] == sel_piso) & (df_r[ecol] == sel_team) & (df_r[dcol] == sel_dia)]
                    if not hit.empty:
                        cupos_val = int(pd.to_numeric(hit.iloc[0][ccol], errors="coerce") or 0)
                        cupos_msg = cupos_val

            if cupos_msg is not None:
                st.caption(f"✅ Cupos asignados a **{sel_team}** el **{sel_dia}**: **{cupos_msg}**")
            else:
                st.caption("Selecciona piso/equipo/día para ver los cupos asignados.")

            st.divider()

            if st.button("Guardar todo", type="primary", key="zp_btn_save_all", use_container_width=True):
                try:
                    ze = st.session_state["zone_editor"]
                    base_img = _pick_floor_image(sel_piso)
                    if ze.get("committed_json") is None:
                        st.warning("Primero dibuja y presiona **Guardar zona** (en el panel derecho).")
                    else:
                        out_prefix = f"plano_editado_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        title_text = ze.get("title_text", "") if ze.get("show_title") else ""
                        png_path, pdf_path = _save_canvas_outputs(
                            piso_label=sel_piso,
                            base_image_path=base_img,
                            canvas_json=ze["committed_json"],
                            out_prefix=out_prefix,
                            title_text=title_text,
                        )
                        st.success("✅ Guardado listo (PNG + PDF).")
                        st.download_button("⬇️ Descargar PNG", data=png_path.read_bytes(), file_name=png_path.name, mime="image/png", use_container_width=True)
                        st.download_button("⬇️ Descargar PDF", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf", use_container_width=True)
                except Exception as e:
                    st.error(f"No pude guardar: {e}")

        with right:
            ze = st.session_state["zone_editor"]

            box1, box2, box3 = st.columns([1, 1, 1], vertical_alignment="top")

            with box1:
                st.markdown("<div class='mk-box'>", unsafe_allow_html=True)
                st.markdown("<h4>Formas</h4>", unsafe_allow_html=True)
                shape_label = st.selectbox(
                    "Tipo",
                    ["Rectángulo", "Círculo", "Triángulo", "Cuadrado"],
                    index=0,
                    key="zp_shape_select",
                    label_visibility="collapsed"
                )
                if shape_label == "Rectángulo":
                    ze["shape"] = "rect"
                elif shape_label == "Cuadrado":
                    ze["shape"] = "rect"
                elif shape_label == "Círculo":
                    ze["shape"] = "circle"
                else:
                    ze["shape"] = "triangle"
                st.markdown("<div class='mk-muted'>El “Cuadrado” se dibuja como rectángulo.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with box2:
                st.markdown("<div class='mk-box'>", unsafe_allow_html=True)
                st.markdown("<h4>Colores</h4>", unsafe_allow_html=True)
                palette = [
                    ("Rojo", "rgba(255, 59, 48, 0.25)"),
                    ("Naranjo", "rgba(255, 149, 0, 0.25)"),
                    ("Amarillo", "rgba(255, 204, 0, 0.25)"),
                    ("Verde", "rgba(52, 199, 89, 0.25)"),
                    ("Azul", "rgba(0, 122, 255, 0.25)"),
                    ("Morado", "rgba(175, 82, 222, 0.25)"),
                    ("Gris", "rgba(142, 142, 147, 0.25)"),
                    ("Negro", "rgba(0, 0, 0, 0.20)"),
                ]
                color_label = st.selectbox(
                    "Color",
                    [p[0] for p in palette],
                    index=0,
                    key="zp_color_select",
                    label_visibility="collapsed"
                )
                ze["fill"] = dict(palette).get(color_label, ze["fill"])
                st.markdown("<div class='mk-muted'>Relleno transparente para ver el plano atrás.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with box3:
                st.markdown("<div class='mk-box'>", unsafe_allow_html=True)
                st.markdown("<h4>Título</h4>", unsafe_allow_html=True)
                ze["show_title"] = st.toggle("Activar título", value=bool(ze.get("show_title", True)), key="zp_title_toggle")
                ze["title_text"] = st.text_input("Texto", value=str(ze.get("title_text", "")), key="zp_title_text")
                cA, cB = st.columns([1, 1])
                with cA:
                    ze["title_size"] = st.selectbox("Tamaño", [18, 22, 26, 28, 32, 36, 42], index=3, key="zp_title_size")
                with cB:
                    ze["title_font"] = st.selectbox("Fuente", ["DejaVuSans", "Helvetica", "Times"], index=0, key="zp_title_font")
                st.markdown("</div>", unsafe_allow_html=True)

            a1, a2, a3, a4 = st.columns([1, 1, 1, 1], vertical_alignment="center")
            if a1.button("Deshacer", key="zp_btn_undo", use_container_width=True):
                prev = _pop_undo()
                if prev is not None:
                    ze["committed_json"] = prev
                    st.rerun()
            if a2.button("Rehacer", key="zp_btn_redo", use_container_width=True):
                nxt = _pop_redo()
                if nxt is not None:
                    ze["committed_json"] = nxt
                    st.rerun()
            if a3.button("Borrar todo", key="zp_btn_clear", use_container_width=True):
                _push_undo(ze.get("committed_json"))
                ze["committed_json"] = {"version": "4.4.0", "objects": []}
                st.rerun()
            save_zone = a4.button("Guardar zona", key="zp_btn_commit", type="primary", use_container_width=True)

            base_img_path = _pick_floor_image(st.session_state.get("zp_sel_piso", "Piso 1"))
            if base_img_path is None:
                st.warning("No hay planos en `modules/planos`. Sube imágenes (png/jpg) para poder editar.")
            else:
                try:
                    img = Image.open(base_img_path).convert("RGBA")
                except Exception as e:
                    st.error(f"No pude abrir el plano: {e}")
                    img = None

                if img is None:
                    st.stop()

                max_w = 1000
                orig_w, orig_h = img.size
                if not orig_w or not orig_h:
                    st.error("El plano tiene tamaño inválido.")
                    st.stop()

                scale = min(1.0, float(max_w) / float(orig_w))
                w = int(round(orig_w * scale))
                h = int(round(orig_h * scale))
                w = max(1, int(w))
                h = max(1, int(h))

                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.LANCZOS

                img_resized = img.resize((w, h), resample=resample)

                initial_drawing = ze.get("committed_json")
                if not isinstance(initial_drawing, dict):
                    initial_drawing = None
                else:
                    initial_drawing.setdefault("objects", [])
                    initial_drawing.setdefault("version", "4.4.0")

                drawing_mode = ze.get("shape", "rect")
                canvas_key = f"zp_canvas_{st.session_state.get('zp_sel_piso', 'Piso 1')}"

                canvas_res = st_canvas(
                    fill_color=str(ze.get("fill", "rgba(255, 99, 71, 0.25)")),
                    stroke_color=str(ze.get("stroke", "rgba(30,30,30,0.55)")),
                    stroke_width=int(ze.get("stroke_width", 2)),
                    background_image=img_resized,
                    update_streamlit=True,
                    height=int(h),
                    width=int(w),
                    drawing_mode=drawing_mode,
                    initial_drawing=initial_drawing,
                    key=canvas_key,
                )

                if save_zone:
                    try:
                        current = canvas_res.json_data if canvas_res is not None else None
                        if not current or not isinstance(current, dict):
                            st.warning("No hay nada para guardar todavía.")
                        else:
                            current.setdefault("objects", [])
                            current.setdefault("version", "4.4.0")

                            _push_undo(ze.get("committed_json"))
                            ze["committed_json"] = current
                            st.success("✅ Zona guardada (queda lista para Guardar todo).")
                            st.rerun()
                    except Exception as e:
                        st.error(f"No pude guardar zona: {e}")
# ---------------------------------------------------------
# 6) MAIN EXECUTION FLOW
# ---------------------------------------------------------

# 1. Renderizar la barra superior y el menú siempre
render_topbar_and_menu()

# 2. Decidir qué pantalla mostrar según el estado de la sesión
screen = st.session_state.get("screen", "Administrador")

if screen == "Administrador":
    # Lógica de Login: Si no es admin, muestra login. Si es admin, muestra el panel.
    if st.session_state["is_admin"]:
        admin_panel(conn)
    else:
        # --- PANTALLA DE LOGIN SIMPLE ---
        st.markdown("### Acceso Administrador")
        c_login = st.container()
        with c_login:
            l_email = st.text_input("Email", key="login_email")
            l_pass = st.text_input("Contraseña", type="password", key="login_pass")
            
            if st.button("Ingresar", type="primary"):
                if _validate_admin_login(l_email, l_pass):
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas o usuario no autorizado.")

elif screen == "Reservas":
    st.subheader("Gestión de Reservas")
    st.info("Aquí deberías llamar a tu función de reservas. Ej: reservas_panel(conn)")
    # reservas_panel(conn) # Descomentar cuando importes la función

elif screen == "Ver Distribución y Planos": 
    # (Nota: En tu menú usaste 'Ver Distribución y Planos', asegúrate que coincida con el if)
    st.subheader("Visualización de Planos")
    st.info("Aquí deberías llamar a tu función de planos. Ej: planos_viewer(conn)")
    # planos_viewer(conn) # Descomentar cuando importes la función

elif screen == "Planos": # Captura por si el menú envía "Planos"
    st.subheader("Planos")
    st.write("Vista de planos.")

else:
    st.warning(f"Pantalla no encontrada: {screen}")


