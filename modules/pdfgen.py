# pdfgen.py
from fpdf import FPDF
from pathlib import Path
import pandas as pd
from datetime import datetime

STATIC_DIR = Path("static")


def _fmt_pct(val, default="-"):
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    try:
        return f"{float(val):.1f}%"
    except Exception:
        return default


def _fmt_int(val, default="0"):
    try:
        return str(int(val))
    except Exception:
        return default


def _safe_str(val, default=""):
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    return str(val)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _add_footer(pdf: FPDF, text: str):
    """Pie de página para la página actual."""
    pdf.ln(4)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(0, 4, text)


def _table(
    pdf: FPDF,
    df: pd.DataFrame,
    headers: list,
    widths: list,
    align: list = None,
    row_height: int = 6,
    header_height: int = 7,
    max_rows_per_page: int = None,
):
    """
    Tabla simple con salto de página.
    - align: lista por columna (L/C/R). Default: L.
    - max_rows_per_page: si quieres forzar paginado (si None, usa auto_page_break).
    """
    if align is None:
        align = ["L"] * len(headers)

    pdf.set_font("Arial", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], header_height, str(h), 1, 0, "C")
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    row_count = 0
    for _, r in df.iterrows():
        if max_rows_per_page is not None and row_count >= max_rows_per_page:
            pdf.add_page()
            pdf.set_font("Arial", "B", 9)
            for i, h in enumerate(headers):
                pdf.cell(widths[i], header_height, str(h), 1, 0, "C")
            pdf.ln()
            pdf.set_font("Arial", "", 9)
            row_count = 0

        for i, col in enumerate(headers):
            val = r.get(col, "")
            txt = _safe_str(val)
            if i == 1:  # Equipo suele ser largo
                txt = txt[:60]
            pdf.cell(widths[i], row_height, txt, 1, 0, align[i])
        pdf.ln()
        row_count += 1


def generate_pdf_from_df(
    df: pd.DataFrame,
    deficit_report: pd.DataFrame | list | None = None,
    out_path: str = "informe_distribucion.pdf",
    logo_path: Path = STATIC_DIR / "logo.png",
    company_name: str = "Casa Central",
    issued_at: datetime | None = None,
):
    """
    Informe SIN planos (eso va en otro informe).

    Espera en df (salida de Seats):
      - piso, equipo, dia, cupos
      - % uso diario (para la tabla diaria)
      - % uso semanal (para el resumen semanal)
      - (más adelante tú me pasas Seats con dotación incluida en df; por ahora NO es requisito)

    deficit_report (salida de Seats):
      - lista[dict] o DataFrame con: piso, equipo, dia, dotacion, minimo, asignado, deficit, causa, etc.

    Reglas:
      - “Cupos libres” NO aparece en el PDF (aunque venga como fila en df/deficit).
      - Incluye glosarios a pie de página para explicar cálculos.
      - Incluye hoja final SOLO con detalle de déficit por equipo y día (sin hoja de resumen por equipo).
      - Incluye fecha de emisión + logo si existe.
    """
    if df is None or df.empty:
        raise ValueError("DF vacío: no hay datos para generar PDF")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    issued_at = issued_at or datetime.now()

    df = _normalize_cols(df)
    required = {"piso", "equipo", "dia", "cupos"}
    missing = [c for c in required if c not in set(df.columns)]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df: {missing}")

    col_uso_diario = "% uso diario" if "% uso diario" in df.columns else None
    col_uso_semanal = "% uso semanal" if "% uso semanal" in df.columns else None

    # Excluir cupos libres de todo lo que se muestra
    df_show = df[df["equipo"].astype(str).str.strip().str.lower() != "cupos libres"].copy()

    # Ordenar días
    dias_orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    df_show["dia"] = pd.Categorical(df_show["dia"], categories=dias_orden, ordered=True)
    df_show = df_show.sort_values(["piso", "equipo", "dia"])

    # Normalizar deficit_report a DataFrame
    if deficit_report is None:
        df_def = pd.DataFrame()
    elif isinstance(deficit_report, list):
        df_def = pd.DataFrame(deficit_report)
    else:
        df_def = deficit_report.copy()

    if not df_def.empty:
        df_def.columns = [str(c).strip().lower() for c in df_def.columns]
        if "equipo" in df_def.columns:
            df_def = df_def[df_def["equipo"].astype(str).str.strip().str.lower() != "cupos libres"].copy()

    # ------------------------------------------------------------
    # Resumen semanal por equipo (desde df: cupos + % uso semanal)
    # ------------------------------------------------------------
    weekly_cupos = df_show.groupby("equipo")["cupos"].sum().to_dict()
    weekly_usage = {}
    if col_uso_semanal:
        weekly_usage = df_show.groupby("equipo")[col_uso_semanal].first().to_dict()

    weekly_rows = []
    for eq in sorted(weekly_cupos.keys(), key=lambda x: str(x).lower()):
        wk = int(weekly_cupos.get(eq, 0))
        avg_day = wk / 5.0
        uso = weekly_usage.get(eq, None) if col_uso_semanal else None
        weekly_rows.append(
            {
                "Equipo": eq,
                "Cupos semana": str(wk),
                "Prom. cupos/día": f"{avg_day:.1f}",
                "% uso semanal": _fmt_pct(uso) if uso is not None else "-",
            }
        )
    df_weekly = pd.DataFrame(weekly_rows)

    # ------------------------------------------------------------
    # HOJA 1: Portada
    # ------------------------------------------------------------
    pdf.add_page()
    if logo_path and Path(logo_path).exists():
        try:
            pdf.image(str(logo_path), x=10, y=8, w=30)
        except Exception:
            pass

    pdf.ln(20)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Informe de Distribución de Puestos - {company_name}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Fecha de emisión: {issued_at.strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")

    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(
        0,
        6,
        "Este informe resume la asignación de cupos por día y por equipo, y un cierre semanal de uso.\n"
        "Nota: La reserva diaria ('Cupos libres') no se muestra en tablas ni en resúmenes.",
    )

    # ------------------------------------------------------------
    # HOJA 2: Distribución diaria + % uso diario (tabla)
    # ------------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Distribución diaria (por piso/equipo/día)", ln=True)
    pdf.ln(1)

    base_cols = ["piso", "equipo", "dia", "cupos"]
    headers = ["Piso", "Equipo", "Día", "Cupos"]
    widths = [15, 85, 22, 18]
    align = ["L", "L", "L", "R"]

    if col_uso_diario:
        base_cols.append(col_uso_diario)
        headers.append("% uso diario")
        widths.append(25)
        align.append("R")

    df_daily_table = df_show[base_cols].copy()
    rename_map = {c: h for c, h in zip(base_cols, headers)}
    df_daily_table = df_daily_table.rename(columns=rename_map)

    df_daily_table["Cupos"] = df_daily_table["Cupos"].apply(_fmt_int)
    if "% uso diario" in df_daily_table.columns:
        df_daily_table["% uso diario"] = df_daily_table["% uso diario"].apply(lambda v: _fmt_pct(v))

    _table(pdf, df_daily_table, headers=headers, widths=widths, align=align, max_rows_per_page=35)

    _add_footer(
        pdf,
        "Glosario:\n"
        "- Cupos: puestos asignados al equipo ese día, usando capacidad usable del piso (capacidad total menos reserva diaria).\n"
        "- % uso diario: (cupos del equipo ese día / capacidad usable total del piso ese día) * 100.\n"
        "  (El valor viene calculado desde Seats; la reserva diaria no se considera en el denominador.)"
    )

    # ------------------------------------------------------------
    # HOJA 3: Resumen semanal (tabla)
    # ------------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Resumen semanal por equipo", ln=True)
    pdf.ln(1)

    headers_w = ["Equipo", "Cupos semana", "Prom. cupos/día", "% uso semanal"]
    widths_w = [80, 30, 35, 35]
    align_w = ["L", "R", "R", "R"]

    _table(pdf, df_weekly, headers=headers_w, widths=widths_w, align=align_w, max_rows_per_page=40)

    _add_footer(
        pdf,
        "Glosario:\n"
        "- Cupos semana: suma de cupos asignados a ese equipo de Lunes a Viernes.\n"
        "- Prom. cupos/día: (cupos semana / 5).\n"
        "- % uso semanal: (cupos semana / (dotación * 5)) * 100.\n"
        "  (La dotación la entrega Seats; la reserva diaria no se considera.)"
    )

    # ------------------------------------------------------------
    # HOJA FINAL: Déficit (solo detalle por equipo y día)
    # ------------------------------------------------------------
    if not df_def.empty and {"equipo", "dia", "deficit"}.issubset(set(df_def.columns)):
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Reporte de déficit (detalle por equipo y día)", ln=True)
        pdf.ln(1)

        # Selección de columnas, tolerante
        colmap = []
        if "piso" in df_def.columns:
            colmap.append(("piso", "Piso"))
        colmap += [("equipo", "Equipo"), ("dia", "Día")]
        if "dotacion" in df_def.columns:
            colmap.append(("dotacion", "Dotación"))
        if "asignado" in df_def.columns:
            colmap.append(("asignado", "Asignado"))
        colmap.append(("deficit", "Déficit"))
        if "causa" in df_def.columns:
            colmap.append(("causa", "Causa"))

        df_det = df_def[[c for c, _ in colmap]].copy()
        df_det = df_det.rename(columns={c: h for c, h in colmap})

        # Formateo numérico
        for c in ["Dotación", "Asignado", "Déficit"]:
            if c in df_det.columns:
                df_det[c] = df_det[c].apply(_fmt_int)

        # Ordenar (si hay piso)
        sort_cols = [c for c in ["Piso", "Equipo", "Día"] if c in df_det.columns]
        if sort_cols:
            df_det = df_det.sort_values(sort_cols)

        # widths heurísticos
        widths_d = []
        align_d = []
        for h in df_det.columns:
            if h == "Piso":
                widths_d.append(14); align_d.append("L")
            elif h == "Día":
                widths_d.append(18); align_d.append("L")
            elif h in ("Dotación", "Asignado", "Déficit"):
                widths_d.append(18); align_d.append("R")
            elif h == "Causa":
                widths_d.append(80); align_d.append("L")
            else:  # Equipo
                widths_d.append(55); align_d.append("L")

        _table(pdf, df_det, headers=list(df_det.columns), widths=widths_d, align=align_d, max_rows_per_page=28)

        _add_footer(
            pdf,
            "Glosario:\n"
            "- Déficit: cupos faltantes reportados por Seats cuando las restricciones/capacidad no alcanzan."
        )

    # Export
    pdf.output(out_path)
    return out_path
