# pdfgen.py
from fpdf import FPDF
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
from datetime import datetime

STATIC_DIR = Path("static")


def _tmp_png_path(filename: str) -> Path:
    return Path(tempfile.gettempdir()) / filename


def _save_barh(series_or_df, filename: str, title: str = "") -> Path:
    """
    Guarda un gráfico horizontal. Acepta:
      - Series: barh simple
      - DataFrame: stacked barh
    """
    plt.figure(figsize=(8, 4))
    ax = None
    if isinstance(series_or_df, pd.Series):
        ax = series_or_df.plot(kind="barh")
    else:
        ax = series_or_df.plot(kind="barh", stacked=True)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    tmp = _tmp_png_path(filename)
    plt.savefig(tmp)
    plt.close()
    return tmp


def _fmt_pct(x, decimals=1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return ""


def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        if float(x).is_integer():
            return str(int(float(x)))
        return str(x)
    except Exception:
        return str(x)


class ReportPDF(FPDF):
    def __init__(self, issued_at_str: str, logo_path: Path | None = None):
        super().__init__()
        self.issued_at_str = issued_at_str
        self.logo_path = logo_path

    def header(self):
        # Logo (izquierda)
        if self.logo_path and self.logo_path.exists():
            try:
                self.image(str(self.logo_path), x=10, y=8, w=22)
            except Exception:
                pass

        # Fecha emisión (derecha, estilo encabezado)
        self.set_font("Arial", "", 9)
        self.set_xy(10, 10)
        self.cell(0, 6, f"Emisión: {self.issued_at_str}", ln=0, align="R")

        # Separador
        self.ln(16)
        self.set_draw_color(210, 210, 210)
        self.line(10, 26, 200, 26)
        self.ln(6)

    def footer(self):
        # Número de página "X de N"
        self.set_y(-12)
        self.set_draw_color(230, 230, 230)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font("Arial", "", 9)
        page = self.page_no()
        total = getattr(self, "alias_nb_pages_value", None)
        if total is None:
            self.cell(0, 8, f"{page}", align="C")
        else:
            self.cell(0, 8, f"{page} de {total}", align="C")


def _add_section_title(pdf: FPDF, title: str, subtitle: str | None = None):
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, title, ln=True)
    if subtitle:
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, subtitle)
    pdf.ln(2)


def _add_glossary_box(pdf: FPDF, lines: list[str]):
    """
    Caja tipo glosario al final de la página (sin mencionar cupos libres).
    """
    pdf.ln(4)
    x = pdf.get_x()
    y = pdf.get_y()
    w = 190

    pdf.set_fill_color(248, 248, 248)
    pdf.set_draw_color(220, 220, 220)
    pdf.rect(x, y, w, 28, style="DF")

    pdf.set_xy(x + 3, y + 2)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 5, "Glosario", ln=True)

    pdf.set_font("Arial", "", 8.5)
    for ln in lines:
        pdf.multi_cell(w - 6, 4.2, f"• {ln}")

    # dejar cursor debajo
    pdf.set_xy(x, y + 30)


def _table(pdf: FPDF, headers: list[str], rows: list[list[str]], widths: list[int], aligns=None):
    aligns = aligns or ["L"] * len(headers)

    pdf.set_font("Arial", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, "C")
    pdf.ln()

    pdf.set_font("Arial", "", 8.8)
    for r in rows:
        for i, cell in enumerate(r):
            pdf.cell(widths[i], 6, cell, 1, 0, aligns[i])
        pdf.ln()


def generate_pdf_from_df(
    df: pd.DataFrame,
    deficit_report: list[dict] | None = None,
    out_path: str = "distribucion_final.pdf",
    logo_path: Path = STATIC_DIR / "logo.png",
    issued_at: datetime | None = None,
):
    """
    Genera un informe SIN planos.

    Páginas (pueden crecer según cantidad de filas):
      1) Portada
      2) Distribución diaria (tabla) + glosario (% uso diario + método Sainte-Laguë)
      3) Resumen semanal por equipo (tabla) + glosario (% uso semanal)
      4) (Opcional) Tablas de déficit (si hay registros) + glosario

    Reglas:
      - NO mostrar filas con equipo == "Cupos libres".
      - La tabla diaria usa "% uso diario".
      - El resumen semanal usa "% uso semanal" y dotación.
      - El déficit no muestra "causa".
      - Encabezado con fecha emisión + paginado "X de N".
    """
    issued_at = issued_at or datetime.now()
    issued_at_str = issued_at.strftime("%Y-%m-%d %H:%M")

    pdf = ReportPDF(issued_at_str=issued_at_str, logo_path=logo_path)
    pdf.alias_nb_pages()
    pdf.alias_nb_pages_value = "{nb}"
    pdf.set_auto_page_break(auto=True, margin=16)

    if df is None or df.empty:
        pdf.add_page()
        _add_section_title(pdf, "Distribución de puestos Casa Central")
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, "No hay datos para generar el informe.")
        pdf.output(out_path)
        return out_path

    df = df.copy()

    # Filtrar Cupos libres para el PDF (pero siguen existiendo en data)
    df = df[df["equipo"].astype(str).str.strip().str.lower() != "cupos libres"].copy()

    # Normalizar columnas esperadas
    # (si falta alguna, no explotamos: usamos vacío)
    for col in ["piso", "equipo", "dia", "cupos", "dotacion", "% uso diario", "% uso semanal"]:
        if col not in df.columns:
            df[col] = None

    # Asegurar tipos para orden
    df["piso"] = df["piso"].astype(str)
    df["equipo"] = df["equipo"].astype(str)
    df["dia"] = df["dia"].astype(str)

    # Orden fijo de días
    day_order = {"Lunes": 1, "Martes": 2, "Miércoles": 3, "Jueves": 4, "Viernes": 5}
    df["_day_order"] = df["dia"].map(day_order).fillna(99).astype(int)

    # -------------------------
    # Portada
    # -------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.ln(10)
    pdf.cell(0, 10, "Distribución de puestos", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Casa Central", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 10)
    pisos = ", ".join(sorted(df["piso"].unique()))
    pdf.multi_cell(
        0,
        6,
        f"Informe de asignación diaria de cupos por equipo y piso.\n"
        f"Pisos incluidos: {pisos}",
        align="C",
    )

    # -------------------------
    # Página: Distribución diaria
    # -------------------------
    pdf.add_page()
    _add_section_title(pdf, "Distribución diaria (detalle)")

    df_daily = df.sort_values(["piso", "_day_order", "equipo"]).copy()

    headers = ["Piso", "Equipo", "Día", "Cupos", "% Uso diario"]
    widths = [18, 84, 22, 18, 22]

    rows_out: list[list[str]] = []
    for _, r in df_daily.iterrows():
        rows_out.append([
            str(r["piso"]),
            str(r["equipo"])[:55],
            str(r["dia"]),
            _fmt_num(r["cupos"]),
            _fmt_pct(r["% uso diario"], decimals=2),
        ])

        # paginar si se llena mucho
        if pdf.get_y() > 260:
            _table(pdf, headers, rows_out, widths, aligns=["C", "L", "C", "R", "R"])
            rows_out = []
            pdf.add_page()
            _add_section_title(pdf, "Distribución diaria (continuación)")

    if rows_out:
        _table(pdf, headers, rows_out, widths, aligns=["C", "L", "C", "R", "R"])

    _add_glossary_box(pdf, [
        "Capacidad usable diaria (100%): Capacidad total del piso − Reserva diaria.",
        "% Uso diario = (Cupos asignados al equipo en el día / Capacidad usable diaria) × 100.",
        "La asignación se realiza con restricciones hard (día completo y mínimos, si aplican) y reparto proporcional con el método Sainte-Laguë sobre la demanda restante.",
        "Sainte-Laguë asigna cupos uno a uno, maximizando w/(2a+1), donde w es demanda restante y a es cupos ya asignados al equipo.",
    ])

    # -------------------------
    # Página: Resumen semanal por equipo (tabla)
    # -------------------------
    pdf.add_page()
    _add_section_title(pdf, "Resumen semanal por equipo")

    # Tomamos % uso semanal desde la data (viene repetido por fila equipo)
    # Construimos una tabla agregada por (piso,equipo).
    df_week = df.copy()
    df_week["cupos"] = pd.to_numeric(df_week["cupos"], errors="coerce").fillna(0).astype(int)
    df_week["dotacion"] = pd.to_numeric(df_week["dotacion"], errors="coerce").fillna(0).astype(int)
    df_week["% uso semanal"] = pd.to_numeric(df_week["% uso semanal"], errors="coerce")

    # Agregados:
    # - cupos semana total
    # - cupos promedio diario = cupos_semana/5
    # - dotación (tomamos max por seguridad)
    # - % uso semanal (tomamos max/mean; deben ser iguales por equipo)
    agg = (
        df_week.groupby(["piso", "equipo"], as_index=False)
        .agg({
            "cupos": "sum",
            "dotacion": "max",
            "% uso semanal": "max",
        })
    )
    agg["cupos_promedio_diario"] = (agg["cupos"] / 5.0).round(2)

    agg = agg.sort_values(["piso", "equipo"])

    headers2 = ["Piso", "Equipo", "Dotación", "Cupos/sem", "Prom/día", "% Uso semanal"]
    widths2 = [18, 72, 20, 20, 20, 25]

    rows2: list[list[str]] = []
    for _, r in agg.iterrows():
        rows2.append([
            str(r["piso"]),
            str(r["equipo"])[:45],
            _fmt_num(r["dotacion"]),
            _fmt_num(r["cupos"]),
            f'{float(r["cupos_promedio_diario"]):.2f}',
            _fmt_pct(r["% uso semanal"], decimals=2),
        ])

        if pdf.get_y() > 260:
            _table(pdf, headers2, rows2, widths2, aligns=["C", "L", "R", "R", "R", "R"])
            rows2 = []
            pdf.add_page()
            _add_section_title(pdf, "Resumen semanal por equipo (continuación)")

    if rows2:
        _table(pdf, headers2, rows2, widths2, aligns=["C", "L", "R", "R", "R", "R"])

    _add_glossary_box(pdf, [
        "% Uso semanal = (Cupos totales asignados al equipo en la semana / (Dotación del equipo × 5)) × 100.",
        "La semana considera lunes a viernes (5 días).",
    ])

    # -------------------------
    # Página: gráficos de % uso semanal (opcional pero útil)
    # -------------------------
    pdf.add_page()
    _add_section_title(pdf, "Uso semanal (%), vista gráfica")

    # Gráfico 1: % uso semanal promedio por equipo (global)
    team_usage = agg.groupby("equipo")["% uso semanal"].mean().sort_values(ascending=True)
    plot1 = _save_barh(team_usage, "plot_weekly_usage.png", title="% Uso semanal promedio por equipo")
    try:
        pdf.image(str(plot1), x=14, w=182)
    finally:
        try:
            os.remove(plot1)
        except Exception:
            pass

    pdf.ln(2)

    # Gráfico 2: cupos por día (stacked) por equipo (global)
    df_weekday = (
        df.groupby(["equipo", "dia"])["cupos"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], fill_value=0)
    )
    plot2 = _save_barh(df_weekday, "plot_weekly_balance.png", title="Cupos por día (suma semanal por equipo)")
    try:
        pdf.image(str(plot2), x=14, y=140, w=182)
    finally:
        try:
            os.remove(plot2)
        except Exception:
            pass

    _add_glossary_box(pdf, [
        "Los gráficos usan los mismos campos calculados por Seats (sin recalcular): cupos diarios y % uso semanal.",
        "% Uso semanal depende de dotación y cupos asignados acumulados en la semana.",
    ])

    # -------------------------
    # Página: déficit (si existe)
    # -------------------------
    deficit_report = deficit_report or []
    if len(deficit_report) > 0:
        pdf.add_page()
        _add_section_title(pdf, "Reporte de déficit (por día)")

        dfd = pd.DataFrame(deficit_report).copy()
        for col in ["piso", "equipo", "dia", "dotacion", "asignado", "deficit"]:
            if col not in dfd.columns:
                dfd[col] = None

        dfd["deficit"] = pd.to_numeric(dfd["deficit"], errors="coerce").fillna(0).astype(int)
        dfd = dfd[dfd["deficit"] > 0].copy()

        # Orden días
        dfd["_day_order"] = dfd["dia"].map(day_order).fillna(99).astype(int)
        dfd = dfd.sort_values(["piso", "_day_order", "equipo"])

        headers3 = ["Piso", "Equipo", "Día", "Dotación", "Asignado", "Déficit"]
        widths3 = [18, 72, 22, 20, 20, 18]

        rows3: list[list[str]] = []
        for _, r in dfd.iterrows():
            rows3.append([
                str(r["piso"]),
                str(r["equipo"])[:45],
                str(r["dia"]),
                _fmt_num(r["dotacion"]),
                _fmt_num(r["asignado"]),
                _fmt_num(r["deficit"]),
            ])

            if pdf.get_y() > 260:
                _table(pdf, headers3, rows3, widths3, aligns=["C", "L", "C", "R", "R", "R"])
                rows3 = []
                pdf.add_page()
                _add_section_title(pdf, "Reporte de déficit (continuación)")

        if rows3:
            _table(pdf, headers3, rows3, widths3, aligns=["C", "L", "C", "R", "R", "R"])

        _add_glossary_box(pdf, [
            "Déficit = max(0, Dotación del equipo − Cupos asignados al equipo ese día).",
            "Si existen restricciones hard (día completo/mínimos), pueden forzar asignaciones y/o generar déficit cuando la capacidad usable diaria no alcanza.",
        ])

    pdf.output(out_path)
    return out_path

