from fpdf import FPDF
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
from PIL import Image
import os

from modules.zones import load_zones, generate_colored_plan, COLORED_DIR, PLANOS_DIR, ZONES_PATH, _hex_to_rgba

STATIC_DIR = Path("static")
PLANOS_DIR = Path("planos")
COLORED_DIR = Path("planos_coloreados")

def _save_plot_series(series, filename):
    plt.figure(figsize=(8,4))
    series.plot(kind="barh")
    plt.tight_layout()
    tmp = Path(tempfile.gettempdir()) / filename
    plt.savefig(tmp)
    plt.close()
    return tmp

def generate_pdf_from_df(df, out_path="distribucion_final.pdf", logo_path=STATIC_DIR/"logo.png"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # portada
    pdf.add_page()
    if logo_path.exists():
        try:
            pdf.image(str(logo_path), x=10, y=8, w=30)
        except: pass
    pdf.ln(25)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Distribución de puestos Casa Central", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6,
        "Este informe presenta la distribución diaria de puestos por equipo y piso.\n"
        "Nota: %Distrib = proporción de cupos de cada equipo respecto al total de cupos disponibles en su piso (por día)."
    )
    pdf.ln(6)

    # tabla resumida
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Distribución diaria (resumen)", ln=True)
    pdf.set_font("Arial", '', 9)

    headers = ["Piso","Equipo","Día","Cupos","%"]
    widths = [25, 70, 25, 20, 20]
    for i,h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C')
    pdf.ln()

    for _, row in df.iterrows():
        pdf.cell(widths[0], 6, str(row["piso"]), 1)
        pdf.cell(widths[1], 6, str(row["equipo"])[:45], 1)
        pdf.cell(widths[2], 6, str(row["dia"]), 1)
        pdf.cell(widths[3], 6, str(row["cupos"]), 1)
        pdf.cell(widths[4], 6, f"{row['pct']}%", 1)
        pdf.ln()

    # gráfico: % promedio por equipo (horizontal)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Distribución porcentual por equipo (promedio)", ln=True)
    df_team = df.groupby("equipo")["pct"].mean().sort_values(ascending=True)
    plot = _save_plot_series(df_team, "plot_team.png")
    pdf.image(str(plot), x=15, w=180)
    try:
        os.remove(plot)
    except: pass

    # gráfico: equilibrio semanal por equipo (stacked)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Equilibrio semanal por equipo (por día)", ln=True)
    df_week = df.groupby(["equipo","dia"])["cupos"].sum().unstack(fill_value=0)
    plot2 = _save_plot_series(df_week, "plot_week.png")
    pdf.image(str(plot2), x=15, w=180)
    try:
        os.remove(plot2)
    except: pass

    # Planos coloreados (si existen zonas)
    zones = load_zones()
    pisos = sorted(df["piso"].unique())
    for piso in pisos:
        piso_num = piso.replace("Piso ","").strip()
        colored = generate_colored_plan(piso)
        if colored:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Plano {piso}", ln=True)
            # mostrar plano coloreado
            try:
                pdf.image(str(colored), x=10, y=25, w=190)
            except Exception as e:
                try:
                    # fallback: insertar original si coloreado falla
                    orig = PLANOS_DIR / f"piso {piso_num}.png"
                    if orig.exists():
                        pdf.image(str(orig), x=10, y=25, w=190)
                except: pass
            # leyenda: listar zonas
            piso_zs = zones.get(piso, [])
            if piso_zs:
                pdf.ln(95)
                pdf.set_font("Arial", '', 10)
                pdf.cell(0,6, "Leyenda:", ln=True)
                for z in piso_zs:
                    team = z.get("team","")
                    color = z.get("color","#00A04A")
                    pdf.cell(0,5, f" - {team}  ({color})", ln=True)

    pdf.output(out_path)
    return out_path
