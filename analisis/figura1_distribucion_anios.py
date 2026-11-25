"""
figura1_distribucion_anios.py
Genera la figura 1 del artículo: distribución del puntaje global por prueba y año
(S1, S2 y Saber 11°), con los estudiantes que presentaron las tres pruebas.
Uso:  python figura1_distribucion_anios.py . salida.png     (desde la carpeta 03_Analisis)
Requiere: numpy, pandas, matplotlib, openpyxl.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA = sys.argv[1]
SALIDA = sys.argv[2] if len(sys.argv) > 2 else "fig1_distribucion_anios.png"

COLORES = ["#4C86C6", "#5B8C3E", "#D4A017"]
ETIQUETAS = ["Simulacro 1", "Simulacro 2", "Saber 11°"]
COLS = ["S1_GLOBAL", "S2_GLOBAL", "SB11_GLOBAL"]

datos = {}
for y in (2025, 2026):
    f = f"{RUTA}/MDT_Simulacros_Saber11_{y}_anonimizada.xlsx"
    d = pd.read_excel(f, sheet_name="MDT")
    a = pd.read_excel(f, sheet_name="ALERTAS")
    imp = a[a.Alerta.isin(["Simulacro 1 imputado", "Simulacro 2 imputado"])].ID_ESTUDIANTE
    datos[y] = d[~d.ID_ESTUDIANTE.isin(imp)]

rng = np.random.default_rng(2025)
fig, ejes = plt.subplots(1, 2, figsize=(10, 3.93), dpi=300, sharey=True)
for ax, y in zip(ejes, (2025, 2026)):
    d = datos[y]
    valores = [d[c].dropna().values for c in COLS]
    bp = ax.boxplot(valores, positions=[1, 2, 3], widths=0.5, whis=(0, 100),
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="black", lw=1.2),
                    capprops=dict(color="black", lw=1.2))
    for caja, col in zip(bp["boxes"], COLORES):
        caja.set_facecolor(col); caja.set_alpha(0.4); caja.set_edgecolor(col)
    for i, (v, col) in enumerate(zip(valores, COLORES), start=1):
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, s=14, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        m = v.mean()
        ax.scatter([i], [m], marker="D", s=45, color="black", zorder=4)
        ax.text(i + 0.29, m, f"M = {m:.1f}", va="center", fontsize=8.5)
    ax.set_title(f"Año {y} (n = {len(d)})", fontsize=11)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(ETIQUETAS, fontsize=9.5)
    ax.set_xlim(0.5, 3.5)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
ejes[0].set_ylabel("Puntaje global (0–500)", fontsize=9.5)
fig.tight_layout()
fig.savefig(SALIDA, dpi=300)
print("Figura guardada en", SALIDA)
for y in (2025, 2026):
    print(y, len(datos[y]), [round(datos[y][c].mean(), 1) for c in COLS])
