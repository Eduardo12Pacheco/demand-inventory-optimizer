"""Visor de Streamlit de solo lectura para el reporte de benchmark acotado congelado.

El visor renderiza exactamente un archivo de reporte relativo al repositorio
(``data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json``). Es solo
agregados: nunca se muestran filas crudas, vectores horarios, predicciones
individuales, fechas faltantes por producto ni series sintéticas, y la app no
realiza solicitudes de datos en vivo de ningún tipo (sin Hugging Face, sin
loaders, sin URLs, sin controles de carga/ruta).

Todos los números y declaraciones se derivan del reporte en tiempo de ejecución;
nada está hardcodeado. La capa de datos derivada vive en ``app/viewer_helpers.py``
(funciones puras verificables por las pruebas) y el renderizado en HTML/CSS/SVG
propio no agrega dependencias de gráficos (sin Plotly/Altair/Matplotlib). Si el
reporte falta o está malformado, la app falla cerrado con un error visible y
claro en lugar de sustituir valores por defecto que podrían falsear el benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# El visor se ejecuta con ``streamlit run app/streamlit_app.py``; se agrega el
# directorio de la app al path para importar su capa derivada local sin tocar
# las dependencias del paquete central.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_optimizer.demo.benchmark_report import (
    REPORT_RELATIVE_PATH,
    BenchmarkReport,
    BenchmarkReportError,
    load_report,
)

from viewer_helpers import (
    METRIC_KEYS,
    METRIC_LABELS,
    METRIC_ORDER,
    METHODOLOGY_STEPS,
    MODEL_PUBLIC_NAMES,
    comparison_data,
    comparison_equivalent_rows,
    comparison_min_values,
    conclusion_summary,
    fmt_count,
    fmt_metric,
    limitation_items,
    model_explanations,
    protocol_segments,
    public_fold_labels,
    stockout_facts,
)

EXACT_NO_WINNER_SENTENCE = "Este benchmark acotado no selecciona automáticamente un ganador."

# Traducciones al español de la capa de presentación para la prosa del propio
# reporte congelado (limitaciones y advertencias). El JSON nunca se modifica;
# cualquier texto del reporte no cubierto aquí falla cerrado abajo en lugar de
# mostrarse en inglés.
REPORT_TEXT_TRANSLATIONS = {
    "The evaluation is bounded to the streamed prefix, not a complete snapshot.": (
        "La evaluación está acotada al prefijo transmitido, no es una instantánea completa."
    ),
    "The input prefix may have incomplete product histories and missing calendar dates.": (
        "El prefijo de entrada puede tener historiales de producto incompletos y "
        "fechas de calendario faltantes."
    ),
    "Advanced forecasting and latent-demand recovery are not included.": (
        "No se incluyen pronósticos avanzados ni recuperación de demanda latente."
    ),
    "bounded_real_evaluation": "Modo de evaluación real acotada.",
    "This report covers at most the first 1,000 streamed train rows and does not represent the full dataset.": (
        "Este reporte cubre como máximo las primeras 1.000 filas de entrenamiento "
        "transmitidas y no representa el conjunto de datos completo."
    ),
    "Metrics evaluate observed_sales only; they are not latent or true demand metrics.": (
        "Las métricas evalúan únicamente las ventas observadas; no son métricas de demanda "
        "latente ni de demanda real."
    ),
    "Rows with stockout metadata are preserved; censoring may affect observed-sales error interpretation.": (
        "Las filas con metadatos de quiebre se conservan; la censura puede afectar "
        "la interpretación del error sobre ventas observadas."
    ),
    "No automatic best-model selection was performed.": (
        "No se realizó ninguna selección automática de mejor modelo."
    ),
}

_CSS = """
/* ---------------------------------------------------------------------------
   Visor de benchmark acotado — instrumento científico editorial.
   Sistema visual (ver DESIGN.md en la raíz del repositorio):
   superficies off-white cálidas, tinta, un único acento verde azulado apagado,
   datos en monoespaciada, reglas de trazo fino, sin degradados ni sombras
   decorativas. Tarjetas solo para hechos y conclusiones; el resto respira con
   hairlines y espacio. Movimiento mínimo y solo cuando explica estado.
   ------------------------------------------------------------------------- */
:root {
  --paper: #F6F4EE;
  --paper-2: #FBFAF6;
  --ink: #1E2622;
  --ink-2: #55605A;
  --line: #DCD7CA;
  --line-soft: #F0ECDF;
  --accent: #2E6B5E;
  --accent-soft: #E4ECE8;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace;
}
html { scroll-behavior: smooth; }
[data-testid="stAppViewContainer"] {
  background-color: var(--paper);
  color: var(--ink);
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0.35rem; padding-bottom: 2.6rem; max-width: 1180px; margin: 0 auto; }
#MainMenu { visibility: hidden; }

/* Tipografía: jerarquía fuerte, datos en mono. */
[data-testid="stAppViewContainer"] h1 {
  color: var(--ink); font-weight: 650; letter-spacing: -0.02em; line-height: 1.08;
  font-size: 1.4rem; margin: 0.08rem 0 0.05rem;
}
[data-testid="stAppViewContainer"] h2 {
  color: var(--ink); font-size: 0.88rem; font-weight: 650;
  text-transform: uppercase; letter-spacing: 0.08em;
  border-top: 1px solid var(--line); padding-top: 0.28rem; margin: 0.44rem 0 0.3rem;
}
p, li { color: var(--ink); line-height: 1.42; margin-bottom: 0.22rem; }
.mono { font-family: var(--mono); }

/* Navegación de anclas + salto de contenido. */
.skip-link {
  position: absolute; left: -9999px; top: 0; background: var(--accent); color: #FFFFFF;
  padding: 0.4rem 0.7rem; z-index: 40; font-size: 0.8rem; border-radius: 2px;
}
.skip-link:focus-visible { left: 0.5rem; top: 0.5rem; }
.viewer-nav {
  position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap;
  gap: 0.2rem 1.2rem; background: var(--paper); border-bottom: 1px solid var(--line);
  padding: 0.36rem 0 0.34rem; margin-bottom: 0.45rem;
}
.viewer-nav a { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; color: var(--ink-2); text-decoration: none; }
.viewer-nav a:hover { color: var(--accent); }
.anchor { display: block; height: 0; scroll-margin-top: 3.4rem; }

/* Resumen: chips, hechos, hallazgo, conclusiones y procedencia. */
.chips {
  display: inline-block; font-size: 0.68rem; font-weight: 650; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--ink-2); border: 1px solid var(--line);
  background: var(--paper-2); padding: 0.22rem 0.6rem; border-radius: 2px;
}
.subtitle { color: var(--ink-2); font-size: 0.9rem; margin: 0 0 0.36rem; }
.summary > * { animation: viewer-rise 0.45s ease-out both; }
.summary > *:nth-child(2) { animation-delay: 0.05s; }
.summary > *:nth-child(3) { animation-delay: 0.1s; }
.summary > *:nth-child(4) { animation-delay: 0.15s; }
@keyframes viewer-rise { from { opacity: 0; transform: translateY(6px); } }

.facts {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.5rem; margin: 0 0 0.42rem;
}
.fact {
  background: var(--paper-2); border: 1px solid var(--line); border-radius: 2px;
  padding: 0.38rem 0.65rem 0.42rem;
}
.fact-label {
  display: block; font-size: 0.62rem; font-weight: 650; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ink-2); margin-bottom: 0.12rem;
}
.fact-value {
  display: block; font-size: 1.06rem; font-weight: 600; color: var(--ink);
  font-family: var(--mono); letter-spacing: -0.01em;
}
.fact-value.mono { font-size: 0.9rem; font-weight: 600; }

.finding {
  background: var(--accent-soft); border: 1px solid #C6D8CF; border-radius: 2px;
  padding: 0.36rem 0.7rem; margin: 0 0 0.42rem;
  font-family: var(--mono); font-size: 0.8rem; font-weight: 600; color: var(--ink);
}

.conclusions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0 0 0.42rem; }
.conclusion-card {
  background: var(--paper-2); border: 1px solid var(--line); border-radius: 2px;
  padding: 0.42rem 0.6rem 0.46rem;
}
.conclusion-title {
  display: block; font-size: 0.62rem; font-weight: 650; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 0.14rem;
}
.conclusion-text { font-size: 0.76rem; line-height: 1.38; color: var(--ink); margin: 0; }

.provenance {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.3rem 1.6rem;
  border-top: 1px solid var(--line-soft); padding-top: 0.35rem; margin-bottom: 0.4rem;
}
.prov-item { display: flex; align-items: baseline; gap: 0.5rem; }
.prov-label {
  font-size: 0.6rem; font-weight: 650; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--ink-2); white-space: nowrap;
}
.prov-value { font-family: var(--mono); font-size: 0.68rem; color: var(--ink); word-break: break-all; }

/* Protocolo temporal: carril proporcional + leyenda + historial de ajuste. */
.proto {
  border: 1px solid var(--line); background: var(--paper-2);
  padding: 0.45rem 0.8rem 0.55rem; margin: 0.1rem 0 0.4rem;
}
.proto-fold { padding: 0.34rem 0; border-bottom: 1px solid var(--line-soft); }
.proto-fold:last-child { border-bottom: none; padding-bottom: 0.05rem; }
.proto-fold-head { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.26rem; }
.proto-fold-name {
  font-family: var(--mono); font-size: 0.72rem; font-weight: 650;
  color: var(--accent); white-space: nowrap;
}
.proto-lane { display: flex; flex: 1; height: 0.95rem; gap: 2px; }
.seg { display: block; height: 100%; border-radius: 1px; }
.seg-train { background: #4A554F; }
.seg-val { background: #B4BCB6; }
.seg-test { background: var(--paper-2); border: 1px dashed var(--accent); }
.proto-legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.26rem 0 0.28rem; }
.legend-col { display: flex; align-items: baseline; gap: 0.35rem; flex-wrap: wrap; }
.legend-name { font-size: 0.6rem; font-weight: 650; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink); }
.legend-line { font-family: var(--mono); font-size: 0.64rem; color: var(--ink); }
.fit-row { display: flex; align-items: baseline; gap: 0.55rem; flex-wrap: wrap; }
.fit-name {
  font-size: 0.62rem; font-weight: 650; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink-2); white-space: nowrap;
}
.fit-meta { font-size: 0.7rem; color: var(--ink); }
.fit-track { display: block; height: 0.36rem; background: var(--line-soft); margin: 0.22rem 0 0.05rem; border-radius: 1px; }
.fit-bar { display: block; height: 100%; background: var(--accent); border-radius: 1px; }
.hist-note {
  border: 1px solid var(--line); background: var(--paper);
  padding: 0.5rem 0.8rem 0.55rem; margin: 0.15rem 0 0.5rem;
}
.hist-note h3 {
  font-size: 0.66rem; font-weight: 650; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--ink-2); margin: 0 0 0.25rem;
}
.hist-note p { font-size: 0.78rem; color: var(--ink-2); line-height: 1.45; margin: 0; }

/* Comparación: selector de métrica + gráfico de barras propio (sin librerías). */
[data-testid="stSelectbox"] { margin: 0.04rem 0 0.06rem; }
[data-testid="stSelectbox"] label { color: var(--ink); font-weight: 650; font-size: 0.78rem; margin-bottom: 0.03rem; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--paper-2); border-color: #B9B3A3; border-radius: 2px; min-height: 26px;
}
.chart { margin: 0.15rem 0 0.35rem; animation: chart-in 0.45s ease-out both; }
@keyframes chart-in { from { opacity: 0; transform: translateY(5px); } }
.chart-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
.chart-col-head {
  display: flex; align-items: baseline; gap: 0.55rem; flex-wrap: wrap;
  padding: 0.26rem 0 0.32rem; border-bottom: 1px solid var(--line); margin-bottom: 0.24rem;
}
.chart-col-name { font-size: 0.78rem; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink); }
.chart-col-range { font-family: var(--mono); font-size: 0.68rem; color: var(--ink-2); }
.bar-row {
  display: grid; grid-template-columns: 9.2rem 1fr 3.5rem 6.2rem;
  align-items: center; gap: 0.5rem; padding: 0.14rem 0;
}
.bar-model { font-size: 0.78rem; color: var(--ink); font-weight: 500; }
.bar-track { display: block; height: 0.72rem; background: var(--line-soft); border-radius: 1px; }
.bar-fill { display: block; height: 100%; background: #C9C4B4; border-radius: 1px; animation: bar-grow 0.55s ease-out both; }
.bar-fill.is-min { background: var(--accent); }
@keyframes bar-grow { from { width: 0; } }
.bar-value { font-family: var(--mono); font-size: 0.78rem; color: var(--ink); text-align: right; }
.min-tag {
  display: inline-block; font-size: 0.6rem; font-weight: 650; letter-spacing: 0.06em;
  text-transform: uppercase; background: var(--accent); color: #FFFFFF;
  padding: 0.14rem 0.4rem; border-radius: 2px; text-align: center; white-space: nowrap;
}
.min-tag.is-empty { visibility: hidden; }

/* Tabla estática de agregados. */
[data-testid="stTable"] table {
  width: 100%; border-collapse: collapse; background: #FFFFFF;
  border: 1px solid var(--line); font-size: 0.8rem;
}
[data-testid="stTable"] th {
  background: var(--paper-2); color: var(--ink-2); text-transform: uppercase;
  font-size: 0.64rem; font-weight: 650; letter-spacing: 0.07em;
  border-bottom: 1px solid var(--line); padding: 0.38rem 0.55rem; text-align: left;
}
[data-testid="stTable"] td {
  border-bottom: 1px solid var(--line-soft); padding: 0.32rem 0.55rem; color: var(--ink);
  white-space: nowrap; font-family: var(--mono); font-size: 0.76rem;
}
[data-testid="stTable"] tr:nth-child(even) td { background: var(--paper-2); }
[data-testid="stTable"] tr:last-child td { border-bottom: none; }
[data-testid="stCaptionContainer"] { color: var(--ink-2); font-size: 0.76rem; }

/* Quiebres de stock: mosaicos + bandas de proporción derivadas. */
.bands { margin: 0.5rem 0 0.45rem; }
.band-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(10rem, 22rem) 3.2rem; align-items: center; gap: 0.7rem; padding: 0.26rem 0; }
.band-label { font-size: 0.76rem; color: var(--ink); }
.band-track { display: block; height: 0.66rem; background: var(--line-soft); border-radius: 1px; }
.band-fill { display: block; height: 100%; background: #C9C4B4; border-radius: 1px; }
.band-fill.is-min { background: var(--accent); }
.band-pct { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-2); text-align: right; }

/* Metodología: flujo + bloques de modelo. */
.flow {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.15rem;
  border: 1px solid var(--line); background: var(--paper-2);
  padding: 0.55rem 0.8rem; margin: 0.1rem 0 0.7rem;
}
.flow-step { font-size: 0.78rem; font-weight: 600; color: var(--ink); padding: 0.15rem 0.4rem; }
.flow-arrow { color: var(--accent); font-size: 0.9rem; padding: 0 0.15rem; }
.model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 0.55rem; }
.model-block {
  border: 1px solid var(--line); background: var(--paper-2); border-radius: 2px;
  padding: 0.55rem 0.7rem 0.6rem;
}
.model-name { display: block; font-size: 0.8rem; font-weight: 650; color: var(--ink); margin-bottom: 0.18rem; }
.model-block p { font-size: 0.78rem; color: var(--ink-2); line-height: 1.45; margin: 0; }
.metrics-note { font-size: 0.78rem; color: var(--ink-2); margin-bottom: 0.5rem; }

/* Limitaciones: grilla escaneable sin tarjetas (solo hairlines y aire). */
.limit-grid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 1.6rem; margin-bottom: 0.5rem; }
.limit-item { border-top: 1px solid var(--line); padding: 0.45rem 0.2rem 0.5rem; }
.limit-title { display: block; font-size: 0.74rem; font-weight: 650; color: var(--ink); margin-bottom: 0.16rem; }
.limit-item p { font-size: 0.78rem; color: var(--ink-2); line-height: 1.45; margin: 0; }

.obs-list { padding-left: 1.1rem; }
.obs-list li { font-size: 0.82rem; margin-bottom: 0.18rem; }

/* Accesibilidad: foco visible, equivalente textual, contraste. */
a:focus-visible, button:focus-visible,
[data-testid="stSelectbox"] button:focus-visible,
[data-testid="stTable"] *:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px;
}
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  max-width: 1px; max-height: 1px;
}
/* El equivalente textual es una <table>: con ancho fijo y layout fijo el
   contenido nowrap nunca expande la caja (evita overflow horizontal de la
   página en viewports pequeños; la tabla oculta no aporta scroll extra). */
table.sr-only { table-layout: fixed; }
[data-testid="stAlert"] { border-radius: 2px; border: 1px solid #D9B8AD; border-top: 3px solid #8A3A2E; }

/* Responsive: 1024px y 640px. */
@media (max-width: 1024px) {
  .conclusions { grid-template-columns: 1fr; }
  .bar-row { grid-template-columns: 8.2rem 1fr 3.4rem 6rem; }
  .provenance { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .facts { grid-template-columns: 1fr 1fr; }
  .chart-cols { grid-template-columns: 1fr; }
  .model-grid { grid-template-columns: 1fr; }
  .limit-grid { grid-template-columns: 1fr; }
  .proto-legend { grid-template-columns: 1fr; gap: 0.4rem; }
  .bar-row { grid-template-columns: 1fr 1fr; }
  .bar-model { grid-column: 1 / 2; }
  .bar-track { grid-column: 1 / 2; grid-row: 2; }
  .bar-value { grid-column: 2 / 3; grid-row: 1; }
  .min-tag { grid-column: 2 / 3; grid-row: 2; justify-self: start; }
  .band-row { grid-template-columns: 1fr 2.6rem; }
  .band-track { grid-column: 1 / 3; grid-row: 2; }
  .flow { flex-direction: column; align-items: flex-start; }
  .flow-arrow { transform: rotate(90deg); padding: 0.1rem 0; }
  .viewer-nav { gap: 0.15rem 0.9rem; }
}

/* Movimiento: un momento por componente; desactivado bajo prefers-reduced-motion. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}
"""


def _anchor(anchor_id: str) -> str:
    return f'<span id="{anchor_id}" class="anchor" aria-hidden="true"></span>'


def build_table_rows(
    report: BenchmarkReport,
    fold: str | None,
    fold_labels: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Filas agregadas planas para el dataframe, formateadas para mostrar."""
    labels = fold_labels or public_fold_labels(report)
    rows: list[dict[str, object]] = []
    for row in report.results_table(fold=fold):
        rows.append(
            {
                "Partición": labels[row["fold"]],
                "Modelo": MODEL_PUBLIC_NAMES[row["model"]],
                "MAE": fmt_metric(row["mae"]),
                "RMSE": fmt_metric(row["rmse"]),
                "WMAPE": fmt_metric(row["wmape"]),
                "Sesgo": fmt_metric(row["bias"]),
                "Cobertura": fmt_metric(row["coverage"], ".2f"),
                "Disponibles": fmt_count(row["available"]),
                "Solicitadas": fmt_count(row["requested"]),
                "Evaluadas": fmt_count(row["evaluated"]),
                "Filas con quiebre": fmt_count(row["stockout_rows"]),
            }
        )
    return rows


def spanish_observations(report: BenchmarkReport) -> list[str]:
    """Declaraciones de observación en español calculadas del reporte en tiempo de ejecución.

    Por partición y métrica, el valor no None más pequeño se deriva de los datos
    actuales del reporte (nunca hardcodeado); las particiones y modelos se
    muestran con nombres públicos en español, nunca con claves técnicas crudas.
    La declaración de no-ganador universal sigue a ``report.has_universal_winner``.
    """
    fold_labels = public_fold_labels(report)
    statements: list[str] = []
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("wmape", "WMAPE")):
        for fold in report.folds:
            candidates = [
                row for row in report.results_table(fold=fold.fold)
                if row[metric] is not None
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: row[metric])
            statements.append(
                f"En la {fold_labels[fold.fold]}, el "
                f"{MODEL_PUBLIC_NAMES[best['model']].lower()} tiene el {label} "
                f"más bajo ({best[metric]:.3f})."
            )
    names = [MODEL_PUBLIC_NAMES[model] for model in report.models]
    statements.append(
        f"Líneas base evaluadas: {', '.join(names[:-1])} y {names[-1]}."
    )
    if not report.has_universal_winner:
        statements.append(
            "Ningún modelo individual tiene el error más bajo en todas las métricas "
            "de ambas particiones."
        )
    return statements


def render_nav() -> None:
    """Navegación de anclas: las secciones principales del visor, una línea."""
    links = (
        ("#resumen", "Resumen"),
        ("#protocolo", "Protocolo"),
        ("#comparacion", "Comparación"),
        ("#quiebres", "Quiebres"),
        ("#metodologia", "Metodología"),
        ("#limitaciones", "Limitaciones"),
        ("#detalles", "Detalles técnicos"),
    )
    items = "".join(
        f'<a href="{href}">{label}</a>' for href, label in links
    )
    st.markdown(
        f'<a class="skip-link" href="#resumen">Saltar al contenido</a>'
        f'<nav class="viewer-nav" aria-label="Secciones del visor">{items}</nav>',
        unsafe_allow_html=True,
    )


def render_summary(report: BenchmarkReport) -> None:
    """Primer bloque visible: hechos, hallazgo de no-ganador, conclusiones y procedencia."""
    facts = [
        ("Filas transmitidas", fmt_count(report.rows), True),
        ("Productos", fmt_count(report.products), True),
        ("Particiones", fmt_count(len(report.folds)), True),
        ("Líneas base", fmt_count(len(report.models)), True),
    ]
    cells = "".join(
        f'<div class="fact" role="listitem">'
        f'<span class="fact-label">{label}</span>'
        f'<span class="fact-value{" mono" if mono else ""}">{value}</span>'
        f"</div>"
        for label, value, mono in facts
    )

    summary = conclusion_summary(report)
    cards = []
    for part in summary["partitions"]:
        by_model: dict[str, list[str]] = {}
        for metric in METRIC_ORDER:
            entry = part["per_metric"].get(metric)
            if entry is None:
                continue
            model_label, value = entry
            by_model.setdefault(model_label, []).append(f"{METRIC_LABELS[metric]} ({value:.3f})")
        if len(by_model) == 1:
            model_label, parts = next(iter(by_model.items()))
            text = f"Menor valor: {model_label} en {', '.join(parts)}."
        else:
            chunks = []
            for metric in METRIC_ORDER:
                entry = part["per_metric"].get(metric)
                if entry is None:
                    continue
                model_label, value = entry
                chunks.append(f"{METRIC_LABELS[metric]} lo registra {model_label} ({value:.3f})")
            text = "Menor valor: " + "; ".join(chunks) + "."
        cards.append(f'<div class="conclusion-card" data-testid="conclusion-card">'
                     f'<span class="conclusion-title">{part["fold_label"]}</span>'
                     f'<p class="conclusion-text">{text}</p></div>')
    if summary["has_universal_winner"]:
        why = "Un mismo modelo registra el menor valor en todas las métricas de ambas particiones."
    else:
        why = (
            "El modelo con el menor valor cambia según la partición y la métrica; "
            "ningún modelo lo registra en todas las métricas de ambas particiones."
        )
    cards.append(f'<div class="conclusion-card" data-testid="conclusion-card">'
                 f'<span class="conclusion-title">Sin ganador universal</span>'
                 f'<p class="conclusion-text">{why}</p></div>')

    st.markdown(
        f'<div class="summary" data-testid="summary-block">'
        f'{_anchor("resumen")}'
        f'<div class="facts" role="list" data-testid="summary-facts">{cells}</div>'
        f'<p class="finding" role="note" data-testid="no-winner-finding">'
        f"{EXACT_NO_WINNER_SENTENCE}</p>"
        f'<div class="conclusions" data-testid="summary-conclusions">{"".join(cards)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_provenance(report: BenchmarkReport) -> None:
    """Procedencia del reporte: conjunto de datos, revisión, ID y rango observado."""
    provenance = [
        ("Conjunto de datos", report.dataset),
        ("Revisión", report.dataset_revision),
        ("ID de evaluación", report.evaluation_id),
        ("Rango observado", f"{report.date_min} → {report.date_max} ({fmt_count(report.calendar_span_days)} días)"),
    ]
    prov_cells = "".join(
        f'<div class="prov-item"><span class="prov-label">{label}</span>'
        f'<span class="prov-value">{value}</span></div>'
        for label, value in provenance
    )
    st.markdown(
        f'<div class="provenance" data-testid="provenance-strip">{prov_cells}</div>',
        unsafe_allow_html=True,
    )


def _lane_aria(fold: dict) -> str:
    parts = []
    for seg in fold["segments"]:
        parts.append(
            f"{seg['name'].lower()} del {seg['start']} al {seg['end']} "
            f"con {fmt_count(seg['rows'])} filas"
        )
    fit = fold["fit"]
    return (
        f"{fold['fold_label']}: {'; '.join(parts)}. Historial de ajuste del "
        f"{fit['start']} al {fit['end']} con {fmt_count(fit['rows'])} filas, "
        "entrenamiento más validación; las filas de prueba nunca entran al "
        "historial de ajuste."
    )


def render_protocol(report: BenchmarkReport) -> None:
    """Protocolo temporal visual: carril proporcional por partición + historial de ajuste.

    Cada Partición muestra Entrenamiento → Validación → Prueba con fechas y
    conteos, y el historial de ajuste (entrenamiento + validación) con su propio
    carril: la prueba nunca entra al historial de ajuste.
    """
    folds = protocol_segments(report)
    blocks = []
    for fold in folds:
        kind_by_name = {"Entrenamiento": "train", "Validación": "val", "Prueba": "test"}
        lane_parts = []
        for seg in fold["segments"]:
            lane_parts.append(
                f'<span class="seg seg-{kind_by_name[seg["name"]]}" '
                f'style="width:{seg["width_pct"]}%" '
                f'title="{seg["name"]}: {seg["start"]} → {seg["end"]} · '
                f'{fmt_count(seg["rows"])} filas" aria-hidden="true"></span>'
            )
        legend = "".join(
            f'<div class="legend-col">'
            f'<span class="legend-name">{seg["name"]}</span>'
            f'<span class="legend-line">{seg["start"]} → {seg["end"]} · {fmt_count(seg["rows"])} filas</span>'
            f"</div>"
            for seg in fold["segments"]
        )
        fit = fold["fit"]
        fit_html = (
            f'<div class="fit-row"><span class="fit-name">historial de ajuste</span>'
            f'<span class="fit-meta"><span class="mono">{fit["start"]} → {fit["end"]}</span>, '
            f'<span class="mono">{fmt_count(fit["rows"])} filas</span> = '
            f"entrenamiento + validación; las filas de prueba nunca entran al historial de ajuste</span></div>"
            f'<div class="fit-track"><span class="fit-bar" style="width:{fit["width_pct"]}%"></span></div>'
        )
        blocks.append(
            f'<div class="proto-fold" data-testid="protocol-fold">'
            f'<div class="proto-fold-head">'
            f'<span class="proto-fold-name">{fold["fold_label"]}</span>'
            f'<div class="proto-lane" role="img" aria-label="{_lane_aria(fold)}">'
            f'{"".join(lane_parts)}</div></div>'
            f'<div class="proto-legend">{legend}</div>{fit_html}</div>'
        )
    st.markdown(
        f'<div class="proto" role="group" aria-label="Protocolo temporal: '
        f'entrenamiento a validación a prueba en cada partición; las filas de '
        f'prueba nunca entran al historial de ajuste" '
        f'data-testid="protocol-diagram">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def render_hist_note() -> None:
    """Nota histórica v1 -> v2: el visor nunca lee el reporte v1 de diagnóstico."""
    st.markdown(
        '<div class="hist-note" data-testid="historical-note">'
        "<h3>Nota histórica del protocolo</h3>"
        "<p>El reporte v1 de diagnóstico (que este visor nunca lee) registró una "
        "cobertura de prueba del naive estacional de 0 con el ajuste antiguo de "
        "solo entrenamiento. La v2 registra cobertura 1.00 para cada modelo en cada "
        "partición. La diferencia es el protocolo de historial de ajuste (la v2 "
        "ajusta con entrenamiento + validación ordenados), no una mejora "
        "artificial del modelo.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _equivalent_table_html(report: BenchmarkReport, metric: str) -> str:
    """Equivalente textual del gráfico: tabla oculta visualmente, legible por AT."""
    equivalent = comparison_equivalent_rows(report, metric)
    head = (
        "<thead><tr><th>Partición</th><th>Modelo</th>"
        f"<th>{metric}</th><th>Mínimo de la partición</th></tr></thead>"
    )
    body = "".join(
        f"<tr><td>{row['partition']}</td><td>{row['model']}</td>"
        f"<td>{row['value']}</td><td>{row['minimum']}</td></tr>"
        for row in equivalent
    )
    return (
        f'<table class="sr-only" data-testid="chart-equivalent" '
        f'aria-label="Equivalente textual del gráfico de {metric}">'
        f"{head}<tbody>{body}</tbody></table>"
    )


def _chart_html(report: BenchmarkReport, metric: str) -> str:
    """Gráfico de comparación por partición, construido solo con las 8 filas del reporte.

    Barras con longitud relativa al mayor valor de la partición, valor exacto de
    3 decimales junto a cada barra y un marcador accesible "menor valor" sobre el
    mínimo de cada partición. Sin librerías de gráficos: HTML/CSS/SVG propios.
    """
    fold_labels = public_fold_labels(report)
    rows = comparison_data(report, metric)
    by_fold: dict[str, list[dict]] = {}
    for row in rows:
        by_fold.setdefault(row["fold_label"], []).append(row)
    min_by_fold = comparison_min_values(report, metric)

    columns = []
    for fold in report.folds:
        label = fold_labels[fold.fold]
        part_rows = by_fold.get(label, [])
        if not part_rows:
            continue
        max_value = max(row["value"] for row in part_rows)
        bars = []
        for row in part_rows:
            width = round(row["value"] / max_value * 100, 1)
            aria = f"{row['model_label']}: {row['formatted']} de {metric} en la {label}"
            if row["is_min"]:
                aria += ". Es el menor valor de la partición"
            bar = (
                f'<div class="bar-row" aria-label="{aria}">'
                f'<span class="bar-model">{row["model_label"]}</span>'
                f'<span class="bar-track"><span class="bar-fill'
                f'{" is-min" if row["is_min"] else ""}" style="width:{width}%"></span></span>'
                f'<span class="bar-value">{row["formatted"]}</span>'
            )
            if row["is_min"]:
                bar += '<span class="min-tag" data-testid="min-marker">menor valor</span>'
            else:
                bar += '<span class="min-tag is-empty" aria-hidden="true"></span>'
            bar += "</div>"
            bars.append(bar)
        columns.append(
            f'<div class="chart-col" data-testid="partition-column">'
            f'<div class="chart-col-head"><span class="chart-col-name">{label}</span>'
            f'<span class="chart-col-range">{fold.test_start} → {fold.test_end}, '
            f'{fmt_count(fold.test_rows)} filas de prueba</span></div>'
            + "".join(bars)
            + "</div>"
        )
    minima = ", ".join(f"{label}: {value}" for label, value in min_by_fold.items())
    aria = (
        f"Gráfico de comparación de {metric} entre modelos y particiones. "
        f"La barra resaltada marca el valor mínimo de cada partición ({minima}). "
        "Los valores exactos están junto a cada barra y en la tabla equivalente."
    )
    return (
        f'<div class="chart" role="group" aria-label="{aria}" '
        f'data-testid="benchmark-chart">'
        f'<div class="chart-cols">{"".join(columns)}</div>'
        f"{_equivalent_table_html(report, metric)}</div>"
    )


def render_comparison(report: BenchmarkReport) -> None:
    """Selector de métrica + gráfico de comparación + tabla estática completa."""
    fold_public = public_fold_labels(report)
    fold_by_public = {label: key for key, label in fold_public.items()}
    fold_options = ["Todas las particiones", *fold_public.values()]

    fold_col, metric_col = st.columns(2, gap="large")
    with fold_col:
        selected_fold = st.selectbox(
            "Filtrar por partición",
            options=fold_options,
            help="Filtra la tabla de agregados; el reporte nunca se recarga.",
        )
    with metric_col:
        selected_metric = st.selectbox(
            "Métrica para comparar",
            options=list(METRIC_KEYS),
            help="Cambia la métrica del gráfico; los valores provienen del reporte.",
        )

    st.markdown(_chart_html(report, selected_metric), unsafe_allow_html=True)
    st.caption(
        "El modelo con el menor valor cambia según la partición y la métrica; este "
        "benchmark no selecciona ganador. La barra resaltada marca el mínimo de cada "
        "partición (longitud relativa al mayor valor); el filtro de partición afecta "
        "la tabla y el gráfico siempre muestra ambas particiones."
    )

    fold = None if selected_fold == "Todas las particiones" else fold_by_public[selected_fold]
    table_rows = build_table_rows(report, fold, fold_labels=fold_public)
    # Tabla estática (no virtualizada) para que cada fila agregada siempre se
    # pinte; el selector de partición de arriba sigue siendo la interacción de filtro.
    st.table(pd.DataFrame(table_rows).style.hide(axis="index"))
    st.caption(
        f"{len(table_rows)} filas agregadas. Orden fijo del reporte: partición, luego "
        "modelo. Sin ranking, sin selección de ganador."
    )


def render_observations(report: BenchmarkReport) -> None:
    """Observaciones calculadas del reporte más las salvedades fijas de alcance."""
    items = list(spanish_observations(report))
    items.append(
        f"Alcance limitado: solo {fmt_count(report.rows)} filas, "
        f"{fmt_count(report.products)} productos y {len(report.folds)} particiones."
    )
    items.append(
        "Las ventas observadas no son demanda real ni estimación de demanda "
        "latente recuperada."
    )
    items.append("Los quiebres de stock no fueron corregidos ni enmascarados.")
    items.append("Ningún modelo está listo para producción.")
    html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<ul class="obs-list" data-testid="observations-list">{html}</ul>',
        unsafe_allow_html=True,
    )


def render_stockouts(report: BenchmarkReport) -> None:
    """Quiebres de stock: conteos derivados, proporciones y salvedad de censura."""
    facts = stockout_facts(report)
    test_p1 = facts["test_rows_by_fold"].get("Partición 1")
    test_p2 = facts["test_rows_by_fold"].get("Partición 2")
    tiles = [
        ("Filas con quiebre en el acotado", fmt_count(facts["stockout_rows"])),
        ("Filas con alguna hora con estado == 1", fmt_count(facts["rows_with_any_observed_stockout_status"])),
        ("Filas con quiebre en prueba (Partición 1 / Partición 2)", f"{fmt_count(test_p1)} / {fmt_count(test_p2)}"),
    ]
    cells = "".join(
        f'<div class="fact" role="listitem"><span class="fact-label">{label}</span>'
        f'<span class="fact-value mono">{value}</span></div>'
        for label, value in tiles
    )
    pct_bounded = facts["stockout_rows"] / facts["rows"] * 100
    pct_any = facts["rows_with_any_observed_stockout_status"] / facts["rows"] * 100
    bands = (
        f'<div class="band-row">'
        f'<span class="band-label">{fmt_count(facts["stockout_rows"])} de {fmt_count(facts["rows"])} filas con quiebre</span>'
        f'<span class="band-track"><span class="band-fill" style="width:{pct_bounded:.1f}%" aria-hidden="true"></span></span>'
        f'<span class="band-pct">{pct_bounded:.1f}%</span></div>'
        f'<div class="band-row">'
        f'<span class="band-label">{fmt_count(facts["rows_with_any_observed_stockout_status"])} de {fmt_count(facts["rows"])} filas con alguna hora con estado == 1</span>'
        f'<span class="band-track"><span class="band-fill" style="width:{pct_any:.1f}%" aria-hidden="true"></span></span>'
        f'<span class="band-pct">{pct_any:.1f}%</span></div>'
    )
    st.markdown(
        f'<div data-testid="stockout-section">'
        f'<div class="facts" role="list" data-testid="stockout-facts">{cells}</div>'
        f'<div class="bands" data-testid="stockout-bands">{bands}</div>'
        "<ul class=\"obs-list\">"
        "<li>Las ventas observadas permanecen sin corregir; este reporte "
        "no contiene ninguna estimación de demanda latente.</li>"
        "<li>Los quiebres de stock no se corrigen ni se enmascaran: las filas con "
        "estado == 1 permanecen en la evaluación con sus valores observados.</li>"
        "<li>Las filas con quiebre se conservan tal cual: "
        f"{fmt_count(facts['stockout_rows'])} en el acotado, {fmt_count(test_p1)} en la "
        f"prueba de la Partición 1 y {fmt_count(test_p2)} en la de la Partición 2.</li>"
        "<li>Las ventas observadas no son demanda real; las métricas evalúan "
        "únicamente las ventas observadas.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )


def render_methodology(report: BenchmarkReport) -> None:
    """Flujo metodológico y bloques explicativos de las cuatro líneas base."""
    steps = []
    for index, step in enumerate(METHODOLOGY_STEPS):
        steps.append(f'<span class="flow-step" role="listitem">{step}</span>')
        if index < len(METHODOLOGY_STEPS) - 1:
            steps.append('<span class="flow-arrow" aria-hidden="true">→</span>')
    blocks = "".join(
        f'<div class="model-block" data-testid="model-block">'
        f'<span class="model-name">{block["model_label"]}</span>'
        f'<p>{block["explanation"]}</p></div>'
        for block in model_explanations(report)
    )
    st.markdown(
        f'<div data-testid="methodology-section">'
        f'<div class="flow" role="list" aria-label="Flujo metodológico del benchmark">'
        f'{"".join(steps)}</div>'
        f'<p class="metrics-note">Las métricas se calculan sobre las ventas observadas: '
        "MAE, RMSE y WMAPE (un valor menor indica un error menor), además de sesgo "
        "y cobertura.</p>"
        f'<div class="model-grid">{blocks}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_limitations(report: BenchmarkReport) -> None:
    """Limitaciones en una grilla escaneable: escala, alcance y honestidad."""
    items = "".join(
        f'<div class="limit-item"><span class="limit-title">{item["title"]}</span>'
        f'<p>{item["text"]}</p></div>'
        for item in limitation_items(report, REPORT_TEXT_TRANSLATIONS)
    )
    st.markdown(
        f'<div class="limit-grid" data-testid="limitations-section">{items}</div>',
        unsafe_allow_html=True,
    )


def render_warnings(report: BenchmarkReport) -> None:
    """Advertencias del reporte traducidas (el JSON congelado nunca se modifica)."""
    html = "".join(
        f"<li>{REPORT_TEXT_TRANSLATIONS[warning]}</li>" for warning in report.warnings
    )
    st.markdown(
        f'<ul class="obs-list" data-testid="warnings-list">{html}</ul>',
        unsafe_allow_html=True,
    )


def render_details(report: BenchmarkReport) -> None:
    """Procedencia + detalles técnicos: claves exactas, aisladas de la lectura pública."""
    render_provenance(report)
    st.markdown(
        f"- **Reporte congelado (solo lectura, ruta fija):** `{REPORT_RELATIVE_PATH}`.\n"
        f"- **Conjunto de datos:** `{report.dataset}`; **Revisión:** `{report.dataset_revision}`.\n"
        f"- **ID de evaluación:** `{report.evaluation_id}`.\n"
        f"- **Protocolo:** `{report.evaluation_protocol_version}`; partición evaluada "
        f"`{report.evaluation_partition}`; `fit_partition` = `{report.fit_partition}`; "
        f"`test_excluded_from_fit` = `{str(report.test_excluded_from_fit).lower()}`; "
        f"`baseline_parameters_fixed` = `{str(report.baseline_parameters_fixed).lower()}`.\n"
        f"- **Claves de partición:** {', '.join(f'`{fold.fold}`' for fold in report.folds)}.\n"
        f"- **Claves de modelo:** {', '.join(f'`{model}`' for model in report.models)}; ventana de "
        f"promedio móvil {report.moving_average_window} días; alpha de suavizamiento exponencial "
        f"{report.ses_alpha}; parámetros fijos, no se realiza ninguna búsqueda de parámetros.\n"
        f"- **Objetivo evaluado:** `observed_sales`.\n"
        f"- **Esquema del reporte:** `{report.schema_name}`.\n"
        "- **v1 de diagnóstico (nunca se carga):** `freshretailnet-50k-bounded-real-1000.json`."
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Visor de benchmark acotado — FreshRetailNet-50K",
    layout="wide",
)

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

try:
    report = load_report()
except BenchmarkReportError as exc:
    st.error(
        f"**No se puede mostrar el reporte del benchmark.**\n\n{exc}\n\n"
        f"Archivo esperado: `{REPORT_RELATIVE_PATH}` (relativo al repositorio, solo lectura). "
        "El visor se detiene sin mostrar datos antes que sustituir valores que podrían falsear el benchmark."
    )
    st.stop()

untranslated = [
    text for text in (*report.limitations, *report.warnings)
    if text not in REPORT_TEXT_TRANSLATIONS
]
if untranslated:
    st.error(
        "**El reporte contiene texto sin traducción al español en este visor.**\n\n"
        "Este visor solo muestra en español el texto del reporte que tiene una traducción "
        "definida en su capa de presentación; el reporte congelado nunca se modifica. "
        "Actualice el mapa `REPORT_TEXT_TRANSLATIONS` del visor o revise el reporte.\n\n"
        f"Texto sin traducir: {untranslated}"
    )
    st.stop()

render_nav()

st.markdown('<span class="chips" role="status">Solo lectura · Solo agregados</span>', unsafe_allow_html=True)
st.title("Visor de benchmark acotado de pronósticos")
st.markdown(
    '<div class="subtitle">'
    f"{fmt_count(report.rows)} filas transmitidas · {fmt_count(report.products)} productos · "
    f"{len(report.folds)} particiones · {len(report.models)} líneas base"
    "</div>",
    unsafe_allow_html=True,
)

render_summary(report)

st.markdown("## Protocolo temporal")
st.markdown(_anchor("protocolo"), unsafe_allow_html=True)
render_protocol(report)

st.markdown("## Resultados por partición y modelo")
st.markdown(_anchor("comparacion"), unsafe_allow_html=True)
render_comparison(report)

st.markdown("## Observaciones")
render_observations(report)

st.markdown("## Datos de quiebres de stock: observados, sin corregir")
st.markdown(_anchor("quiebres"), unsafe_allow_html=True)
render_stockouts(report)

st.markdown("## Metodología")
st.markdown(_anchor("metodologia"), unsafe_allow_html=True)
render_methodology(report)

st.markdown("## Limitaciones")
st.markdown(_anchor("limitaciones"), unsafe_allow_html=True)
render_limitations(report)

st.markdown("## Advertencias del reporte")
render_warnings(report)

render_hist_note()

st.markdown("## Detalles técnicos y procedencia")
st.markdown(_anchor("detalles"), unsafe_allow_html=True)
render_details(report)

st.markdown("---")
st.markdown(f"**{EXACT_NO_WINNER_SENTENCE}**")
st.caption(
    "Datos: ver los detalles técnicos al final de la página. Solo lectura, solo "
    "agregados y sin solicitudes de datos en vivo."
)
