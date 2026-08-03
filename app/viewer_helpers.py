"""Capa de datos derivados del visor de benchmark (funciones puras, sin Streamlit).

Toda la capa de datos del visor vive aquí para que las pruebas puedan verificarla
sin depender del renderizado de Streamlit: cada función recibe un
:class:`~inventory_optimizer.demo.benchmark_report.BenchmarkReport` y devuelve
estructuras planas con nombres públicos en español, listas para renderizar.

Contrato de datos: ningún valor métrico está hardcodeado. Todo se deriva en
tiempo de ejecución de ``report.results_table()`` y de los campos del reporte
congelado; los ``None`` nunca se inventan. Este módulo no realiza acceso a la
red, no importa Streamlit y nunca modifica el reporte.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from inventory_optimizer.demo.benchmark_report import BenchmarkReport

# Nombres públicos en español de los modelos para la UI principal (sin claves
# técnicas crudas). Las claves técnicas exactas viven solo en la sección
# "Detalles técnicos y procedencia" del visor.
MODEL_PUBLIC_NAMES: dict[str, str] = {
    "naive": "Último valor",
    "seasonal_naive": "Naive estacional",
    "moving_average": "Promedio móvil",
    "ses": "Suavizamiento exponencial",
}

# Métricas: etiqueta pública del selector -> clave técnica del reporte.
METRIC_KEYS: dict[str, str] = {"MAE": "mae", "RMSE": "rmse", "WMAPE": "wmape"}
# Orden canónico de las métricas de error dentro del reporte.
METRIC_ORDER: tuple[str, ...] = ("mae", "rmse", "wmape")
METRIC_LABELS: dict[str, str] = {"mae": "MAE", "rmse": "RMSE", "wmape": "WMAPE"}

# Pasos del flujo metodológico en orden real del pipeline (público en español;
# "transmisión acotada" evita el anglicismo "streaming" en la copia pública).
METHODOLOGY_STEPS: tuple[str, ...] = (
    "Fuente",
    "Transmisión acotada",
    "Particiones temporales",
    "Líneas base",
    "Métricas",
    "Análisis de limitaciones",
)

_SEGMENT_NAMES: tuple[str, str, str] = ("Entrenamiento", "Validación", "Prueba")

_MONO_STACK = (
    'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, '
    '"DejaVu Sans Mono", monospace'
)


def fmt_count(value: int) -> str:
    """Separador de miles en español para conteos enteros públicos (1,000 -> 1.000)."""
    return f"{value:,}".replace(",", ".")


def fmt_metric(value: float | None, spec: str = ".3f") -> str:
    """Formato de métricas: exactamente 3 decimales; ``None`` -> '—'."""
    return "—" if value is None else f"{value:{spec}}"


def public_fold_labels(report: BenchmarkReport) -> dict[str, str]:
    """Nombres públicos en español de las particiones, en orden del reporte."""
    return {fold.fold: f"Partición {index}" for index, fold in enumerate(report.folds, 1)}


def comparison_data(report: BenchmarkReport, metric: str) -> list[dict[str, Any]]:
    """Filas del gráfico de comparación para una métrica pública ("MAE"/"RMSE"/"WMAPE").

    Devuelve una fila por (partición, modelo) con el valor exacto del reporte, su
    formato de 3 decimales y si es el menor valor de su partición. Solo se marca
    un mínimo por partición y métrica; los valores ``None`` nunca se inventan y
    las claves de métrica desconocidas fallan con ``KeyError``.
    """
    key = METRIC_KEYS[metric]
    fold_labels = public_fold_labels(report)
    rows: list[dict[str, Any]] = []
    for fold in report.folds:
        fold_rows = [r for r in report.results_table(fold=fold.fold) if r[key] is not None]
        if not fold_rows:
            continue
        best = min(fold_rows, key=lambda r: r[key])
        for r in fold_rows:
            rows.append(
                {
                    "fold": r["fold"],
                    "fold_label": fold_labels[r["fold"]],
                    "model": r["model"],
                    "model_label": MODEL_PUBLIC_NAMES[r["model"]],
                    "value": r[key],
                    "formatted": f"{r[key]:.3f}",
                    "is_min": r is best,
                }
            )
    return rows


def comparison_equivalent_rows(report: BenchmarkReport, metric: str) -> list[dict[str, str]]:
    """Filas planas del equivalente textual del gráfico (mismas reglas que ``comparison_data``)."""
    return [
        {
            "partition": r["fold_label"],
            "model": r["model_label"],
            "value": r["formatted"],
            "minimum": "menor valor" if r["is_min"] else "",
        }
        for r in comparison_data(report, metric)
    ]


def comparison_min_values(report: BenchmarkReport, metric: str) -> dict[str, str]:
    """Valor mínimo por partición (etiqueta pública -> cadena de 3 decimales)."""
    key = METRIC_KEYS[metric]
    fold_labels = public_fold_labels(report)
    out: dict[str, str] = {}
    for fold in report.folds:
        values = [r[key] for r in report.results_table(fold=fold.fold) if r[key] is not None]
        if values:
            out[fold_labels[fold.fold]] = f"{min(values):.3f}"
    return out


def conclusion_summary(report: BenchmarkReport) -> dict[str, Any]:
    """Resumen derivado: menor valor por partición y por qué no hay ganador universal.

    Devuelve una entrada por partición con el modelo y el valor del mínimo de cada
    métrica (mae/rmse/wmape) y el indicador ``has_universal_winner`` del reporte.
    Nunca declara un ganador global: solo el menor valor por partición y métrica.
    """
    fold_labels = public_fold_labels(report)
    partitions: list[dict[str, Any]] = []
    for fold in report.folds:
        per_metric: dict[str, tuple[str, float]] = {}
        for metric in METRIC_ORDER:
            candidates = [
                r for r in report.results_table(fold=fold.fold) if r[metric] is not None
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: r[metric])
            per_metric[metric] = (MODEL_PUBLIC_NAMES[best["model"]], best[metric])
        partitions.append({"fold_label": fold_labels[fold.fold], "per_metric": per_metric})
    return {"partitions": partitions, "has_universal_winner": report.has_universal_winner}


def stockout_facts(report: BenchmarkReport) -> dict[str, Any]:
    """Conteos de quiebre derivados del reporte, con etiquetas públicas de partición."""
    fold_labels = public_fold_labels(report)
    return {
        "stockout_rows": report.coverage.stockout_rows,
        "rows_with_any_observed_stockout_status": report.coverage.rows_with_any_observed_stockout_status,
        "rows": report.rows,
        "test_rows_by_fold": {
            fold_labels[fold.fold]: report.fold_test_stockout_rows(fold.fold)
            for fold in report.folds
        },
    }


def _span_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def protocol_segments(report: BenchmarkReport) -> list[dict[str, Any]]:
    """Segmentos del protocolo por partición con anchos proporcionales al lapso.

    Cada partición devuelve tres segmentos (Entrenamiento, Validación, Prueba)
    con fechas, filas y ancho porcentual (la suma es exactamente 100.0) y el
    historial de ajuste (entrenamiento + validación) con su propio ancho. El
    ancho del ajuste es la suma de los dos primeros segmentos: la prueba nunca
    entra al historial de ajuste.
    """
    fold_labels = public_fold_labels(report)
    out: list[dict[str, Any]] = []
    for fold in report.folds:
        segments = [
            {
                "name": _SEGMENT_NAMES[0],
                "start": fold.train_start,
                "end": fold.train_end,
                "rows": fold.train_rows,
            },
            {
                "name": _SEGMENT_NAMES[1],
                "start": fold.validation_start,
                "end": fold.validation_end,
                "rows": fold.validation_rows,
            },
            {
                "name": _SEGMENT_NAMES[2],
                "start": fold.test_start,
                "end": fold.test_end,
                "rows": fold.test_rows,
            },
        ]
        days = [_span_days(seg["start"], seg["end"]) for seg in segments]
        total = sum(days)
        widths = [round(day / total * 100, 1) for day in days[:2]]
        widths.append(round(100.0 - widths[0] - widths[1], 1))
        for seg, width in zip(segments, widths):
            seg["width_pct"] = width
        out.append(
            {
                "fold_label": fold_labels[fold.fold],
                "segments": segments,
                "fit": {
                    "start": fold.fit_history_start,
                    "end": fold.fit_history_end,
                    "rows": fold.fit_row_count,
                    "width_pct": round(widths[0] + widths[1], 1),
                },
            }
        )
    return out


def model_explanations(report: BenchmarkReport) -> list[dict[str, str]]:
    """Bloques explicativos de las cuatro líneas base, con parámetros del reporte.

    La ventana del promedio móvil y el alpha del suavizamiento exponencial se
    derivan del reporte en tiempo de ejecución; nunca están hardcodeados.
    """
    return [
        {
            "model_label": "Último valor",
            "explanation": "Predice el último valor de ventas observadas de cada producto.",
        },
        {
            "model_label": "Naive estacional",
            "explanation": "Repite el valor de ventas observadas de exactamente 7 días antes.",
        },
        {
            "model_label": "Promedio móvil",
            "explanation": (
                f"Promedia los últimos {report.moving_average_window} valores de "
                "ventas observadas de cada producto."
            ),
        },
        {
            "model_label": "Suavizamiento exponencial",
            "explanation": (
                f"Suavizamiento exponencial con alpha {report.ses_alpha:g}: pondera "
                "más las observaciones recientes."
            ),
        },
    ]


def limitation_items(
    report: BenchmarkReport, translations: dict[str, str]
) -> list[dict[str, str]]:
    """Items de limitaciones: escala, alcance y honestidad de los datos.

    Las traducciones de la prosa congelada del reporte vienen de la capa de
    presentación (``REPORT_TEXT_TRANSLATIONS``) para que el JSON nunca se
    modifique; los conteos de escala se derivan del reporte en tiempo de ejecución.
    """
    t = translations
    return [
        {
            "title": "Escala acotada",
            "text": (
                f"Acotado a {fmt_count(report.rows)} filas transmitidas, "
                f"{fmt_count(report.products)} productos y {len(report.folds)} particiones."
            ),
        },
        {
            "title": "Instantánea incompleta",
            "text": t["The evaluation is bounded to the streamed prefix, not a complete snapshot."],
        },
        {
            "title": "Historiales y fechas incompletos",
            "text": (
                t["The input prefix may have incomplete product histories and missing calendar dates."]
                + " El producto 548 tiene un historial parcial; las fechas de calendario faltantes "
                "se agregan en el reporte y nunca se listan aquí."
            ),
        },
        {
            "title": "Ventas observadas, no demanda real",
            "text": "Las ventas observadas no son demanda real.",
        },
        {
            "title": "Quiebres preservados, sin corregir",
            "text": t[
                "Rows with stockout metadata are preserved; censoring may affect observed-sales error interpretation."
            ],
        },
        {
            "title": "Sin demanda latente",
            "text": "Sin estimación de demanda latente; las ventas observadas no son demanda real.",
        },
        {
            "title": "Sin simulación ni optimización",
            "text": (
                "No es el conjunto de datos completo; no es una política de "
                "inventario. No hay plazo de entrega, costo, nivel de servicio ni "
                "restricciones operativas."
            ),
        },
        {
            "title": "Sin modelos avanzados",
            "text": t["Advanced forecasting and latent-demand recovery are not included."],
        },
        {
            "title": "Sin recomendación operativa",
            "text": "Ningún modelo está listo para producción.",
        },
    ]
