"""Paquete del visor de demo: visor de benchmark agregado de solo lectura.

La demo consiste en un lector de reporte puro, solo con stdlib
(:mod:`inventory_optimizer.demo.benchmark_report`) y una capa de renderizado en
Streamlit en ``app/streamlit_app.py``. Nunca importa los ejecutores de ingesta o
evaluación, nunca lee conjuntos de datos y nunca realiza acceso a la red.
"""

__all__ = [
    "BenchmarkReport",
    "BenchmarkReportError",
    "REPORT_PATH",
    "REPORT_RELATIVE_PATH",
    "load_report",
]

from inventory_optimizer.demo.benchmark_report import (
    REPORT_PATH,
    REPORT_RELATIVE_PATH,
    BenchmarkReport,
    BenchmarkReportError,
    load_report,
)
