"""Constructores de fixtures pequeños para las pruebas de ingesta de FreshRetailNet-50K.

Son solo fixtures offline (pocos productos, sin descargas). El invariante global de
865 productos NO se aplica aquí deliberadamente; pertenece a la validación de
instantánea completa con contexto explícito (ver test_fresh_retail_snapshot.py).

Las filas de fixture usan la forma oficial de la fuente: clave ``dt``, ids de
producto numéricos opcionales y SIN entrada cruda de ``stockout_hours_6_22`` (ese
campo se deriva de la ventana de estado validada al momento del parseo).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from inventory_optimizer.ingestion.fresh_retail import (
    DailySourceRow,
    FRESH_RETAIL_PINNED_REVISION,
)

HOURS_PER_DAY = 24
WINDOW_START_HOUR = 6
WINDOW_EXCLUSIVE_END_HOUR = 22  # ventana semiabierta: índices 6..21 (16 posiciones)


def make_raw_row(
    *,
    product_id: str | int = "P0001",
    dt: str = "2024-01-15",
    sale_amount: float | None = None,
    hours_sale: list[float] | None = None,
    hours_stock_status: list[int] | None = None,
    stock_hour6_22_cnt: int | None = None,
) -> dict[str, Any]:
    """Construye una fila cruda diaria válida de la fuente; sobreescribe campos para romper invariantes.

    El contador de stock por defecto se deriva de la ventana de estado semiabierta
    [6:22) como el número de posiciones con status==1, de modo que una fila por
    defecto siempre valida; pasar un valor explícito permite a las pruebas crear
    filas inconsistentes.
    """
    status = hours_stock_status if hours_stock_status is not None else [0] * HOURS_PER_DAY
    sales = hours_sale if hours_sale is not None else [1.0] * HOURS_PER_DAY
    window = status[WINDOW_START_HOUR:WINDOW_EXCLUSIVE_END_HOUR]
    if stock_hour6_22_cnt is None:
        stock_hour6_22_cnt = sum(1 for value in window if value == 1)
    if sale_amount is None:
        sale_amount = float(sum(sales))
    return {
        "product_id": product_id,
        "dt": dt,
        "sale_amount": sale_amount,
        "hours_sale": list(sales),
        "hours_stock_status": list(status),
        "stock_hour6_22_cnt": stock_hour6_22_cnt,
    }


def make_daily_row(
    *,
    product_id: str | int = "P0001",
    dt: str = "2024-01-15",
    sale_amount: float = 24.0,
    stockout_hours_6_22: int = 0,
    revision: str = FRESH_RETAIL_PINNED_REVISION,
) -> DailySourceRow:
    """Construye una fila diaria validada construida directamente para pruebas de pronóstico.

    El vector de estado de 24 horas coloca los flags de quiebre solicitados al
    inicio de la ventana semiabierta [6:22) para que stock_hour6_22_cnt siga
    siendo consistente. El total diario se concentra en la hora 0 de hours_sale.
    """
    if not 0 <= stockout_hours_6_22 <= 16:
        raise ValueError("stockout_hours_6_22 must be between 0 and 16.")
    status = [0] * HOURS_PER_DAY
    for hour in range(WINDOW_START_HOUR, WINDOW_START_HOUR + stockout_hours_6_22):
        status[hour] = 1
    hours_sale = [0.0] * HOURS_PER_DAY
    hours_sale[0] = float(sale_amount)
    return DailySourceRow(
        product_id=product_id,
        dt=date.fromisoformat(dt),
        sale_amount=float(sale_amount),
        hours_sale=tuple(hours_sale),
        hours_stock_status=tuple(status),
        stock_hour6_22_cnt=stockout_hours_6_22,
        stockout_hours_6_22=stockout_hours_6_22,
        revision=revision,
    )


def make_series(
    *,
    product_id: str | int = "P0001",
    start: str = "2024-01-01",
    days: int = 10,
    base: float = 1.0,
) -> list[DailySourceRow]:
    """Filas diarias consecutivas con sale_amount = base + day_index."""
    start_date = date.fromisoformat(start)
    return [
        make_daily_row(
            product_id=product_id,
            dt=(start_date + timedelta(days=index)).isoformat(),
            sale_amount=base + index,
        )
        for index in range(days)
    ]


# --- fixture del ejecutor de evaluación offline (sintética, determinista) --------

EVALUATION_FIXTURE_SOURCE_ID = "fresh-retailnet-50k-dev"
EVALUATION_FIXTURE_REVISION = "test-rev-eval-1"
EVALUATION_FIXTURE_EVALUATION_ID = "eval-dev-001"
"""Identificadores del fixture sintético de evaluación offline.

El fixture usa una revisión de prueba distinta (no la revisión en vivo fijada)
para demostrar que las filas sintéticas pueden llevar cualquier revisión
siempre que sea igual a ``config.dataset_revision``.
"""


def make_evaluation_rows() -> tuple[DailySourceRow, ...]:
    """Filas sintéticas deterministas para el ejecutor de evaluación offline.

    El producto ``A`` corre de 2024-01-01 a 2024-01-20 con ``sale_amount`` igual
    al índice del día del mes (1..20), lo suficiente para dos particiones con
    filas reales de train/validation/test. El producto ``C`` (sin historia de
    entrenamiento en absoluto) cubre 2024-01-14..2024-01-19 con ``sale_amount``
    5.0, de modo que sus predicciones quedan explícitamente no disponibles.
    Metadatos de quiebre: ``A``@01-12 (1h), ``A``@01-15 (2h), ``A``@01-18 (2h);
    ``C``@01-15 y ``C``@01-18 llevan flags de quiebre que NO deben contarse
    porque las predicciones de ``C`` no están disponibles.
    """
    rows: list[DailySourceRow] = []
    stockout_a = {
        date(2024, 1, 12): 1,
        date(2024, 1, 15): 2,
        date(2024, 1, 18): 2,
    }
    for index in range(1, 21):
        dt = date(2024, 1, index)
        rows.append(
            make_daily_row(
                product_id="A",
                dt=dt.isoformat(),
                sale_amount=float(index),
                stockout_hours_6_22=stockout_a.get(dt, 0),
                revision=EVALUATION_FIXTURE_REVISION,
            )
        )
    for index in range(14, 20):
        rows.append(
            make_daily_row(
                product_id="C",
                dt=date(2024, 1, index).isoformat(),
                sale_amount=5.0,
                stockout_hours_6_22=3 if index in (15, 18) else 0,
                revision=EVALUATION_FIXTURE_REVISION,
            )
        )
    return tuple(rows)
