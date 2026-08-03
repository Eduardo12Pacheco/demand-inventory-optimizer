"""Métricas de error de pronóstico sobre la forma común de predicción de líneas base.

Las métricas se calculan solo sobre pares EVALUADOS: filas donde tanto la
predicción como ``observed_sales`` están presentes. Fórmulas (documentadas con
exactitud):

* ``MAE = mean(|actual - prediction|)``
* ``RMSE = sqrt(mean((actual - prediction)**2))``
* ``WMAPE = sum(|actual - prediction|) / sum(|actual|)``
* ``bias = mean(prediction - actual)``

WMAPE tiene un estado explícito: ``calculable``, ``not_calculable_zero_actuals``
(un denominador cero mantiene el WMAPE numérico en ``None``), o
``no_evaluable_predictions`` (no se fabrica ninguna métrica numérica). Las filas
con quiebre se cuentan entre las predicciones evaluadas con
``stockout_hours_6_22 > 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from inventory_optimizer.forecasting.baselines import BaselineForecastResult

TARGET_LABEL = "observed_sales"

WMAPE_CALCULABLE = "calculable"
WMAPE_ZERO_ACTUALS = "not_calculable_zero_actuals"
WMAPE_NO_EVALUABLE = "no_evaluable_predictions"

WMAPEStatus = Literal["calculable", "not_calculable_zero_actuals", "no_evaluable_predictions"]


@dataclass(frozen=True)
class ForecastMetrics:
    """Reporte tipado de error de pronóstico por modelo/partición.

    ``target`` es siempre la etiqueta exacta ``observed_sales``. Todas las
    métricas numéricas son ``None`` cuando no existen predicciones evaluables.
    """

    model: str
    fold: str | int | None
    target: str
    requested: int
    available: int
    evaluated: int
    coverage: float
    stockout_rows: int
    mae: float | None
    rmse: float | None
    wmape: float | None
    bias: float | None
    wmape_status: WMAPEStatus


def evaluate_metrics(
    result: BaselineForecastResult,
    *,
    fold: str | int | None = None,
) -> ForecastMetrics:
    """Calcula el reporte métrico tipado para un resultado de pronóstico.

    ``fold`` es un identificador opcional provisto por el llamador que se
    traslada sin cambios. ``coverage`` es ``available / requested`` (``0.0``
    cuando no se solicitó nada). Las filas con predicción faltante o real
    faltante se excluyen de la evaluación y nunca fabrican números.
    """
    evaluated = [
        row
        for row in result.rows
        if row.prediction is not None and row.observed_sales is not None
    ]
    requested = result.requested
    available = result.available
    coverage = available / requested if requested else 0.0
    evaluated_count = len(evaluated)
    stockout_rows = sum(1 for row in evaluated if row.stockout_hours_6_22 > 0)

    if not evaluated:
        return ForecastMetrics(
            model=result.model,
            fold=fold,
            target=TARGET_LABEL,
            requested=requested,
            available=available,
            evaluated=0,
            coverage=coverage,
            stockout_rows=0,
            mae=None,
            rmse=None,
            wmape=None,
            bias=None,
            wmape_status=WMAPE_NO_EVALUABLE,
        )

    absolute_errors = [
        abs(row.prediction - row.observed_sales) for row in evaluated
    ]
    squared_errors = [error * error for error in absolute_errors]
    signed_errors = [row.prediction - row.observed_sales for row in evaluated]
    mae = math.fsum(absolute_errors) / evaluated_count
    rmse = math.sqrt(math.fsum(squared_errors) / evaluated_count)
    bias = math.fsum(signed_errors) / evaluated_count
    wmape_denominator = math.fsum(abs(row.observed_sales) for row in evaluated)
    if wmape_denominator == 0.0:
        wmape = None
        wmape_status = WMAPE_ZERO_ACTUALS
    else:
        wmape = math.fsum(absolute_errors) / wmape_denominator
        wmape_status = WMAPE_CALCULABLE

    return ForecastMetrics(
        model=result.model,
        fold=fold,
        target=TARGET_LABEL,
        requested=requested,
        available=available,
        evaluated=evaluated_count,
        coverage=coverage,
        stockout_rows=stockout_rows,
        mae=mae,
        rmse=rmse,
        wmape=wmape,
        bias=bias,
        wmape_status=wmape_status,
    )
