"""Líneas base de pronóstico deterministas y explicables sobre ventas observadas.

Cada línea base se ajusta EXCLUSIVAMENTE con las filas de entrenamiento pasadas
y predice solo filas objetivo futuras (por producto, estrictamente después de la
última fecha de entrenamiento del producto). Las filas objetivo nunca se usan
durante el ajuste, aunque lleven ``observed_sales`` para la evaluación posterior.
Las ventas observadas nunca se recortan, imputan, enmascaran ni re-etiquetan;
las predicciones no disponibles permanecen ``None`` con una razón explícita y
nunca se eliminan del resultado.

Líneas base
-----------
* ``naive`` — último ``sale_amount`` observado por producto.
* ``seasonal_naive`` — valor observado exactamente 7 días de calendario antes de
  la fecha objetivo, buscado solo en filas de entrenamiento (nunca de forma
  recursiva desde reales de validation/test).
* ``moving_average`` — media de los últimos ``window`` valores de entrenamiento
  observados por producto (usa todos los valores observados cuando la historia
  es más corta que la ventana).
* ``ses`` — suavizamiento exponencial simple con ``alpha`` configurable,
  ``level_t = alpha * x_t + (1 - alpha) * level_{t-1}`` sembrado con la primera
  observación de entrenamiento; los pronósticos provienen del nivel final.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Literal, Sequence

from inventory_optimizer.forecasting._ordering import validate_unique_and_sort
from inventory_optimizer.forecasting.targets import ObservationState, TargetRow
from inventory_optimizer.ingestion.fresh_retail import DailySourceRow

MODEL_NAIVE = "naive"
MODEL_SEASONAL_NAIVE = "seasonal_naive"
MODEL_MOVING_AVERAGE = "moving_average"
MODEL_SES = "ses"

PREDICTION_AVAILABLE = "available"
PREDICTION_UNAVAILABLE = "unavailable"

REASON_NO_HISTORY = "no training history for product"
REASON_NO_SEASONAL_LAG = "no seasonal lag observation in training"


class BaselineError(Exception):
    """Clase base para errores de líneas base de pronóstico."""


class TargetNotInFutureError(BaselineError):
    """Una fila objetivo cae en o antes de la última fecha de entrenamiento de su producto."""


@dataclass(frozen=True)
class ForecastRow:
    """Una predicción de modelo para un producto/fecha.

    ``observed_sales`` son las ventas observadas del objetivo (``None`` solo
    cuando no hay real adjunto); ``prediction`` es ``None`` cuando no está
    disponible, con ``unavailable_reason`` explicando por qué.
    """

    model: str
    product_id: str | int
    dt: date
    observed_sales: float | None
    prediction: float | None
    prediction_state: Literal["available", "unavailable"]
    unavailable_reason: str | None
    stockout_hours_6_22: int
    observation_state: ObservationState
    revision: str


@dataclass(frozen=True)
class BaselineForecastResult:
    """Forma de predicción común con contabilidad de cobertura explícita."""

    model: str
    rows: tuple[ForecastRow, ...]
    requested: int
    available: int
    unavailable: int
    coverage: float


def _check_future_targets(
    train_rows: Sequence[DailySourceRow], target_rows: Sequence[TargetRow]
) -> None:
    """Rechaza fechas objetivo en o antes de la última fecha de entrenamiento de su producto."""
    last_train: dict[str | int, date] = {}
    for row in train_rows:
        if row.product_id not in last_train or row.dt > last_train[row.product_id]:
            last_train[row.product_id] = row.dt
    offenders = [
        target
        for target in target_rows
        if target.product_id in last_train
        and target.dt <= last_train[target.product_id]
    ]
    if offenders:
        sample = ", ".join(
            f"{target.product_id!r}@{target.dt.isoformat()}" for target in offenders[:5]
        )
        raise TargetNotInFutureError(
            "Baselines predict only future target rows (strictly after each "
            f"product's last training date), got: {sample}."
        )


def _sorted_training_series(
    train_rows: Sequence[DailySourceRow],
) -> dict[str | int, list[DailySourceRow]]:
    series: dict[str | int, list[DailySourceRow]] = defaultdict(list)
    for row in train_rows:
        series[row.product_id].append(row)
    for rows in series.values():
        rows.sort(key=lambda row: row.dt)
    return dict(series)


def _build_result(
    model: str,
    target_rows: Sequence[TargetRow],
    predictions: Sequence[tuple[float | None, str | None]],
) -> BaselineForecastResult:
    rows = tuple(
        ForecastRow(
            model=model,
            product_id=target.product_id,
            dt=target.dt,
            observed_sales=target.observed_sales,
            prediction=prediction,
            prediction_state=(
                PREDICTION_AVAILABLE
                if prediction is not None
                else PREDICTION_UNAVAILABLE
            ),
            unavailable_reason=reason,
            stockout_hours_6_22=target.stockout_hours_6_22,
            observation_state=target.observation_state,
            revision=target.revision,
        )
        for target, (prediction, reason) in zip(target_rows, predictions)
    )
    requested = len(rows)
    available = sum(1 for row in rows if row.prediction is not None)
    return BaselineForecastResult(
        model=model,
        rows=rows,
        requested=requested,
        available=available,
        unavailable=requested - available,
        coverage=available / requested if requested else 0.0,
    )


def fit_naive(
    train: Iterable[DailySourceRow],
    targets: Iterable[TargetRow],
) -> BaselineForecastResult:
    """Predice cada fecha objetivo futura con el último valor de entrenamiento observado."""
    train_rows = validate_unique_and_sort(train)
    target_rows = validate_unique_and_sort(targets)
    _check_future_targets(train_rows, target_rows)
    last = {
        product_id: rows[-1].sale_amount
        for product_id, rows in _sorted_training_series(train_rows).items()
    }
    predictions = [
        (last[target.product_id], None)
        if target.product_id in last
        else (None, REASON_NO_HISTORY)
        for target in target_rows
    ]
    return _build_result(MODEL_NAIVE, target_rows, predictions)


def fit_seasonal_naive(
    train: Iterable[DailySourceRow],
    targets: Iterable[TargetRow],
) -> BaselineForecastResult:
    """Predice con el valor de entrenamiento exactamente 7 días de calendario antes del objetivo.

    La búsqueda del rezago se restringe solo a filas de entrenamiento; los
    reales de validation/test nunca se usan como entrada recursiva. Una
    observación de rezago faltante o un producto sin historia de entrenamiento
    producen una predicción no disponible explícita.
    """
    train_rows = validate_unique_and_sort(train)
    target_rows = validate_unique_and_sort(targets)
    _check_future_targets(train_rows, target_rows)
    lag = timedelta(days=7)
    products_with_history = {row.product_id for row in train_rows}
    lookup = {
        (row.product_id, row.dt): row.sale_amount for row in train_rows
    }
    predictions = []
    for target in target_rows:
        if target.product_id not in products_with_history:
            predictions.append((None, REASON_NO_HISTORY))
            continue
        key = (target.product_id, target.dt - lag)
        if key in lookup:
            predictions.append((lookup[key], None))
        else:
            predictions.append((None, REASON_NO_SEASONAL_LAG))
    return _build_result(MODEL_SEASONAL_NAIVE, target_rows, predictions)


def fit_moving_average(
    train: Iterable[DailySourceRow],
    targets: Iterable[TargetRow],
    *,
    window: int = 7,
) -> BaselineForecastResult:
    """Predice con la media de los últimos ``window`` valores de entrenamiento observados.

    Los productos con menos observaciones de entrenamiento que ``window`` usan
    todos sus valores observados; los productos sin historia de entrenamiento
    están explícitamente no disponibles.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    train_rows = validate_unique_and_sort(train)
    target_rows = validate_unique_and_sort(targets)
    _check_future_targets(train_rows, target_rows)
    means = {}
    for product_id, rows in _sorted_training_series(train_rows).items():
        recent = rows[-window:]
        means[product_id] = math.fsum(row.sale_amount for row in recent) / len(recent)
    predictions = [
        (means[target.product_id], None)
        if target.product_id in means
        else (None, REASON_NO_HISTORY)
        for target in target_rows
    ]
    return _build_result(MODEL_MOVING_AVERAGE, target_rows, predictions)


def fit_ses(
    train: Iterable[DailySourceRow],
    targets: Iterable[TargetRow],
    *,
    alpha: float = 0.3,
) -> BaselineForecastResult:
    """Ajusta suavizamiento exponencial simple por producto y predice el nivel final.

    ``alpha`` debe estar en ``[0, 1]``; ``alpha == 1.0`` colapsa a la última
    observación y ``alpha == 0.0`` conserva la primera observación. Los
    productos sin historia de entrenamiento están explícitamente no disponibles.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    train_rows = validate_unique_and_sort(train)
    target_rows = validate_unique_and_sort(targets)
    _check_future_targets(train_rows, target_rows)
    levels = {}
    for product_id, rows in _sorted_training_series(train_rows).items():
        level = rows[0].sale_amount
        for row in rows[1:]:
            level = alpha * row.sale_amount + (1.0 - alpha) * level
        levels[product_id] = level
    predictions = [
        (levels[target.product_id], None)
        if target.product_id in levels
        else (None, REASON_NO_HISTORY)
        for target in target_rows
    ]
    return _build_result(MODEL_SES, target_rows, predictions)


BASELINES: dict[str, Callable[..., BaselineForecastResult]] = {
    MODEL_NAIVE: fit_naive,
    MODEL_SEASONAL_NAIVE: fit_seasonal_naive,
    MODEL_MOVING_AVERAGE: fit_moving_average,
    MODEL_SES: fit_ses,
}


def fit_baseline(
    name: str,
    train: Iterable[DailySourceRow],
    targets: Iterable[TargetRow],
    **params,
) -> BaselineForecastResult:
    """Ajusta una línea base por nombre (``naive``, ``seasonal_naive``,
    ``moving_average``, ``ses``) con parámetros por palabra clave explícitos."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(BASELINES)}.")
    return BASELINES[name](train, targets, **params)
