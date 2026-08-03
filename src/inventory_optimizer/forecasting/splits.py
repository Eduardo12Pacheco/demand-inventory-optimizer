"""Divisiones temporales para pronóstico de ventas observadas.

Convención de límites
---------------------
Cada límite es un día de calendario INCLUSIVO: una fila cuyo ``dt`` coincide con
un límite de un conjunto pertenece a ese conjunto. Los conjuntos se solicitan
con pares explícitos ``*_start`` / ``*_end``; un conjunto que no se solicita se
representa como una tupla vacía con límites ``None`` — la API nunca inventa
fechas ni horizontes ocultos.

Garantías
---------
* Las filas nunca se mezclan; cada conjunto se ordena por ``(product_id, dt)``.
* Las claves duplicadas ``(product_id, dt)`` lanzan :class:`DuplicateKeyError`;
  las filas nunca se deduplican en silencio.
* Las fechas de calendario faltantes dentro de los rangos solicitados se
  reportan (globalmente y por producto) y nunca se rellenan: nunca aparecen
  filas sintéticas.
* Los productos con muy poca historia de entrenamiento observada aparecen en
  registros ``insufficient_history`` que llevan el conteo/extensión observados y
  los mínimos requeridos.
* Las filas fuera de los rangos solicitados se excluyen y se cuentan en
  ``excluded_out_of_range``.
* Las particiones walk-forward de ventana expansiva mantienen el inicio de
  entrenamiento fijo mientras el final del entrenamiento se expande; la
  validación sigue inmediatamente, y ninguna fila futura de validación/prueba
  entra jamás al conjunto de entrenamiento de una partición. Reutilizar filas
  entre particiones de entrenamiento expansivas es esperado.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

from inventory_optimizer.forecasting._ordering import (
    DuplicateKeyError,
    product_sort_key,
    validate_unique_and_sort,
)
from inventory_optimizer.ingestion.fresh_retail import DailySourceRow


class SplittingError(Exception):
    """Clase base para errores de división temporal."""


class InvalidBoundaryOrderError(SplittingError):
    """Los límites de fecha solicitados o los conjuntos resultantes violan el orden estricto."""


@dataclass(frozen=True)
class InsufficientHistory:
    """Un producto cuya historia de entrenamiento observada está por debajo de los mínimos."""

    product_id: str | int
    observed_days: int
    observed_span_days: int
    required_days: int
    required_span_days: int


@dataclass(frozen=True)
class MissingDatesForProduct:
    """Fechas de calendario dentro del lapso solicitado sin fila para un producto."""

    product_id: str | int
    dates: tuple[date, ...]


@dataclass(frozen=True)
class TemporalSplit:
    """Una división temporal explícita con límites inclusivos y reporte completo.

    Un conjunto que no se solicitó es una tupla vacía con límites ``None``; los
    conjuntos ausentes son explícitos, nunca inventados.
    """

    train: tuple[DailySourceRow, ...]
    validation: tuple[DailySourceRow, ...]
    test: tuple[DailySourceRow, ...]
    train_start: date
    train_end: date
    validation_start: date | None = None
    validation_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    missing_train_dates: tuple[date, ...] = ()
    missing_validation_dates: tuple[date, ...] = ()
    missing_test_dates: tuple[date, ...] = ()
    missing_per_product: tuple[MissingDatesForProduct, ...] = ()
    insufficient_history: tuple[InsufficientHistory, ...] = ()
    excluded_out_of_range: int = 0


@dataclass(frozen=True)
class ExpandingWindowFold:
    """Una partición de ventana expansiva: inicio de entrenamiento fijo, final expansivo.

    ``split`` lleva los límites de fecha propios de la partición, sus conjuntos,
    los reportes de fechas faltantes y los registros de historia insuficiente.
    """

    fold_index: int
    split: TemporalSplit


def _calendar_days(start: date, end: date) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _missing_dates(
    rows_in_set: Sequence[DailySourceRow], start: date, end: date
) -> tuple[date, ...]:
    present = {row.dt for row in rows_in_set}
    return tuple(day for day in _calendar_days(start, end) if day not in present)


def _observed_span_days(rows: Sequence[DailySourceRow]) -> int:
    """Extensión de calendario (días inclusivos) de las filas dadas, 0 cuando está vacía."""
    if not rows:
        return 0
    return (max(row.dt for row in rows) - min(row.dt for row in rows)).days + 1


def _build_split(
    rows: Sequence[DailySourceRow],
    *,
    train_start: date,
    train_end: date,
    validation_start: date | None,
    validation_end: date | None,
    test_start: date | None,
    test_end: date | None,
    min_history_days: int,
    min_history_span_days: int,
) -> TemporalSplit:
    train = tuple(row for row in rows if train_start <= row.dt <= train_end)
    validation = (
        tuple(row for row in rows if validation_start <= row.dt <= validation_end)
        if validation_start is not None
        else ()
    )
    test = (
        tuple(row for row in rows if test_start <= row.dt <= test_end)
        if test_start is not None
        else ()
    )
    train_set = set(train)
    validation_set = set(validation)
    test_set = set(test)
    excluded_out_of_range = sum(
        1
        for row in rows
        if row not in train_set and row not in validation_set and row not in test_set
    )
    missing_train = _missing_dates(train, train_start, train_end)
    missing_validation = (
        _missing_dates(validation, validation_start, validation_end)
        if validation_start is not None
        else ()
    )
    missing_test = (
        _missing_dates(test, test_start, test_end) if test_start is not None else ()
    )

    span_starts = [train_start]
    span_ends = [train_end]
    if validation_start is not None:
        span_starts.append(validation_start)
        span_ends.append(validation_end)
    if test_start is not None:
        span_starts.append(test_start)
        span_ends.append(test_end)
    span_start = min(span_starts)
    span_end = max(span_ends)
    calendar = set(_calendar_days(span_start, span_end))

    in_split_rows = train + validation + test
    dates_by_product: dict[str | int, set[date]] = defaultdict(set)
    train_by_product: dict[str | int, list[DailySourceRow]] = defaultdict(list)
    for row in in_split_rows:
        dates_by_product[row.product_id].add(row.dt)
    for row in train:
        train_by_product[row.product_id].append(row)
    per_product: list[MissingDatesForProduct] = []
    insufficient: list[InsufficientHistory] = []
    for product_id in sorted(dates_by_product, key=product_sort_key):
        per_product.append(
            MissingDatesForProduct(
                product_id=product_id,
                dates=tuple(sorted(calendar - dates_by_product[product_id])),
            )
        )
        product_train = train_by_product[product_id]
        observed_days = len(product_train)
        observed_span_days = _observed_span_days(product_train)
        if observed_days < min_history_days or observed_span_days < min_history_span_days:
            insufficient.append(
                InsufficientHistory(
                    product_id=product_id,
                    observed_days=observed_days,
                    observed_span_days=observed_span_days,
                    required_days=min_history_days,
                    required_span_days=min_history_span_days,
                )
            )

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        missing_train_dates=missing_train,
        missing_validation_dates=missing_validation,
        missing_test_dates=missing_test,
        missing_per_product=tuple(per_product),
        insufficient_history=tuple(insufficient),
        excluded_out_of_range=excluded_out_of_range,
    )


def _validate_minimums(min_history_days: int, min_history_span_days: int) -> None:
    if min_history_days < 1 or min_history_span_days < 1:
        raise ValueError(
            "min_history_days and min_history_span_days must be >= 1, got "
            f"({min_history_days}, {min_history_span_days})."
        )


def split_temporal(
    rows: Iterable[DailySourceRow],
    *,
    train_start: date,
    train_end: date,
    validation_start: date | None = None,
    validation_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    min_history_days: int = 1,
    min_history_span_days: int = 1,
) -> TemporalSplit:
    """Particiona filas en train/validation/test con límites inclusivos explícitos.

    El orden temporal estricto ``max(train.dt) < min(validation.dt) <
    min(test.dt)`` se valida cuando los tres conjuntos están no vacíos; el
    orden de límites inválido se rechaza con
    :class:`InvalidBoundaryOrderError`. Las fechas de calendario faltantes
    dentro de los rangos solicitados se reportan, nunca se rellenan; las fechas
    faltantes por producto cubren todo el lapso de calendario solicitado para
    cada producto observado dentro de los rangos solicitados.
    """
    _validate_minimums(min_history_days, min_history_span_days)
    if train_end < train_start:
        raise InvalidBoundaryOrderError(
            f"train_end {train_end} is before train_start {train_start}."
        )
    if (validation_start is None) != (validation_end is None):
        raise ValueError("validation_start and validation_end must be given together.")
    if (test_start is None) != (test_end is None):
        raise ValueError("test_start and test_end must be given together.")
    if validation_start is not None:
        if validation_start <= train_end:
            raise InvalidBoundaryOrderError(
                f"validation_start {validation_start} must follow train_end "
                f"{train_end} strictly (both boundaries are inclusive)."
            )
        if validation_end < validation_start:
            raise InvalidBoundaryOrderError(
                f"validation_end {validation_end} is before validation_start "
                f"{validation_start}."
            )
    if test_start is not None:
        last_end = validation_end if validation_end is not None else train_end
        if test_start <= last_end:
            raise InvalidBoundaryOrderError(
                f"test_start {test_start} must follow {last_end} strictly "
                f"(both boundaries are inclusive)."
            )
        if test_end < test_start:
            raise InvalidBoundaryOrderError(
                f"test_end {test_end} is before test_start {test_start}."
            )

    ordered = validate_unique_and_sort(rows)
    split = _build_split(
        ordered,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        min_history_days=min_history_days,
        min_history_span_days=min_history_span_days,
    )
    if split.train and split.validation and split.test:
        max_train = max(row.dt for row in split.train)
        min_validation = min(row.dt for row in split.validation)
        min_test = min(row.dt for row in split.test)
        if not (max_train < min_validation < min_test):
            raise InvalidBoundaryOrderError(
                "Partitioned sets violate strict temporal ordering "
                f"max(train.dt)={max_train} < min(validation.dt)={min_validation} "
                f"< min(test.dt)={min_test}."
            )
    return split


def expanding_window_folds(
    rows: Iterable[DailySourceRow],
    *,
    start: date,
    initial_train_days: int,
    validation_days: int,
    folds: int,
    step_days: int,
    test_start: date | None = None,
    test_days: int | None = None,
    min_history_days: int = 1,
    min_history_span_days: int = 1,
) -> tuple[ExpandingWindowFold, ...]:
    """Construye particiones walk-forward de ventana expansiva con parámetros explícitos.

    La partición ``i`` (basada en 1) entrena sobre la ventana inclusiva
    ``[start, start + initial_train_days + (i - 1) * step_days - 1]``: el
    inicio de entrenamiento queda fijo mientras el final del entrenamiento se
    expande. La validación son los siguientes ``validation_days`` días de
    calendario inmediatamente después del final del entrenamiento de la
    partición. Un horizonte de prueba explícito opcional (``test_start`` más
    ``test_days`` inclusivos) debe seguir a la ventana de validación de la
    ÚLTIMA partición y es idéntico para cada partición. Cada partición expone
    sus propios límites de fecha y sus propios reportes de fechas faltantes e
    historia insuficiente.
    """
    _validate_minimums(min_history_days, min_history_span_days)
    if initial_train_days < 1 or validation_days < 1 or folds < 1 or step_days < 1:
        raise ValueError(
            "initial_train_days, validation_days, folds, and step_days must be >= 1, "
            f"got ({initial_train_days}, {validation_days}, {folds}, {step_days})."
        )
    if (test_start is None) != (test_days is None):
        raise ValueError("test_start and test_days must be given together.")
    if test_days is not None and test_days < 1:
        raise ValueError(f"test_days must be >= 1, got {test_days}.")

    ordered = validate_unique_and_sort(rows)
    last_train_end = start + timedelta(
        days=initial_train_days + (folds - 1) * step_days - 1
    )
    last_validation_end = last_train_end + timedelta(days=validation_days)
    if test_start is not None and test_start <= last_validation_end:
        raise InvalidBoundaryOrderError(
            f"test_start {test_start} must follow the last validation end "
            f"{last_validation_end} strictly (both boundaries are inclusive)."
        )

    built: list[ExpandingWindowFold] = []
    for fold_index in range(1, folds + 1):
        train_end = start + timedelta(
            days=initial_train_days + (fold_index - 1) * step_days - 1
        )
        validation_start = train_end + timedelta(days=1)
        validation_end = train_end + timedelta(days=validation_days)
        test_end = (
            test_start + timedelta(days=test_days - 1)
            if test_start is not None
            else None
        )
        split = _build_split(
            ordered,
            train_start=start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            min_history_days=min_history_days,
            min_history_span_days=min_history_span_days,
        )
        built.append(ExpandingWindowFold(fold_index=fold_index, split=split))
    return tuple(built)
