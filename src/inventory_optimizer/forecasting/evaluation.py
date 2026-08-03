"""Ejecutor de evaluación de líneas base offline reproducible.

Construye cada partición explícita a través de ``split_temporal`` (todas las
comprobaciones de división existentes y el reporte de calidad de datos siguen
siendo autoritativos), proyecta SOLO la partición de evaluación configurada
(``validation`` o ``test``) en objetivos de ventas observadas, y evalúa cada
línea base configurada a través de ``evaluate_metrics`` usando el protocolo
versionado de historial de ajuste ``EVALUATION_PROTOCOL_VERSION =
"baseline-evaluation-v2"``:

* ``evaluation_partition="validation"`` — el historial de ajuste es SOLO
  ``split.train``; las filas de evaluación son ``split.validation`` (la
  validación es solo una comprobación).
* ``evaluation_partition="test"`` — el historial de ajuste es la combinación
  ordenada y con duplicados verificados ``split.train + split.validation``; las
  filas de evaluación son ``split.test``. Las filas de prueba NUNCA entran al
  historial de ajuste: las claves de ajuste y las de prueba son disjuntas y el
  historial de ajuste termina estrictamente antes de que comience la prueba
  (aplicado antes de cualquier ajuste).

Los parámetros de las líneas base permanecen explícitos y fijos desde
``BaselineEvaluationConfig``; nunca ocurre tuning, ranking ni selección de
mejor modelo. El :class:`EvaluationReport` tipado lleva una fila de resultado
determinista por partición/modelo (orden de particiones, luego orden de modelos
requeridos), estado de calidad de datos a nivel de partición, un sobre de
protocolo a nivel de reporte (``evaluation_protocol_version``,
``evaluation_partition``, ``fit_partition``, ``test_excluded_from_fit``,
``baseline_parameters_fixed``), y campos de ajuste por partición/resultado
(``fit_partition``, ``fit_row_count``, ``fit_history_start``/``fit_history_end``,
``test_excluded_from_fit``, ``insufficient_fit_history`` derivado de las filas de
ajuste reales y los mínimos configurados), y se serializa en JSON determinista y
sin datos crudos usando solo la stdlib. El reporte nunca contiene objetos
``DailySourceRow``, vectores horarios ni filas completas de predicción; el
reporte de productos permanece en la etiqueta agregada ``product_group``
configurada más los conteos y los ids de producto con historia insuficiente.
``observed_sales`` sigue siendo el único objetivo; los metadatos de quiebre son
solo diagnósticos y la demanda latente nunca se estima. Ningún ganador se
selecciona ni se ranquea automáticamente.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal, Sequence

from inventory_optimizer.forecasting._ordering import (
    product_sort_key,
    validate_unique_and_sort,
)
from inventory_optimizer.forecasting.baselines import (
    BASELINES,
    MODEL_MOVING_AVERAGE,
    MODEL_NAIVE,
    MODEL_SEASONAL_NAIVE,
    MODEL_SES,
    fit_baseline,
)
from inventory_optimizer.forecasting.metrics import (
    WMAPEStatus,
    evaluate_metrics,
)
from inventory_optimizer.forecasting.splits import (
    InsufficientHistory,
    MissingDatesForProduct,
    split_temporal,
)
from inventory_optimizer.forecasting.targets import project_targets
from inventory_optimizer.ingestion.fresh_retail import DailySourceRow

EVALUATION_PARTITION_VALIDATION = "validation"
EVALUATION_PARTITION_TEST = "test"
EVALUATION_PARTITIONS = (EVALUATION_PARTITION_VALIDATION, EVALUATION_PARTITION_TEST)
"""Particiones de evaluación soportadas: validation o test (nunca train)."""

EVALUATION_PROTOCOL_VERSION = "baseline-evaluation-v2"
"""Identificador del protocolo de evaluación versionado.

``baseline-evaluation-v2`` corrige la regla del historial de ajuste: la
evaluación de validación ajusta SOLO ``split.train``; la evaluación de prueba
ajusta la combinación ordenada y con duplicados verificados
``split.train + split.validation``. Las filas de prueba nunca entran al
historial de ajuste, los parámetros de las líneas base están fijos (sin tuning),
y cada reporte expone su identidad de protocolo y sus campos de ajuste por
partición.
"""

FIT_PARTITION_TRAIN = "train"
FIT_PARTITION_TRAIN_PLUS_VALIDATION = "train+validation"
"""Etiquetas de partición de ajuste usadas por el protocolo versionado (ver
:data:`EVALUATION_PROTOCOL_VERSION`).

``fit_partition`` nombra las particiones del protocolo usadas para el ajuste:
``"train"`` para la evaluación de validación y ``"train+validation"`` para la
evaluación de prueba. Las filas REALES usadas se describen por
partición/resultado con ``fit_row_count`` y
``fit_history_start``/``fit_history_end``.
"""

EvaluationPartition = Literal["validation", "test"]

REQUIRED_MODEL_ORDER = (
    MODEL_NAIVE,
    MODEL_SEASONAL_NAIVE,
    MODEL_MOVING_AVERAGE,
    MODEL_SES,
)
"""Las cuatro líneas base requeridas en orden de reporte determinista."""


def _require_int_at_least(value: object, name: str, minimum: int = 1) -> None:
    """Requiere un int no bool >= minimum, o lanza un ValueError claro."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be a non-bool int >= {minimum}, got {value!r}."
        )


def _require_alpha(value: object) -> None:
    """Requiere un número no bool en [0, 1], o lanza un ValueError claro."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"ses_alpha must be a non-bool number in [0, 1], got {value!r}."
        )


@dataclass(frozen=True)
class FoldEvaluationConfig:
    """Una partición temporal explícita, convertida a través de ``split_temporal``.

    Los límites son días de calendario inclusivos. ``fold`` debe ser una cadena
    no vacía o un int no bool; los pares de límites de validation/test deben
    darse juntos; los mínimos de historia deben ser ints no bool >= 1. Una
    partición puede omitir la partición que no se evalúa, lo que produce un
    estado explícito de no evaluable en lugar de un error. El ORDEN de los
    límites se delega a ``split_temporal`` y sigue siendo autoritativo allí.
    """

    fold: str | int
    train_start: date
    train_end: date
    validation_start: date | None = None
    validation_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    min_history_days: int = 1
    min_history_span_days: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.fold, bool) or not isinstance(self.fold, (str, int)):
            raise ValueError(
                "fold identifier must be a non-empty string or a non-bool int, "
                f"got {self.fold!r}."
            )
        if isinstance(self.fold, str):
            label = self.fold.strip()
            if not label:
                raise ValueError(
                    f"fold identifier must be a non-empty string, got {self.fold!r}."
                )
            object.__setattr__(self, "fold", label)
        if (self.validation_start is None) != (self.validation_end is None):
            raise ValueError("validation_start and validation_end must be given together.")
        if (self.test_start is None) != (self.test_end is None):
            raise ValueError("test_start and test_end must be given together.")
        _require_int_at_least(self.min_history_days, "min_history_days")
        _require_int_at_least(self.min_history_span_days, "min_history_span_days")

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": (
                self.validation_start.isoformat() if self.validation_start is not None else None
            ),
            "validation_end": (
                self.validation_end.isoformat() if self.validation_end is not None else None
            ),
            "test_start": self.test_start.isoformat() if self.test_start is not None else None,
            "test_end": self.test_end.isoformat() if self.test_end is not None else None,
            "min_history_days": self.min_history_days,
            "min_history_span_days": self.min_history_span_days,
        }


@dataclass(frozen=True)
class EvaluationSplitConfig:
    """Lista de particiones explícita y no vacía más la partición evaluada y su etiqueta.

    Las particiones deben ser instancias de ``FoldEvaluationConfig`` con
    identificadores de partición únicos (los duplicados harían ambiguas las
    filas del reporte) y conservan su orden de entrada para el orden
    determinista del reporte.
    """

    folds: tuple[FoldEvaluationConfig, ...]
    evaluation_partition: EvaluationPartition = EVALUATION_PARTITION_VALIDATION
    product_group: str = "all_products"

    def __post_init__(self) -> None:
        try:
            folds = tuple(self.folds)
        except TypeError as exc:
            raise ValueError(
                "folds must be an iterable of FoldEvaluationConfig instances."
            ) from exc
        if not folds:
            raise ValueError(
                "folds must not be empty; provide at least one FoldEvaluationConfig."
            )
        non_folds = [
            type(entry).__name__
            for entry in folds
            if not isinstance(entry, FoldEvaluationConfig)
        ]
        if non_folds:
            raise ValueError(
                f"folds must contain only FoldEvaluationConfig instances, got: "
                f"{non_folds}."
            )
        fold_ids = [entry.fold for entry in folds]
        duplicates = sorted(
            {fold_id for fold_id in fold_ids if fold_ids.count(fold_id) > 1},
            key=product_sort_key,
        )
        if duplicates:
            raise ValueError(
                f"duplicate fold identifiers are not allowed (report rows would "
                f"be ambiguous): {duplicates}."
            )
        if self.evaluation_partition not in EVALUATION_PARTITIONS:
            raise ValueError(
                f"evaluation_partition must be one of {list(EVALUATION_PARTITIONS)}, "
                f"got {self.evaluation_partition!r}."
            )
        if not isinstance(self.product_group, str):
            raise ValueError(
                f"product_group must be a string label, got {self.product_group!r}."
            )
        label = self.product_group.strip()
        if not label:
            raise ValueError(
                f"product_group must be a non-empty label, got {self.product_group!r}."
            )
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "product_group", label)

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_partition": self.evaluation_partition,
            "product_group": self.product_group,
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True)
class BaselineEvaluationConfig:
    """Las cuatro líneas base requeridas con parámetros explícitos y validados.

    ``models`` debe ser un iterable de CADENAS que contenga exactamente
    ``naive``, ``seasonal_naive``, ``moving_average`` y ``ses`` (se rechazan
    entradas no textuales, duplicados, nombres desconocidos y nombres requeridos
    faltantes) y se normaliza al orden canónico requerido para el reporte
    determinista. ``moving_average_window`` debe ser un int no bool >= 1 y
    ``ses_alpha`` un número no bool en ``[0, 1]``. No hay selección automática
    de mejor modelo.
    """

    models: tuple[str, ...] = REQUIRED_MODEL_ORDER
    moving_average_window: int = 7
    ses_alpha: float = 0.3

    def __post_init__(self) -> None:
        _require_int_at_least(self.moving_average_window, "moving_average_window")
        _require_alpha(self.ses_alpha)
        if isinstance(self.models, str):
            raise ValueError(
                "models must be a tuple/list of baseline model names, "
                "not a single string."
            )
        try:
            models = tuple(self.models)
        except TypeError as exc:
            raise ValueError(
                f"models must be an iterable of baseline model names, got "
                f"{type(self.models).__name__}."
            ) from exc
        if not models:
            raise ValueError("models must not be empty.")
        non_strings = [name for name in models if not isinstance(name, str)]
        if non_strings:
            raise ValueError(
                f"models must contain only string entries, got non-string "
                f"entries: {non_strings!r}."
            )
        unknown = sorted({name for name in models if name not in BASELINES})
        if unknown:
            raise ValueError(
                f"unknown baseline models: {unknown}; choose from {list(REQUIRED_MODEL_ORDER)}."
            )
        duplicates = sorted({name for name in models if models.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate baseline models: {duplicates}.")
        missing = [name for name in REQUIRED_MODEL_ORDER if name not in models]
        if missing:
            raise ValueError(
                f"missing required baseline models: {missing}; all of "
                f"{list(REQUIRED_MODEL_ORDER)} are required."
            )
        object.__setattr__(self, "models", REQUIRED_MODEL_ORDER)

    def to_dict(self) -> dict[str, object]:
        return {
            "models": list(self.models),
            "moving_average_window": self.moving_average_window,
            "ses_alpha": self.ses_alpha,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """Identidad completa de una corrida de evaluación: fuente, revisión, evaluación, divisiones, líneas base.

    Las cadenas de identificación deben ser no vacías; ``split_config`` y
    ``baseline_config`` deben ser los objetos de configuración tipados (los
    tipos incorrectos fallan claramente en la construcción).
    """

    source_id: str
    dataset_revision: str
    evaluation_id: str
    split_config: EvaluationSplitConfig
    baseline_config: BaselineEvaluationConfig

    def __post_init__(self) -> None:
        for field_name in ("source_id", "dataset_revision", "evaluation_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string, got {value!r}."
                )
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.split_config, EvaluationSplitConfig):
            raise ValueError(
                "split_config must be an EvaluationSplitConfig instance, got "
                f"{type(self.split_config).__name__}."
            )
        if not isinstance(self.baseline_config, BaselineEvaluationConfig):
            raise ValueError(
                "baseline_config must be a BaselineEvaluationConfig instance, got "
                f"{type(self.baseline_config).__name__}."
            )

    def to_dict(self) -> dict[str, object]:
        """Instantánea de configuración determinista y segura para JSON (fechas ISO)."""
        return {
            "source_id": self.source_id,
            "dataset_revision": self.dataset_revision,
            "evaluation_id": self.evaluation_id,
            "evaluation_partition": self.split_config.evaluation_partition,
            "product_group": self.split_config.product_group,
            "folds": [fold.to_dict() for fold in self.split_config.folds],
            "models": list(self.baseline_config.models),
            "moving_average_window": self.baseline_config.moving_average_window,
            "ses_alpha": self.baseline_config.ses_alpha,
        }


@dataclass(frozen=True)
class EvaluationProtocol:
    """Sobre de protocolo a nivel de reporte para una corrida de evaluación.

    ``fit_partition`` nombra las particiones del protocolo usadas para el ajuste
    (``"train"`` para la evaluación de validación, ``"train+validation"`` para
    la de prueba); las filas REALES se describen por partición/resultado con
    ``fit_row_count`` y ``fit_history_start``/``fit_history_end``.
    ``test_excluded_from_fit`` y ``baseline_parameters_fixed`` son invariantes
    del protocolo versionado y siempre son ``True``.
    """

    evaluation_protocol_version: str = EVALUATION_PROTOCOL_VERSION
    evaluation_partition: EvaluationPartition = EVALUATION_PARTITION_VALIDATION
    fit_partition: str = FIT_PARTITION_TRAIN
    test_excluded_from_fit: bool = True
    baseline_parameters_fixed: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "evaluation_partition": self.evaluation_partition,
            "fit_partition": self.fit_partition,
            "test_excluded_from_fit": self.test_excluded_from_fit,
            "baseline_parameters_fixed": self.baseline_parameters_fixed,
        }


@dataclass(frozen=True)
class FoldEvaluationStatus:
    """Reporte de calidad de datos a nivel de partición: límites, conteos y reportes de división.

    No lleva filas crudas: solo conteos, fechas de calendario faltantes
    (globalmente y por producto vía :class:`MissingDatesForProduct`), registros
    de historia insuficiente y el conteo fuera de rango. Los campos de ajuste
    del protocolo versionado (``fit_partition``, ``fit_row_count``,
    ``fit_history_start``/``fit_history_end``, ``test_excluded_from_fit``,
    ``insufficient_fit_history``) describen las filas de ajuste REALES usadas.
    """

    fold: str | int
    train_start: date
    train_end: date
    validation_start: date | None
    validation_end: date | None
    test_start: date | None
    test_end: date | None
    train_rows: int
    validation_rows: int
    test_rows: int
    evaluation_rows: int
    evaluable: bool
    missing_train_dates: tuple[date, ...]
    missing_validation_dates: tuple[date, ...]
    missing_test_dates: tuple[date, ...]
    missing_per_product: tuple[MissingDatesForProduct, ...]
    insufficient_history: tuple[InsufficientHistory, ...]
    excluded_out_of_range: int
    evaluation_protocol_version: str = EVALUATION_PROTOCOL_VERSION
    fit_partition: str = FIT_PARTITION_TRAIN
    fit_row_count: int = 0
    fit_history_start: date | None = None
    fit_history_end: date | None = None
    test_excluded_from_fit: bool = True
    insufficient_fit_history: tuple[InsufficientHistory, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": (
                self.validation_start.isoformat() if self.validation_start is not None else None
            ),
            "validation_end": (
                self.validation_end.isoformat() if self.validation_end is not None else None
            ),
            "test_start": self.test_start.isoformat() if self.test_start is not None else None,
            "test_end": self.test_end.isoformat() if self.test_end is not None else None,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "evaluation_rows": self.evaluation_rows,
            "evaluable": self.evaluable,
            "missing_train_dates": [day.isoformat() for day in self.missing_train_dates],
            "missing_validation_dates": [
                day.isoformat() for day in self.missing_validation_dates
            ],
            "missing_test_dates": [day.isoformat() for day in self.missing_test_dates],
            "missing_per_product": [
                {
                    "product_id": entry.product_id,
                    "dates": [day.isoformat() for day in entry.dates],
                }
                for entry in self.missing_per_product
            ],
            "insufficient_history": [
                {
                    "product_id": entry.product_id,
                    "observed_days": entry.observed_days,
                    "observed_span_days": entry.observed_span_days,
                    "required_days": entry.required_days,
                    "required_span_days": entry.required_span_days,
                }
                for entry in self.insufficient_history
            ],
            "excluded_out_of_range": self.excluded_out_of_range,
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "fit_partition": self.fit_partition,
            "fit_row_count": self.fit_row_count,
            "fit_history_start": (
                self.fit_history_start.isoformat()
                if self.fit_history_start is not None
                else None
            ),
            "fit_history_end": (
                self.fit_history_end.isoformat()
                if self.fit_history_end is not None
                else None
            ),
            "test_excluded_from_fit": self.test_excluded_from_fit,
            "insufficient_fit_history": [
                {
                    "product_id": entry.product_id,
                    "observed_days": entry.observed_days,
                    "observed_span_days": entry.observed_span_days,
                    "required_days": entry.required_days,
                    "required_span_days": entry.required_span_days,
                }
                for entry in self.insufficient_fit_history
            ],
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Una fila métrica por partición/modelo, totalmente reproducible y autocontenida.

    ``target`` es siempre exactamente ``observed_sales``; las métricas provienen
    de ``evaluate_metrics`` sobre pares evaluados solamente. ``config`` es la
    instantánea de configuración tipada compartida que reproduce esta fila. Los
    campos de historial de ajuste (``fit_partition``, ``fit_row_count``,
    ``fit_history_start``/``fit_history_end``, ``test_excluded_from_fit``,
    ``insufficient_fit_history``) reflejan los hechos de ajuste del protocolo
    versionado de la partición. No se incluyen filas crudas de la fuente,
    vectores horarios ni filas de predicción.
    """

    evaluation_id: str
    source_id: str
    dataset_revision: str
    model: str
    fold: str | int
    evaluation_partition: EvaluationPartition
    product_group: str
    train_start: date
    train_end: date
    validation_start: date | None
    validation_end: date | None
    test_start: date | None
    test_end: date | None
    target: str
    requested: int
    available: int
    unavailable: int
    evaluated: int
    coverage: float
    stockout_rows: int
    mae: float | None
    rmse: float | None
    wmape: float | None
    bias: float | None
    wmape_status: WMAPEStatus
    config: EvaluationConfig
    evaluation_protocol_version: str = EVALUATION_PROTOCOL_VERSION
    fit_partition: str = FIT_PARTITION_TRAIN
    fit_row_count: int = 0
    fit_history_start: date | None = None
    fit_history_end: date | None = None
    test_excluded_from_fit: bool = True
    insufficient_fit_history: tuple[InsufficientHistory, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "source_id": self.source_id,
            "dataset_revision": self.dataset_revision,
            "model": self.model,
            "fold": self.fold,
            "evaluation_partition": self.evaluation_partition,
            "product_group": self.product_group,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": (
                self.validation_start.isoformat() if self.validation_start is not None else None
            ),
            "validation_end": (
                self.validation_end.isoformat() if self.validation_end is not None else None
            ),
            "test_start": self.test_start.isoformat() if self.test_start is not None else None,
            "test_end": self.test_end.isoformat() if self.test_end is not None else None,
            "target": self.target,
            "requested": self.requested,
            "available": self.available,
            "unavailable": self.unavailable,
            "evaluated": self.evaluated,
            "coverage": self.coverage,
            "stockout_rows": self.stockout_rows,
            "mae": self.mae,
            "rmse": self.rmse,
            "wmape": self.wmape,
            "bias": self.bias,
            "wmape_status": self.wmape_status,
            "configuration": self.config.to_dict(),
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "fit_partition": self.fit_partition,
            "fit_row_count": self.fit_row_count,
            "fit_history_start": (
                self.fit_history_start.isoformat()
                if self.fit_history_start is not None
                else None
            ),
            "fit_history_end": (
                self.fit_history_end.isoformat()
                if self.fit_history_end is not None
                else None
            ),
            "test_excluded_from_fit": self.test_excluded_from_fit,
            "insufficient_fit_history": [
                {
                    "product_id": entry.product_id,
                    "observed_days": entry.observed_days,
                    "observed_span_days": entry.observed_span_days,
                    "required_days": entry.required_days,
                    "required_span_days": entry.required_span_days,
                }
                for entry in self.insufficient_fit_history
            ],
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Reporte de evaluación determinista: sobre de protocolo, configuración, estados y filas.

    ``protocol`` es el sobre de protocolo a nivel de reporte
    (:class:`EvaluationProtocol`); los hechos de ajuste por partición viven en
    cada :class:`FoldEvaluationStatus` y :class:`EvaluationResult`.
    """

    config: EvaluationConfig
    protocol: EvaluationProtocol
    results: tuple[EvaluationResult, ...]
    fold_statuses: tuple[FoldEvaluationStatus, ...]

    def to_dict(self) -> dict[str, object]:
        """Diccionario determinista y seguro para JSON (fechas ISO, tuplas como listas)."""
        return {
            "config": self.config.to_dict(),
            "protocol": self.protocol.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "fold_statuses": [status.to_dict() for status in self.fold_statuses],
        }

    def to_json(self) -> str:
        """Serialización JSON estable: cadena idéntica para reportes idénticos."""
        return json.dumps(self.to_dict(), sort_keys=True)


def _validate_source_revisions(
    rows: Sequence[DailySourceRow], dataset_revision: str
) -> None:
    """Rechaza revisiones de fuente mezcladas o que no coinciden en lugar de mezclarlas en silencio."""
    distinct = sorted({row.revision for row in rows})
    if len(distinct) > 1:
        raise ValueError(
            "Mixed source revisions are not allowed: got "
            + ", ".join(repr(revision) for revision in distinct)
            + f"; every row must equal config.dataset_revision {dataset_revision!r}."
        )
    if distinct and distinct[0] != dataset_revision:
        raise ValueError(
            f"Source row revision {distinct[0]!r} does not match "
            f"config.dataset_revision {dataset_revision!r}."
        )


def _fit_partition_for(partition: EvaluationPartition) -> str:
    """Etiqueta de partición de ajuste del protocolo para una partición de evaluación."""
    if partition == EVALUATION_PARTITION_VALIDATION:
        return FIT_PARTITION_TRAIN
    return FIT_PARTITION_TRAIN_PLUS_VALIDATION


def _derive_insufficient_fit_history(
    fit_rows: Sequence[DailySourceRow],
    evaluation_rows: Sequence[DailySourceRow],
    min_history_days: int,
    min_history_span_days: int,
) -> tuple[InsufficientHistory, ...]:
    """Productos con historia insuficiente sobre las filas de ajuste REALES.

    Un producto se considera cuando aparece en el historial de ajuste o en las
    filas de la partición de evaluación (productos que necesitan predicciones).
    La suficiencia usa los mínimos configurados de la partición sobre las filas
    de ajuste solamente — la misma regla que el reporte basado en train de
    ``split_temporal``, aplicada al conjunto de ajuste real.
    """
    by_product: dict[str | int, list[DailySourceRow]] = defaultdict(list)
    for row in fit_rows:
        by_product[row.product_id].append(row)
    products = set(by_product) | {row.product_id for row in evaluation_rows}
    insufficient: list[InsufficientHistory] = []
    for product_id in sorted(products, key=product_sort_key):
        rows = by_product[product_id]
        observed_days = len(rows)
        observed_span_days = (
            (max(row.dt for row in rows) - min(row.dt for row in rows)).days + 1
            if rows
            else 0
        )
        if (
            observed_days < min_history_days
            or observed_span_days < min_history_span_days
        ):
            insufficient.append(
                InsufficientHistory(
                    product_id=product_id,
                    observed_days=observed_days,
                    observed_span_days=observed_span_days,
                    required_days=min_history_days,
                    required_span_days=min_history_span_days,
                )
            )
    return tuple(insufficient)


def _check_fit_and_test_disjoint(
    fit_rows: Sequence[DailySourceRow],
    test_rows: Sequence[DailySourceRow],
    fold: str | int,
) -> None:
    """Aplica el invariante del protocolo: las filas de prueba nunca entran al ajuste.

    Las claves de ajuste y de prueba deben ser disjuntas, y el historial de
    ajuste debe terminar antes de que la prueba comience cuando la prueba se
    evalúa. La maquinaria de división ya garantiza esto vía límites ordenados;
    esta comprobación hace explícito el invariante del protocolo antes de que
    ocurra cualquier ajuste.
    """
    fit_keys = {(row.product_id, row.dt) for row in fit_rows}
    test_keys = {(row.product_id, row.dt) for row in test_rows}
    overlap = sorted(
        fit_keys & test_keys,
        key=lambda key: (product_sort_key(key[0]), key[1]),
    )
    if overlap:
        sample = ", ".join(
            f"{key[0]!r}@{key[1].isoformat()}" for key in overlap[:5]
        )
        raise ValueError(
            "Evaluation protocol violation: test rows must never enter fit "
            f"history (fold {fold!r}), got {len(overlap)} overlapping "
            f"key(s): {sample}."
        )
    if fit_rows and test_rows:
        max_fit = max(row.dt for row in fit_rows)
        min_test = min(row.dt for row in test_rows)
        if max_fit >= min_test:
            raise ValueError(
                "Evaluation protocol violation: fit history must end before "
                f"test starts (fold {fold!r}), got fit end "
                f"{max_fit.isoformat()} and test start {min_test.isoformat()}."
            )


def run_baseline_evaluation(
    rows: Iterable[DailySourceRow], config: EvaluationConfig
) -> EvaluationReport:
    """Ejecuta cada línea base configurada en cada partición y devuelve un reporte tipado.

    Flujo por partición: construir la división temporal a través de
    ``split_temporal``, proyectar SOLO la partición de evaluación configurada
    como objetivos, y ajustar cada línea base (ventana de promedio móvil y alpha
    de SES pasados explícitamente) usando el historial de ajuste del protocolo
    versionado: ``split.train`` solamente para la evaluación de validación; la
    combinación ordenada y con duplicados verificados
    ``split.train + split.validation`` para la evaluación de prueba. Las filas
    de prueba nunca entran al historial de ajuste (claves de ajuste/prueba
    disjuntas, el ajuste termina antes de que la prueba comience — aplicado).
    Los resultados se ordenan por partición y luego por el orden de modelos
    requerido.
    """
    ordered = validate_unique_and_sort(rows)
    _validate_source_revisions(ordered, config.dataset_revision)
    partition = config.split_config.evaluation_partition
    fit_partition = _fit_partition_for(partition)
    results: list[EvaluationResult] = []
    fold_statuses: list[FoldEvaluationStatus] = []

    for fold_config in config.split_config.folds:
        split = split_temporal(
            ordered,
            train_start=fold_config.train_start,
            train_end=fold_config.train_end,
            validation_start=fold_config.validation_start,
            validation_end=fold_config.validation_end,
            test_start=fold_config.test_start,
            test_end=fold_config.test_end,
            min_history_days=fold_config.min_history_days,
            min_history_span_days=fold_config.min_history_span_days,
        )
        if partition == EVALUATION_PARTITION_VALIDATION:
            partition_rows = split.validation
            fit_rows = validate_unique_and_sort(split.train)
        else:
            partition_rows = split.test
            fit_rows = validate_unique_and_sort(
                list(split.train) + list(split.validation)
            )
            _check_fit_and_test_disjoint(fit_rows, split.test, fold_config.fold)
        fit_row_count = len(fit_rows)
        fit_history_start = min(row.dt for row in fit_rows) if fit_rows else None
        fit_history_end = max(row.dt for row in fit_rows) if fit_rows else None
        insufficient_fit_history = _derive_insufficient_fit_history(
            fit_rows,
            partition_rows,
            fold_config.min_history_days,
            fold_config.min_history_span_days,
        )
        targets = project_targets(partition_rows)
        fold_statuses.append(
            FoldEvaluationStatus(
                fold=fold_config.fold,
                train_start=fold_config.train_start,
                train_end=fold_config.train_end,
                validation_start=fold_config.validation_start,
                validation_end=fold_config.validation_end,
                test_start=fold_config.test_start,
                test_end=fold_config.test_end,
                train_rows=len(split.train),
                validation_rows=len(split.validation),
                test_rows=len(split.test),
                evaluation_rows=len(partition_rows),
                evaluable=bool(partition_rows),
                missing_train_dates=split.missing_train_dates,
                missing_validation_dates=split.missing_validation_dates,
                missing_test_dates=split.missing_test_dates,
                missing_per_product=split.missing_per_product,
                insufficient_history=split.insufficient_history,
                excluded_out_of_range=split.excluded_out_of_range,
                evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
                fit_partition=fit_partition,
                fit_row_count=fit_row_count,
                fit_history_start=fit_history_start,
                fit_history_end=fit_history_end,
                test_excluded_from_fit=True,
                insufficient_fit_history=insufficient_fit_history,
            )
        )
        for model in config.baseline_config.models:
            params: dict[str, int | float] = {}
            if model == MODEL_MOVING_AVERAGE:
                params = {"window": config.baseline_config.moving_average_window}
            elif model == MODEL_SES:
                params = {"alpha": config.baseline_config.ses_alpha}
            forecast = fit_baseline(model, fit_rows, targets, **params)
            metrics = evaluate_metrics(forecast, fold=fold_config.fold)
            results.append(
                EvaluationResult(
                    evaluation_id=config.evaluation_id,
                    source_id=config.source_id,
                    dataset_revision=config.dataset_revision,
                    model=model,
                    fold=fold_config.fold,
                    evaluation_partition=partition,
                    product_group=config.split_config.product_group,
                    train_start=fold_config.train_start,
                    train_end=fold_config.train_end,
                    validation_start=fold_config.validation_start,
                    validation_end=fold_config.validation_end,
                    test_start=fold_config.test_start,
                    test_end=fold_config.test_end,
                    target=metrics.target,
                    requested=metrics.requested,
                    available=metrics.available,
                    unavailable=metrics.requested - metrics.available,
                    evaluated=metrics.evaluated,
                    coverage=metrics.coverage,
                    stockout_rows=metrics.stockout_rows,
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    wmape=metrics.wmape,
                    bias=metrics.bias,
                    wmape_status=metrics.wmape_status,
                    config=config,
                    evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
                    fit_partition=fit_partition,
                    fit_row_count=fit_row_count,
                    fit_history_start=fit_history_start,
                    fit_history_end=fit_history_end,
                    test_excluded_from_fit=True,
                    insufficient_fit_history=insufficient_fit_history,
                )
            )

    return EvaluationReport(
        config=config,
        protocol=EvaluationProtocol(
            evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
            evaluation_partition=partition,
            fit_partition=fit_partition,
            test_excluded_from_fit=True,
            baseline_parameters_fixed=True,
        ),
        results=tuple(results),
        fold_statuses=tuple(fold_statuses),
    )


__all__ = [
    "BaselineEvaluationConfig",
    "EVALUATION_PARTITIONS",
    "EVALUATION_PARTITION_TEST",
    "EVALUATION_PARTITION_VALIDATION",
    "EVALUATION_PROTOCOL_VERSION",
    "EvaluationConfig",
    "EvaluationPartition",
    "EvaluationProtocol",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSplitConfig",
    "FIT_PARTITION_TRAIN",
    "FIT_PARTITION_TRAIN_PLUS_VALIDATION",
    "FoldEvaluationConfig",
    "FoldEvaluationStatus",
    "REQUIRED_MODEL_ORDER",
    "run_baseline_evaluation",
]
