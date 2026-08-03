"""Lector puro del reporte de benchmark acotado congelado (visor de demo).

Este módulo está deliberadamente libre de dependencias (solo ``json`` de la
stdlib + ``pathlib``) y aislado de los ejecutores de ingesta/evaluación y de los
conjuntos de datos. Lee exactamente un archivo de reporte fijo, relativo al
repositorio, valida su esquema y expone resúmenes agregados inmutables/planos
con orden determinista de particiones/modelos. Nunca lee el reporte v1 de
diagnóstico, nunca realiza acceso a la red y nunca expone filas crudas ni
predicciones individuales.

El visor falla cerrado: cualquier campo requerido faltante o malformado lanza
:class:`BenchmarkReportError` con un mensaje claro en lugar de sustituir un
valor por defecto que podría falsear el benchmark. Los valores opcionales
(``bias``, o cualquier métrica que sea ``None``) se conservan como ``None`` y
nunca se inventan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# La única ruta de reporte en la demo: relativa al repositorio, sin controles
# de carga/ruta/URL en ninguna parte de la app, y nunca el archivo v1 de diagnóstico.
REPORT_RELATIVE_PATH = "data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json"
REPORT_PATH = Path(__file__).resolve().parents[3] / REPORT_RELATIVE_PATH

_REPORT_FILE_NAME = "freshretailnet-50k-bounded-real-1000-v2.json"
_SCHEMA_NAME = "inventory_optimizer.bounded_real_evaluation/v1"

_TOP_LEVEL_STRINGS = (
    "dataset",
    "dataset_revision",
    "evaluation_id",
    "evaluation_mode",
    "evaluation_partition",
    "evaluation_protocol_version",
    "fit_partition",
    "schema",
)
_TOP_LEVEL_BOOLS = ("baseline_parameters_fixed", "test_excluded_from_fit", "bounded_real_evaluation")
_TOP_LEVEL_INTS = ("max_rows", "rows_loaded")
_TOP_LEVEL_LISTS = ("models", "results", "limitations", "warnings", "fold_statuses")
_TOP_LEVEL_DICTS = ("protocol", "coverage", "fold_derivation", "seasonal_naive_by_fold")

_PROTOCOL_KEYS = (
    "evaluation_protocol_version",
    "evaluation_partition",
    "fit_partition",
    "test_excluded_from_fit",
    "baseline_parameters_fixed",
)
_FOLD_KEYS = (
    "fold",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
)
_FOLD_STATUS_KEYS = (
    "fold",
    "evaluable",
    "evaluation_rows",
    "fit_history_start",
    "fit_history_end",
    "fit_partition",
    "fit_row_count",
    "test_excluded_from_fit",
    "test_rows",
    "train_rows",
    "validation_rows",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "evaluation_protocol_version",
)
_RESULT_KEYS = (
    "model",
    "fold",
    "mae",
    "rmse",
    "wmape",
    "coverage",
    "available",
    "requested",
    "evaluated",
    "stockout_rows",
    "fit_partition",
    "test_excluded_from_fit",
)
_COVERAGE_KEYS = (
    "rows",
    "products",
    "date_min",
    "date_max",
    "calendar_span_days",
    "stockout_rows",
    "rows_with_any_observed_stockout_status",
)


class BenchmarkReportError(RuntimeError):
    """Se lanza cuando el reporte congelado falta, no se puede leer o está malformado."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkReportError(message)


def _require_type(value: Any, types: tuple[type, ...], where: str) -> None:
    _require(
        isinstance(value, types),
        f"Malformed report: '{where}' must be {_type_names(types)}, got {type(value).__name__}.",
    )


def _type_names(types: tuple[type, ...]) -> str:
    return " or ".join(t.__name__ for t in types)


def _require_str(value: Any, where: str) -> None:
    _require(
        isinstance(value, str) and bool(value),
        f"Malformed report: '{where}' must be a non-empty string.",
    )


def _require_bool(value: Any, where: str) -> None:
    _require(isinstance(value, bool), f"Malformed report: '{where}' must be a boolean.")


def _require_int(value: Any, where: str) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"Malformed report: '{where}' must be an integer.",
    )


def _require_number_or_none(value: Any, where: str) -> None:
    _require(
        value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)),
        f"Malformed report: '{where}' must be a number or null.",
    )


def _require_list(value: Any, where: str) -> None:
    _require(isinstance(value, list), f"Malformed report: '{where}' must be a list.")


def _require_dict(value: Any, where: str) -> None:
    _require(
        isinstance(value, dict),
        f"Malformed report: '{where}' must be an object.",
    )


def _require_children(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    where: str,
) -> None:
    for key in keys:
        _require(
            key in mapping,
            f"Missing required field '{where}.{key}' in the benchmark report.",
        )


@dataclass(frozen=True)
class CoverageFacts:
    """Hechos de cobertura agregados del reporte (sin detalle por fecha)."""

    rows: int
    products: int
    date_min: str
    date_max: str
    calendar_span_days: int
    stockout_rows: int
    rows_with_any_observed_stockout_status: int


@dataclass(frozen=True)
class FoldInfo:
    """Una partición temporal: límites más hechos del historial de ajuste del protocolo."""

    fold: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_rows: int
    validation_rows: int
    test_rows: int
    evaluation_rows: int
    fit_history_start: str
    fit_history_end: str
    fit_row_count: int
    evaluable: bool


@dataclass(frozen=True)
class ResultRow:
    """Una fila de resultado agregado por partición/modelo (nunca predicciones crudas)."""

    model: str
    fold: str
    mae: float | None
    rmse: float | None
    wmape: float | None
    bias: float | None
    coverage: float | None
    available: int
    requested: int
    evaluated: int
    stockout_rows: int
    fit_partition: str
    test_excluded_from_fit: bool


@dataclass(frozen=True)
class BenchmarkReport:
    """Vista inmutable y validada del reporte de benchmark acotado congelado."""

    dataset: str
    dataset_revision: str
    evaluation_id: str
    evaluation_mode: str
    evaluation_partition: str
    evaluation_protocol_version: str
    fit_partition: str
    test_excluded_from_fit: bool
    baseline_parameters_fixed: bool
    schema_name: str
    max_rows: int
    rows_loaded: int
    models: tuple[str, ...]
    folds: tuple[FoldInfo, ...]
    fold_statuses: tuple[dict[str, Any], ...] = field(repr=False)
    results: tuple[ResultRow, ...] = field(repr=False)
    coverage: CoverageFacts = field(repr=False)
    protocol: dict[str, Any] = field(repr=False)
    seasonal_naive_by_fold: dict[str, dict[str, Any]] = field(repr=False)
    limitations: tuple[str, ...] = field(repr=False)
    warnings: tuple[str, ...] = field(repr=False)
    moving_average_window: int = field(repr=False)
    ses_alpha: float = field(repr=False)

    # -- hechos agregados de conveniencia ---------------------------------------------

    @property
    def rows(self) -> int:
        return self.coverage.rows

    @property
    def products(self) -> int:
        return self.coverage.products

    @property
    def date_min(self) -> str:
        return self.coverage.date_min

    @property
    def date_max(self) -> str:
        return self.coverage.date_max

    @property
    def calendar_span_days(self) -> int:
        return self.coverage.calendar_span_days

    @property
    def coverage_complete(self) -> bool:
        """True solo cuando cada fila de resultado reporta cobertura completa (1.0)."""
        return bool(self.results) and all(
            row.coverage == 1.0 for row in self.results
        )

    @property
    def has_universal_winner(self) -> bool:
        """True solo si un modelo gana cada métrica en cada partición.

        El benchmark congelado no muestra tal modelo; el visor nunca selecciona uno.
        """
        metrics = ("mae", "rmse", "wmape")
        wins: dict[str, int] = {model: 0 for model in self.models}
        for fold in self.folds:
            for metric in metrics:
                best = self._smallest(fold.fold, metric)
                if best is not None:
                    wins[best] += 1
        total = len(self.folds) * len(metrics)
        return any(count == total for count in wins.values())

    # -- resúmenes planos derivados ---------------------------------------------------

    def results_table(self, fold: str | None = None) -> list[dict[str, Any]]:
        """Filas agregadas deterministas: particiones en orden del reporte, modelos en orden fijo.

        ``fold=None`` devuelve todas las particiones; de lo contrario solo esa.
        Los identificadores de partición desconocidos fallan cerrado en lugar de
        devolver una tabla parcial.
        """
        if fold is not None:
            _require(
                any(f.fold == fold for f in self.folds),
                f"Unknown fold '{fold}' in benchmark report.",
            )
        return [
            {
                "fold": result.fold,
                "model": result.model,
                "mae": result.mae,
                "rmse": result.rmse,
                "wmape": result.wmape,
                "bias": result.bias,
                "coverage": result.coverage,
                "available": result.available,
                "requested": result.requested,
                "evaluated": result.evaluated,
                "stockout_rows": result.stockout_rows,
            }
            for result in self.results
            if fold is None or result.fold == fold
        ]

    def fold_test_stockout_rows(self, fold: str) -> int:
        """Filas con quiebre en la prueba de la partición (todos los modelos coinciden; si no, falla cerrado)."""
        values = {r.stockout_rows for r in self.results if r.fold == fold}
        _require(
            len(values) == 1,
            f"Inconsistent stockout rows across models for fold '{fold}'.",
        )
        return next(iter(values))

    def observations(self) -> tuple[str, ...]:
        """Declaraciones descriptivas calculadas de los resultados actuales, no hardcodeadas.

        Cada declaración se deriva del valor métrico no None más pequeño por
        partición; los ``None`` se omiten y nunca se inventan.
        """
        statements: list[str] = []
        for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("wmape", "WMAPE")):
            for fold in self.folds:
                best = self._smallest(fold.fold, metric)
                if best is None:
                    continue
                value = next(
                    r[metric]
                    for r in self.results_table(fold=fold.fold)
                    if r["model"] == best
                )
                statements.append(
                    f"{best} has the smallest {label} in {fold.fold} ({value:.3f})."
                )
        if not self.has_universal_winner:
            statements.append(
                "No single model has the smallest error for every metric in both folds."
            )
        return tuple(statements)

    def _smallest(self, fold: str, metric: str) -> str | None:
        candidates = [
            r for r in self.results
            if r.fold == fold and getattr(r, metric) is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: getattr(r, metric)).model

    # -- validación y construcción ----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        """Valida un dict de reporte y construye una vista inmutable (falla cerrado)."""
        _require_dict(data, "report root")
        for key in _TOP_LEVEL_STRINGS:
            _require_children(data, (key,), "report")
            _require_str(data[key], f"report.{key}")
        for key in _TOP_LEVEL_BOOLS:
            _require_children(data, (key,), "report")
            _require_bool(data[key], f"report.{key}")
        for key in _TOP_LEVEL_INTS:
            _require_children(data, (key,), "report")
            _require_int(data[key], f"report.{key}")
        for key in _TOP_LEVEL_LISTS:
            _require_children(data, (key,), "report")
            _require_list(data[key], f"report.{key}")
        for key in _TOP_LEVEL_DICTS:
            _require_children(data, (key,), "report")
            _require_dict(data[key], f"report.{key}")

        _require(
            data["schema"] == _SCHEMA_NAME,
            f"Unsupported report schema '{data['schema']}'; expected '{_SCHEMA_NAME}'.",
        )
        _require(
            data["evaluation_id"] == _REPORT_FILE_NAME.removesuffix(".json"),
            f"Unexpected evaluation_id '{data['evaluation_id']}'.",
        )
        _require(
            len(data["models"]) > 0,
            "Malformed report: 'report.models' must not be empty.",
        )
        for model in data["models"]:
            _require_str(model, "report.models[]")
        _require(
            len(data["results"]) > 0,
            "Malformed report: 'report.results' must not be empty.",
        )

        protocol = data["protocol"]
        for key in _PROTOCOL_KEYS:
            _require_children(protocol, (key,), "protocol")
            if key in ("test_excluded_from_fit", "baseline_parameters_fixed"):
                _require_bool(protocol[key], f"protocol.{key}")
            elif key == "fit_partition":
                _require_str(protocol[key], "protocol.fit_partition")
            else:
                _require_str(protocol[key], f"protocol.{key}")

        for top_key, protocol_key in (
            ("evaluation_protocol_version", "evaluation_protocol_version"),
            ("fit_partition", "fit_partition"),
            ("test_excluded_from_fit", "test_excluded_from_fit"),
            ("baseline_parameters_fixed", "baseline_parameters_fixed"),
        ):
            _require(
                data[top_key] == protocol[protocol_key],
                f"Inconsistent '{top_key}' between report root and protocol.",
            )

        configuration = data.get("configuration")
        _require_dict(configuration, "report.configuration")
        _require_children(
            configuration,
            ("dataset_revision", "evaluation_id", "models", "moving_average_window", "ses_alpha"),
            "configuration",
        )
        _require_int(
            configuration["moving_average_window"],
            "configuration.moving_average_window",
        )
        _require_number_or_none(
            configuration["ses_alpha"],
            "configuration.ses_alpha",
        )
        _require(
            configuration["dataset_revision"] == data["dataset_revision"],
            "Inconsistent dataset_revision between report root and configuration.",
        )
        _require(
            configuration["evaluation_id"] == data["evaluation_id"],
            "Inconsistent evaluation_id between report root and configuration.",
        )
        _require(
            list(configuration["models"]) == list(data["models"]),
            "Inconsistent models between report root and configuration.",
        )

        fold_derivation = data["fold_derivation"]
        folds_raw = fold_derivation.get("folds")
        _require_list(folds_raw, "fold_derivation.folds")
        _require(len(folds_raw) > 0, "Malformed report: 'fold_derivation.folds' must not be empty.")
        for index, fold_raw in enumerate(folds_raw):
            _require_dict(fold_raw, f"fold_derivation.folds[{index}]")
            _require_children(fold_raw, _FOLD_KEYS, f"fold_derivation.folds[{index}]")
            for key in _FOLD_KEYS:
                _require_str(fold_raw[key], f"fold_derivation.folds[{index}].{key}")

        fold_ids = [fold_raw["fold"] for fold_raw in folds_raw]
        _require(len(set(fold_ids)) == len(fold_ids), "Duplicate fold identifiers in report.")

        status_by_fold: dict[str, dict[str, Any]] = {}
        for index, status in enumerate(data["fold_statuses"]):
            _require_dict(status, f"fold_statuses[{index}]")
            _require_children(status, _FOLD_STATUS_KEYS, f"fold_statuses[{index}]")
            status_fold = status["fold"]
            _require(
                status_fold in fold_ids,
                f"fold_statuses[{index}] references unknown fold '{status_fold}'.",
            )
            status_by_fold[status_fold] = status
        _require(
            len(status_by_fold) == len(fold_ids),
            "fold_statuses must cover every fold exactly once; missing "
            + ", ".join(fold for fold in fold_ids if fold not in status_by_fold)
            + ".",
        )

        models = tuple(data["models"])
        folds: list[FoldInfo] = []
        for fold_raw in folds_raw:
            fold_id = fold_raw["fold"]
            status = status_by_fold[fold_id]
            _require_bool(status["evaluable"], f"fold_statuses[{fold_id}].evaluable")
            for key in ("evaluation_rows", "fit_row_count", "test_rows", "train_rows", "validation_rows"):
                _require_int(status[key], f"fold_statuses[{fold_id}].{key}")
            _require_str(status["fit_history_start"], f"fold_statuses[{fold_id}].fit_history_start")
            _require_str(status["fit_history_end"], f"fold_statuses[{fold_id}].fit_history_end")
            _require_str(status["fit_partition"], f"fold_statuses[{fold_id}].fit_partition")
            _require_bool(status["test_excluded_from_fit"], f"fold_statuses[{fold_id}].test_excluded_from_fit")
            for key in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            ):
                _require(
                    status[key] == fold_raw[key],
                    f"Inconsistent '{key}' between fold_derivation and fold_statuses for '{fold_id}'.",
                )
            folds.append(
                FoldInfo(
                    fold=fold_id,
                    train_start=fold_raw["train_start"],
                    train_end=fold_raw["train_end"],
                    validation_start=fold_raw["validation_start"],
                    validation_end=fold_raw["validation_end"],
                    test_start=fold_raw["test_start"],
                    test_end=fold_raw["test_end"],
                    train_rows=status["train_rows"],
                    validation_rows=status["validation_rows"],
                    test_rows=status["test_rows"],
                    evaluation_rows=status["evaluation_rows"],
                    fit_history_start=status["fit_history_start"],
                    fit_history_end=status["fit_history_end"],
                    fit_row_count=status["fit_row_count"],
                    evaluable=status["evaluable"],
                )
            )

        results: list[ResultRow] = []
        for index, result in enumerate(data["results"]):
            _require_dict(result, f"results[{index}]")
            _require_children(result, _RESULT_KEYS, f"results[{index}]")
            where = f"results[{index}]"
            _require_str(result["model"], f"{where}.model")
            _require_str(result["fold"], f"{where}.fold")
            _require(
                result["fold"] in fold_ids,
                f"{where} references unknown fold '{result['fold']}'.",
            )
            _require(
                result["model"] in models,
                f"{where} references unknown model '{result['model']}'.",
            )
            for key in ("mae", "rmse", "wmape", "coverage"):
                _require_children(result, (key,), f"results[{index}]")
                _require_number_or_none(result[key], f"{where}.{key}")
            if "bias" in result:
                _require_number_or_none(result["bias"], f"{where}.bias")
            for key in ("available", "requested", "evaluated", "stockout_rows"):
                _require_int(result[key], f"{where}.{key}")
            _require_str(result["fit_partition"], f"{where}.fit_partition")
            _require_bool(result["test_excluded_from_fit"], f"{where}.test_excluded_from_fit")
            _require(
                result["fit_partition"] == data["fit_partition"],
                f"Inconsistent fit_partition in {where}.",
            )
            _require(
                result["test_excluded_from_fit"] is True,
                f"{where} must have test_excluded_from_fit=true.",
            )
            results.append(
                ResultRow(
                    model=result["model"],
                    fold=result["fold"],
                    mae=result["mae"],
                    rmse=result["rmse"],
                    wmape=result["wmape"],
                    bias=result.get("bias"),
                    coverage=result["coverage"],
                    available=result["available"],
                    requested=result["requested"],
                    evaluated=result["evaluated"],
                    stockout_rows=result["stockout_rows"],
                    fit_partition=result["fit_partition"],
                    test_excluded_from_fit=result["test_excluded_from_fit"],
                )
            )

        # Orden canónico determinista: particiones en orden del reporte, luego modelos.
        canonical: list[ResultRow] = []
        for fold_id in fold_ids:
            for model in models:
                for result in results:
                    if result.fold == fold_id and result.model == model:
                        canonical.append(result)
        _require(
            len(canonical) == len(results),
            "Results do not form a complete fold × model grid.",
        )

        coverage = data["coverage"]
        for key in _COVERAGE_KEYS:
            _require_children(coverage, (key,), "coverage")
        for key in ("rows", "products", "calendar_span_days", "stockout_rows", "rows_with_any_observed_stockout_status"):
            _require_int(coverage[key], f"coverage.{key}")
        _require_str(coverage["date_min"], "coverage.date_min")
        _require_str(coverage["date_max"], "coverage.date_max")

        seasonal: dict[str, dict[str, Any]] = {}
        for fold_id in fold_ids:
            _require(
                fold_id in data["seasonal_naive_by_fold"],
                f"Missing 'seasonal_naive_by_fold.{fold_id}' in the benchmark report.",
            )
            entry = data["seasonal_naive_by_fold"][fold_id]
            _require_dict(entry, f"seasonal_naive_by_fold.{fold_id}")
            for key in ("available", "requested"):
                _require_children(entry, (key,), f"seasonal_naive_by_fold.{fold_id}")
                _require_int(entry[key], f"seasonal_naive_by_fold.{fold_id}.{key}")
            _require_children(entry, ("coverage",), f"seasonal_naive_by_fold.{fold_id}")
            _require_number_or_none(entry["coverage"], f"seasonal_naive_by_fold.{fold_id}.coverage")
            seasonal[fold_id] = {
                "available": entry["available"],
                "requested": entry["requested"],
                "coverage": entry["coverage"],
                "wmape_status": entry.get("wmape_status"),
            }

        for key in ("limitations", "warnings"):
            for index, item in enumerate(data[key]):
                _require_str(item, f"report.{key}[{index}]")

        return cls(
            dataset=data["dataset"],
            dataset_revision=data["dataset_revision"],
            evaluation_id=data["evaluation_id"],
            evaluation_mode=data["evaluation_mode"],
            evaluation_partition=data["evaluation_partition"],
            evaluation_protocol_version=data["evaluation_protocol_version"],
            fit_partition=data["fit_partition"],
            test_excluded_from_fit=data["test_excluded_from_fit"],
            baseline_parameters_fixed=data["baseline_parameters_fixed"],
            schema_name=data["schema"],
            max_rows=data["max_rows"],
            rows_loaded=data["rows_loaded"],
            models=models,
            folds=tuple(folds),
            fold_statuses=tuple(data["fold_statuses"]),
            results=tuple(canonical),
            coverage=CoverageFacts(
                rows=coverage["rows"],
                products=coverage["products"],
                date_min=coverage["date_min"],
                date_max=coverage["date_max"],
                calendar_span_days=coverage["calendar_span_days"],
                stockout_rows=coverage["stockout_rows"],
                rows_with_any_observed_stockout_status=coverage[
                    "rows_with_any_observed_stockout_status"
                ],
            ),
            protocol={
                "evaluation_protocol_version": protocol["evaluation_protocol_version"],
                "evaluation_partition": protocol["evaluation_partition"],
                "fit_partition": protocol["fit_partition"],
                "test_excluded_from_fit": protocol["test_excluded_from_fit"],
                "baseline_parameters_fixed": protocol["baseline_parameters_fixed"],
            },
            seasonal_naive_by_fold=seasonal,
            limitations=tuple(data["limitations"]),
            warnings=tuple(data["warnings"]),
            moving_average_window=configuration["moving_average_window"],
            ses_alpha=configuration["ses_alpha"],
        )


def load_report(path: str | Path = REPORT_PATH) -> BenchmarkReport:
    """Carga y valida el reporte congelado (por defecto: la única ruta permitida)."""
    report_path = Path(path)
    _require(
        report_path.name == _REPORT_FILE_NAME,
        f"Only the frozen v2 report is allowed; got '{report_path.name}'.",
    )
    if not report_path.is_file():
        raise BenchmarkReportError(f"Benchmark report file not found: {report_path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkReportError(
            f"Benchmark report is not valid JSON: {report_path} ({exc})"
        ) from exc
    _require_dict(data, "report root")
    return BenchmarkReport.from_dict(data)
