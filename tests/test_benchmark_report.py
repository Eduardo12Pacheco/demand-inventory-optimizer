"""Pruebas del lector puro del reporte de benchmark (artefacto v2 congelado).

El visor de demo es solo agregados: estas pruebas fijan la ruta del reporte, la
validación del esquema, los resúmenes deterministas de partición/modelo, las
observaciones calculadas y el comportamiento de falla cerrada ante campos
faltantes o malformados.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from inventory_optimizer.demo.benchmark_report import (
    REPORT_PATH,
    REPORT_RELATIVE_PATH,
    BenchmarkReport,
    BenchmarkReportError,
    load_report,
)

EXPECTED_EVALUATION_ID = "freshretailnet-50k-bounded-real-1000-v2"
EXPECTED_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
EXPECTED_SCHEMA = "inventory_optimizer.bounded_real_evaluation/v1"
EXPECTED_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
EXPECTED_PROTOCOL_VERSION = "baseline-evaluation-v2"
EXPECTED_MODELS = ("naive", "seasonal_naive", "moving_average", "ses")
EXPECTED_FOLDS = ("real-fold-1", "real-fold-2")

EXPECTED_ROW_KEYS = {
    "fold",
    "model",
    "mae",
    "rmse",
    "wmape",
    "bias",
    "coverage",
    "available",
    "requested",
    "evaluated",
    "stockout_rows",
}


def raw_report() -> dict:
    """El reporte v2 congelado exactamente como está almacenado en disco."""
    return json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))


def mutated_report() -> dict:
    """Una copia profunda del reporte congelado para pruebas de fallo por mutación."""
    return copy.deepcopy(raw_report())


# ---------------------------------------------------------------------------
# Ruta fija, carga y hechos de identidad
# ---------------------------------------------------------------------------


def test_report_path_constant_points_to_frozen_v2() -> None:
    assert REPORT_RELATIVE_PATH == (
        "data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json"
    )
    assert REPORT_PATH.name == "freshretailnet-50k-bounded-real-1000-v2.json"
    assert REPORT_PATH.is_file()
    # El visor nunca debe leer el reporte v1 de diagnóstico.
    assert REPORT_PATH.name != "freshretailnet-50k-bounded-real-1000.json"


def test_fixed_v2_report_loads_and_identifies() -> None:
    report = load_report()
    assert report.evaluation_id == EXPECTED_EVALUATION_ID
    assert report.dataset == EXPECTED_DATASET
    assert report.dataset_revision == EXPECTED_REVISION
    assert report.schema_name == EXPECTED_SCHEMA
    assert report.evaluation_mode == "bounded_real_evaluation"
    assert report.evaluation_partition == "test"
    assert report.evaluation_protocol_version == EXPECTED_PROTOCOL_VERSION
    assert report.max_rows == 1000
    assert report.rows_loaded == 1000


def test_rows_products_and_date_span() -> None:
    report = load_report()
    assert report.rows == 1000
    assert report.products == 12
    assert report.date_min == "2024-03-28"
    assert report.date_max == "2024-06-25"
    assert report.calendar_span_days == 90


# ---------------------------------------------------------------------------
# Particiones, modelos y sobre de protocolo
# ---------------------------------------------------------------------------


def test_two_consistent_folds_and_four_models() -> None:
    report = load_report()
    assert tuple(report.models) == EXPECTED_MODELS
    assert tuple(f.fold for f in report.folds) == EXPECTED_FOLDS

    fold1, fold2 = report.folds
    assert (fold1.train_start, fold1.train_end) == ("2024-03-28", "2024-05-28")
    assert (fold1.validation_start, fold1.validation_end) == ("2024-05-29", "2024-06-04")
    assert (fold1.test_start, fold1.test_end) == ("2024-06-05", "2024-06-11")
    assert (fold2.test_start, fold2.test_end) == ("2024-06-19", "2024-06-25")

    # Los límites de la partición de fold_derivation coinciden con los de fold_statuses.
    for fold in report.folds:
        status = next(s for s in report.fold_statuses if s["fold"] == fold.fold)
        assert status["train_start"] == fold.train_start
        assert status["train_end"] == fold.train_end
        assert status["validation_start"] == fold.validation_start
        assert status["validation_end"] == fold.validation_end
        assert status["test_start"] == fold.test_start
        assert status["test_end"] == fold.test_end


def test_protocol_envelope_train_plus_validation_and_test_excluded() -> None:
    report = load_report()
    assert report.fit_partition == "train+validation"
    assert report.test_excluded_from_fit is True
    assert report.baseline_parameters_fixed is True
    assert report.moving_average_window == 7
    assert report.ses_alpha == 0.3
    assert report.protocol == {
        "evaluation_protocol_version": "baseline-evaluation-v2",
        "evaluation_partition": "test",
        "fit_partition": "train+validation",
        "test_excluded_from_fit": True,
        "baseline_parameters_fixed": True,
    }
    for row in report.results:
        assert row.fit_partition == "train+validation"
        assert row.test_excluded_from_fit is True
    for status in report.fold_statuses:
        assert status["fit_partition"] == "train+validation"
        assert status["test_excluded_from_fit"] is True
        assert status["evaluation_protocol_version"] == "baseline-evaluation-v2"


def test_fold_fit_history_dates_and_fit_counts() -> None:
    report = load_report()
    fold1, fold2 = report.folds
    assert (fold1.fit_history_start, fold1.fit_history_end) == ("2024-03-28", "2024-06-04")
    assert fold1.fit_row_count == 769
    assert (fold2.fit_history_start, fold2.fit_history_end) == ("2024-03-28", "2024-06-18")
    assert fold2.fit_row_count == 923
    # Conteos de filas por partición: train / validation / test.
    assert (fold1.train_rows, fold1.validation_rows, fold1.test_rows) == (692, 77, 77)
    assert (fold2.train_rows, fold2.validation_rows, fold2.test_rows) == (846, 77, 77)
    assert fold1.evaluation_rows == 77
    assert fold2.evaluation_rows == 77


# ---------------------------------------------------------------------------
# Cobertura y completitud de resultados
# ---------------------------------------------------------------------------


def test_coverage_complete_and_77_available_per_fold() -> None:
    report = load_report()
    assert len(report.results) == 8
    for row in report.results:
        assert row.available == 77
        assert row.requested == 77
        assert row.evaluated == 77
        assert row.coverage == 1.0
    assert report.coverage_complete is True
    assert report.coverage.rows == 1000
    assert report.coverage.products == 12


def test_seasonal_naive_v2_diagnostic_facts() -> None:
    report = load_report()
    for fold in EXPECTED_FOLDS:
        facts = report.seasonal_naive_by_fold[fold]
        assert facts["available"] == 77
        assert facts["requested"] == 77
        assert facts["coverage"] == 1.0


# ---------------------------------------------------------------------------
# Resúmenes deterministas y filtrado por partición
# ---------------------------------------------------------------------------


def test_deterministic_summaries_and_fixed_order() -> None:
    report = load_report()
    table = report.results_table()
    assert len(table) == 8
    # Orden fijo: particiones en orden del reporte, luego modelos en el orden fijo del reporte.
    expected = [
        (fold, model)
        for fold in EXPECTED_FOLDS
        for model in EXPECTED_MODELS
    ]
    assert [(row["fold"], row["model"]) for row in table] == expected
    assert report.results_table() == report.results_table()  # deterministic


def test_summary_values_match_the_json_exactly() -> None:
    report = load_report()
    raw = raw_report()
    raw_by_key = {(r["fold"], r["model"]): r for r in raw["results"]}
    for row in report.results_table():
        raw_row = raw_by_key[(row["fold"], row["model"])]
        for key in ("mae", "rmse", "wmape", "bias", "coverage"):
            assert row[key] == raw_row[key], key
        for key in ("available", "requested", "evaluated", "stockout_rows"):
            assert row[key] == raw_row[key], key


def test_fold_filter_is_deterministic() -> None:
    report = load_report()
    fold2 = report.results_table(fold="real-fold-2")
    assert len(fold2) == 4
    assert {row["fold"] for row in fold2} == {"real-fold-2"}
    assert [row["model"] for row in fold2] == list(EXPECTED_MODELS)
    assert report.results_table(fold="real-fold-2") == fold2
    with pytest.raises(BenchmarkReportError):
        report.results_table(fold="unknown-fold")


# ---------------------------------------------------------------------------
# Las observaciones se calculan, y no se selecciona ningún ganador
# ---------------------------------------------------------------------------


def _argmin_model(raw: dict, fold: str, metric: str) -> str:
    """Cálculo de referencia: valor métrico no None más pequeño por partición."""
    candidates = [
        r for r in raw["results"]
        if r["fold"] == fold and r.get(metric) is not None
    ]
    assert candidates, (fold, metric)
    return min(candidates, key=lambda r: r[metric])["model"]


def test_observations_are_calculated_from_current_results() -> None:
    report = load_report()
    raw = raw_report()
    observations = report.observations()
    for fold in EXPECTED_FOLDS:
        for metric in ("mae", "rmse", "wmape"):
            expected_model = _argmin_model(raw, fold, metric)
            statement = next(
                s for s in observations if f" in {fold}" in s and metric.upper() in s
            )
            assert expected_model in statement, (fold, metric, statement)


def test_frozen_artifact_observation_facts() -> None:
    """Fija los hechos observables del artefacto congelado (documentados, no ranqueados).

    Según el reporte: ses es el más pequeño en MAE/WMAPE/RMSE en la partición 1;
    moving_average es el más pequeño en MAE/WMAPE/RMSE en la partición 2.
    """
    report = load_report()
    observations = report.observations()
    fold1 = " ".join(s for s in observations if "real-fold-1" in s)
    fold2 = " ".join(s for s in observations if "real-fold-2" in s)
    assert "ses has the smallest MAE in real-fold-1" in fold1
    assert "ses has the smallest WMAPE in real-fold-1" in fold1
    assert "ses has the smallest RMSE in real-fold-1" in fold1
    assert "moving_average has the smallest MAE in real-fold-2" in fold2
    assert "moving_average has the smallest WMAPE in real-fold-2" in fold2
    assert "moving_average has the smallest RMSE in real-fold-2" in fold2


def test_no_automatic_winner_selection() -> None:
    report = load_report()
    assert report.has_universal_winner is False
    assert "No single model has the smallest error for every metric" in report.observations()[-1]


# ---------------------------------------------------------------------------
# Solo agregados: sin filas crudas ni predicciones individuales
# ---------------------------------------------------------------------------


def test_summaries_expose_only_aggregates() -> None:
    report = load_report()
    for row in report.results_table():
        assert set(row) == EXPECTED_ROW_KEYS
        for value in row.values():
            assert not isinstance(value, (list, dict, tuple)), row
    for result in report.results:
        assert not hasattr(result, "predictions")
        assert not hasattr(result, "rows")


# ---------------------------------------------------------------------------
# Falla cerrada: campos requeridos faltantes o malformados
# ---------------------------------------------------------------------------


def test_missing_top_level_fields_fail_clearly() -> None:
    for key in ("protocol", "models", "results", "coverage", "fold_derivation"):
        data = mutated_report()
        del data[key]
        with pytest.raises(BenchmarkReportError) as exc:
            BenchmarkReport.from_dict(data)
        assert key in str(exc.value)


def test_missing_protocol_fields_fail_clearly() -> None:
    for key in (
        "evaluation_protocol_version",
        "fit_partition",
        "test_excluded_from_fit",
        "baseline_parameters_fixed",
    ):
        data = mutated_report()
        del data["protocol"][key]
        with pytest.raises(BenchmarkReportError) as exc:
            BenchmarkReport.from_dict(data)
        assert f"protocol.{key}" in str(exc.value)


def test_missing_fold_fields_fail_clearly() -> None:
    data = mutated_report()
    del data["fold_derivation"]["folds"][0]["validation_start"]
    with pytest.raises(BenchmarkReportError) as exc:
        BenchmarkReport.from_dict(data)
    assert "fold_derivation.folds[0].validation_start" in str(exc.value)


def test_missing_result_fields_fail_clearly() -> None:
    for key in ("mae", "rmse", "wmape", "coverage", "available", "evaluated", "stockout_rows"):
        data = mutated_report()
        del data["results"][0][key]
        with pytest.raises(BenchmarkReportError) as exc:
            BenchmarkReport.from_dict(data)
        assert f"results[0].{key}" in str(exc.value)


def test_malformed_shapes_fail_clearly() -> None:
    cases = [
        ("models", []),
        ("models", "naive"),
        ("results", []),
        ("results", "not-a-list"),
        ("coverage", []),
        ("coverage", {"rows": "one-thousand"}),
        ("max_rows", None),
    ]
    for key, value in cases:
        data = mutated_report()
        data[key] = value
        with pytest.raises(BenchmarkReportError) as exc:
            BenchmarkReport.from_dict(data)
        assert key in str(exc.value)


def test_non_dict_root_fails_clearly() -> None:
    with pytest.raises(BenchmarkReportError):
        BenchmarkReport.from_dict(["not", "an", "object"])


def test_fold_statuses_missing_a_fold_fails_clearly() -> None:
    data = mutated_report()
    data["fold_statuses"] = [s for s in data["fold_statuses"] if s["fold"] != "real-fold-2"]
    with pytest.raises(BenchmarkReportError) as exc:
        BenchmarkReport.from_dict(data)
    assert "real-fold-2" in str(exc.value)


def test_result_with_unknown_fold_or_model_fails_clearly() -> None:
    data = mutated_report()
    data["results"][0]["fold"] = "made-up-fold"
    with pytest.raises(BenchmarkReportError):
        BenchmarkReport.from_dict(data)

    data = mutated_report()
    data["results"][0]["model"] = "deep-learning"
    with pytest.raises(BenchmarkReportError):
        BenchmarkReport.from_dict(data)


# ---------------------------------------------------------------------------
# Métricas None y bias opcional faltante nunca inventan valores
# ---------------------------------------------------------------------------


def test_none_metrics_and_missing_bias_are_preserved() -> None:
    data = mutated_report()
    data["results"][0]["mae"] = None
    data["results"][0]["wmape"] = None
    data["results"][0]["coverage"] = None
    del data["results"][0]["bias"]  # opcional: puede estar ausente

    report = BenchmarkReport.from_dict(data)
    row = next(
        r for r in report.results_table()
        if r["fold"] == "real-fold-1" and r["model"] == "naive"
    )
    assert row["mae"] is None
    assert row["wmape"] is None
    assert row["coverage"] is None
    assert row["bias"] is None  # la ausencia se conserva como None, nunca se inventa
    assert report.coverage_complete is False
    # Las observaciones omiten los valores None y se recalculan a partir de los
    # resultados actuales: los mínimos de la partición 1 ahora son ses (mae/wmape), nunca naive.
    observations = report.observations()
    for metric, label in (("mae", "MAE"), ("wmape", "WMAPE")):
        candidates = [
            r for r in data["results"]
            if r["fold"] == "real-fold-1" and r.get(metric) is not None
        ]
        expected_model = min(candidates, key=lambda r: r[metric])["model"]
        statement = next(
            s for s in observations
            if f" in real-fold-1" in s and label in s
        )
        assert expected_model in statement, (metric, statement)
    assert any("real-fold-2" in s for s in observations)


def test_stockout_facts_from_json() -> None:
    report = load_report()
    assert report.coverage.stockout_rows == 593
    assert report.coverage.rows_with_any_observed_stockout_status == 708
    assert report.fold_test_stockout_rows("real-fold-1") == 34
    assert report.fold_test_stockout_rows("real-fold-2") == 33


def test_load_report_equals_from_dict_of_the_same_file() -> None:
    assert load_report().results_table() == BenchmarkReport.from_dict(raw_report()).results_table()


def test_load_report_missing_file_fails_clearly(tmp_path) -> None:
    missing = tmp_path / REPORT_PATH.name  # nombre permitido, ubicación inexistente
    with pytest.raises(BenchmarkReportError) as exc:
        load_report(missing)
    assert "not found" in str(exc.value)


def test_load_report_rejects_any_file_other_than_the_frozen_v2(tmp_path) -> None:
    other = tmp_path / "freshretailnet-50k-bounded-real-1000.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkReportError) as exc:
        load_report(other)
    assert "frozen v2 report" in str(exc.value)
