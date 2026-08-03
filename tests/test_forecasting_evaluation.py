"""Pruebas del ejecutor de evaluación de líneas base offline.

Cubre: filas deterministas de partición/modelo, preservación explícita de los
límites train/validation/test, ausencia de fuga (mutar valores de particiones
futuras no puede cambiar las predicciones), cobertura y conteos de
solicitadas/disponibles/evaluadas, predicciones ``None`` excluidas de las métricas
pero representadas, conteos de metadatos de quiebre, propagación de la revisión de
fuente e identificadores, estado explícito de WMAPE no calculable y de
denominador cero, fallos de validación de configuración y serialización JSON
estable y sin datos crudos. Todas las filas son sintéticas y offline.

También cubre el protocolo de evaluación versionado (``baseline-evaluation-v2``):
la evaluación de validación ajusta SOLO ``split.train``; la evaluación de prueba
ajusta la combinación ordenada y con duplicados verificados
``split.train + split.validation``; las filas de prueba nunca entran al historial
de ajuste (claves de ajuste y prueba disjuntas, el ajuste termina antes de que
comience la prueba); seasonal_naive puede usar una observación de validación como
su rezago de 7 días para la prueba; se registran los campos de ajuste por
partición/resultado (``fit_partition``, ``fit_row_count``,
``fit_history_start``/``fit_history_end``, ``test_excluded_from_fit``,
``insufficient_fit_history``); existe un sobre de protocolo a nivel de reporte con
parámetros de línea base fijos; y nunca ocurre selección automática de ganador.
"""

from __future__ import annotations

import json
import math
from datetime import date

import pytest

from inventory_optimizer.forecasting.evaluation import (
    EVALUATION_PARTITION_TEST,
    EVALUATION_PARTITION_VALIDATION,
    EVALUATION_PROTOCOL_VERSION,
    BaselineEvaluationConfig,
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    EvaluationSplitConfig,
    FoldEvaluationConfig,
    FoldEvaluationStatus,
    run_baseline_evaluation,
)
from inventory_optimizer.forecasting.metrics import (
    WMAPE_CALCULABLE,
    WMAPE_NO_EVALUABLE,
    WMAPE_ZERO_ACTUALS,
)
from inventory_optimizer.forecasting.splits import (
    DuplicateKeyError,
    MissingDatesForProduct,
)

from helpers import (
    EVALUATION_FIXTURE_EVALUATION_ID,
    EVALUATION_FIXTURE_REVISION,
    EVALUATION_FIXTURE_SOURCE_ID,
    make_daily_row,
    make_evaluation_rows,
)

MODEL_ORDER = ("naive", "seasonal_naive", "moving_average", "ses")


def _default_folds() -> tuple[FoldEvaluationConfig, ...]:
    return (
        FoldEvaluationConfig(
            fold="fold-1",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 10),
            validation_start=date(2024, 1, 11),
            validation_end=date(2024, 1, 13),
            test_start=date(2024, 1, 14),
            test_end=date(2024, 1, 16),
        ),
        FoldEvaluationConfig(
            fold="fold-2",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 13),
            validation_start=date(2024, 1, 14),
            validation_end=date(2024, 1, 16),
            test_start=date(2024, 1, 17),
            test_end=date(2024, 1, 19),
        ),
    )


def _config(
    *,
    evaluation_partition: str = EVALUATION_PARTITION_TEST,
    product_group: str = "all_products",
    folds: tuple[FoldEvaluationConfig, ...] | None = None,
    baseline: BaselineEvaluationConfig | None = None,
) -> EvaluationConfig:
    return EvaluationConfig(
        source_id=EVALUATION_FIXTURE_SOURCE_ID,
        dataset_revision=EVALUATION_FIXTURE_REVISION,
        evaluation_id=EVALUATION_FIXTURE_EVALUATION_ID,
        split_config=EvaluationSplitConfig(
            folds=folds if folds is not None else _default_folds(),
            evaluation_partition=evaluation_partition,
            product_group=product_group,
        ),
        baseline_config=baseline if baseline is not None else BaselineEvaluationConfig(),
    )


def _run(*args, **kwargs) -> EvaluationReport:
    return run_baseline_evaluation(make_evaluation_rows(), _config(*args, **kwargs))


def _ses_level(values: list[float], alpha: float) -> float:
    """Recursión SES de referencia: level_t = alpha*x_t + (1-alpha)*level_{t-1}."""
    level = values[0]
    for value in values[1:]:
        level = alpha * value + (1.0 - alpha) * level
    return level


# --- comportamiento del ejecutor -----------------------------------------------------


def test_all_four_baselines_execute_for_every_fold():
    report = _run()
    assert isinstance(report, EvaluationReport)
    assert len(report.results) == 8
    assert [result.model for result in report.results] == list(MODEL_ORDER) * 2
    assert [result.fold for result in report.results] == ["fold-1"] * 4 + ["fold-2"] * 4
    assert {result.model for result in report.results} == set(MODEL_ORDER)
    assert {result.fold for result in report.results} == {"fold-1", "fold-2"}


def test_multiple_folds_produce_deterministic_fold_model_rows():
    report = _run()
    by_fold_model = {(result.fold, result.model): result for result in report.results}
    assert len(by_fold_model) == 8

    fold1_naive = by_fold_model[("fold-1", "naive")]
    # Ajuste = train 1..10 + validation 11..13 -> predicción naive 13; test 14,15,16.
    assert fold1_naive.mae == pytest.approx(2.0)
    assert fold1_naive.rmse == pytest.approx(math.sqrt(14.0 / 3.0))
    assert fold1_naive.wmape == pytest.approx(6.0 / 45.0)
    assert fold1_naive.bias == pytest.approx(-2.0)
    assert fold1_naive.wmape_status == WMAPE_CALCULABLE

    fold2_naive = by_fold_model[("fold-2", "naive")]
    # Ajuste = train 1..13 + validation 14..16 -> A predice 16; C predice 5.0.
    assert fold2_naive.mae == pytest.approx((1.0 + 2.0 + 3.0) / 6.0)
    assert fold2_naive.wmape == pytest.approx(6.0 / 69.0)

    # Naive estacional: rezagos exactos de 7 días 7,8,9 (train) -> errores 7,7,7 (solo A;
    # los rezagos 10..12 de C son anteriores a sus filas de ajuste 14..16 de la partición 2).
    assert by_fold_model[("fold-1", "seasonal_naive")].mae == pytest.approx(7.0)
    assert by_fold_model[("fold-2", "seasonal_naive")].mae == pytest.approx(7.0)

    # Promedio móvil (ventana 7) partición 1: últimos 7 valores de ajuste 7..13 -> media 10.0.
    assert by_fold_model[("fold-1", "moving_average")].mae == pytest.approx(5.0)
    # Partición 2: últimos 7 valores de ajuste de A 10..16 -> 13.0 (errores 4,5,6); C -> 5.0.
    assert by_fold_model[("fold-2", "moving_average")].mae == pytest.approx(15.0 / 6.0)

    # SES (alpha 0.3): nivel final sobre los valores de ajuste 1..13 de la partición 1.
    level = _ses_level([float(index) for index in range(1, 14)], alpha=0.3)
    fold1_ses = by_fold_model[("fold-1", "ses")]
    assert fold1_ses.mae == pytest.approx((14 - level + 15 - level + 16 - level) / 3.0)
    assert fold1_ses.wmape == pytest.approx(
        (14 - level + 15 - level + 16 - level) / (14.0 + 15.0 + 16.0)
    )
    # SES partición 2: nivel de A sobre 1..16; nivel de C 5.0 (errores 0) -> errores de A / 6.
    level2 = _ses_level([float(index) for index in range(1, 17)], alpha=0.3)
    fold2_ses = by_fold_model[("fold-2", "ses")]
    assert fold2_ses.mae == pytest.approx(
        (17 - level2 + 18 - level2 + 19 - level2) / 6.0
    )


def test_no_leakage_mutating_future_partition_values_cannot_change_predictions():
    config = _config()
    rows = list(make_evaluation_rows())
    # Mutar solo las filas del PROPIO rango de prueba de la partición 2 (17..19): esas filas
    # están fuera de partición para la partición 1 y son filas de prueba para la partición 2,
    # así que ningún historial de ajuste se toca y cada predicción debe permanecer idéntica.
    fold2 = _default_folds()[1]
    mutated = []
    for row in rows:
        if fold2.test_start <= row.dt <= fold2.test_end:
            mutated.append(
                make_daily_row(
                    product_id=row.product_id,
                    dt=row.dt.isoformat(),
                    sale_amount=999.0,
                    stockout_hours_6_22=row.stockout_hours_6_22,
                    revision=row.revision,
                )
            )
        else:
            mutated.append(row)

    original = run_baseline_evaluation(rows, config)
    report = run_baseline_evaluation(mutated, config)

    for before, after in zip(original.results, report.results):
        assert (before.requested, before.available, before.evaluated, before.coverage) == (
            after.requested,
            after.available,
            after.evaluated,
            after.coverage,
        )
    # Las predicciones deben venir SOLO de las filas de ajuste (train + validation), nunca
    # de los reales de prueba mutados: las métricas de la partición 1 son byte-idénticas a la
    # corrida original (sus filas de prueba 14..16 NO se mutaron; las predicciones siguen en
    # 13/7,8,9/10/nivel mientras solo los reales de la partición 2 cambiaron a 999).
    by_fold_model = {(result.fold, result.model): result for result in report.results}
    original_by_fold_model = {
        (result.fold, result.model): result for result in original.results
    }
    assert by_fold_model[("fold-1", "naive")].mae == original_by_fold_model[
        ("fold-1", "naive")
    ].mae == pytest.approx(2.0)
    # Partición 2: A predice 16, C predice 5.0 (últimos valores de ajuste de la partición 2);
    # si el real de prueba mutado (999) se filtrara al ajuste, el MAE sería ~0 en lugar de ~988.5.
    assert by_fold_model[("fold-2", "naive")].mae == pytest.approx(
        (999.0 - 16.0 + 999.0 - 16.0 + 999.0 - 16.0 + 994.0 * 3) / 6.0
    )
    # Naive estacional / promedio móvil / SES partición 1: sin cambios (rezagos 7,8,9;
    # media de ajuste 10.0; nivel de ajuste sobre 1..13) contra los reales no mutados 14,15,16.
    assert by_fold_model[("fold-1", "seasonal_naive")].mae == original_by_fold_model[
        ("fold-1", "seasonal_naive")
    ].mae == pytest.approx(7.0)
    assert by_fold_model[("fold-1", "moving_average")].mae == original_by_fold_model[
        ("fold-1", "moving_average")
    ].mae == pytest.approx(5.0)
    level = _ses_level([float(index) for index in range(1, 14)], alpha=0.3)
    assert by_fold_model[("fold-1", "ses")].mae == original_by_fold_model[
        ("fold-1", "ses")
    ].mae == pytest.approx((14 - level + 15 - level + 16 - level) / 3.0)


def test_no_leakage_unevaluated_partition_changes_nothing():
    # La evaluación de validación ajusta SOLO train: las filas de prueba nunca se usan
    # para ajustar ni para evaluar, así que mutarlas no cambia nada.
    fold = FoldEvaluationConfig(
        fold="single",
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 10),
        validation_start=date(2024, 1, 11),
        validation_end=date(2024, 1, 13),
        test_start=date(2024, 1, 14),
        test_end=date(2024, 1, 16),
    )
    config = _config(folds=(fold,), evaluation_partition=EVALUATION_PARTITION_VALIDATION)
    rows = list(make_evaluation_rows())
    mutated = [
        make_daily_row(
            product_id=row.product_id,
            dt=row.dt.isoformat(),
            sale_amount=777.0,
            stockout_hours_6_22=row.stockout_hours_6_22,
            revision=row.revision,
        )
        if fold.test_start <= row.dt <= fold.test_end
        else row
        for row in rows
    ]

    original = run_baseline_evaluation(rows, config)
    report = run_baseline_evaluation(mutated, config)
    assert report == original
    assert report.to_json() == original.to_json()


def test_train_validation_test_boundaries_preserved():
    report = _run()
    fold1, fold2 = report.fold_statuses
    assert isinstance(fold1, FoldEvaluationStatus)
    assert (fold1.fold, fold2.fold) == ("fold-1", "fold-2")
    assert fold1.train_start == date(2024, 1, 1)
    assert fold1.train_end == date(2024, 1, 10)
    assert fold1.validation_start == date(2024, 1, 11)
    assert fold1.validation_end == date(2024, 1, 13)
    assert fold1.test_start == date(2024, 1, 14)
    assert fold1.test_end == date(2024, 1, 16)
    assert fold2.train_start == date(2024, 1, 1)
    assert fold2.train_end == date(2024, 1, 13)
    assert fold2.validation_start == date(2024, 1, 14)
    assert fold2.validation_end == date(2024, 1, 16)
    assert fold2.test_start == date(2024, 1, 17)
    assert fold2.test_end == date(2024, 1, 19)
    # Conteos de filas por conjunto (A: 01-01..01-20; C: 01-14..01-19).
    assert (fold1.train_rows, fold1.validation_rows, fold1.test_rows) == (10, 3, 6)
    assert (fold2.train_rows, fold2.validation_rows, fold2.test_rows) == (13, 6, 6)
    # Sin huecos en el fixture: los reportes de fechas faltantes son explícitos y vacíos.
    assert fold1.missing_train_dates == ()
    assert fold1.missing_validation_dates == ()
    assert fold1.missing_test_dates == ()
    # Las filas fuera de rango se cuentan, nunca se descartan en silencio.
    assert fold1.excluded_out_of_range == 7
    assert fold2.excluded_out_of_range == 1
    # Cada resultado métrico refleja los límites de la partición.
    for result in report.results:
        assert isinstance(result, EvaluationResult)
        if result.fold == "fold-1":
            assert (result.train_start, result.train_end) == (date(2024, 1, 1), date(2024, 1, 10))
            assert (result.test_start, result.test_end) == (date(2024, 1, 14), date(2024, 1, 16))
        else:
            assert (result.train_start, result.train_end) == (date(2024, 1, 1), date(2024, 1, 13))
            assert (result.test_start, result.test_end) == (date(2024, 1, 17), date(2024, 1, 19))


def test_coverage_and_requested_available_evaluated_counts():
    report = _run()
    for result in report.results:
        # Partición de prueba: A(3) + C(3) solicitadas en cada partición/modelo.
        assert result.requested == 6
        if result.fold == "fold-1":
            # C empieza en las fechas de prueba de la partición 1 -> sin historial de ajuste -> no disponible.
            assert result.available == 3
            assert result.unavailable == 3
            assert result.evaluated == 3
            assert result.coverage == pytest.approx(0.5)
        elif result.model == "seasonal_naive":
            # Las filas de ajuste de C en la partición 2 son 14..16; sus rezagos de 7 días 10..12 faltan.
            assert result.available == 3
            assert result.unavailable == 3
            assert result.evaluated == 3
            assert result.coverage == pytest.approx(0.5)
        else:
            # Naive / moving_average / ses: C ahora tiene historial de ajuste 14..16 en la partición 2.
            assert result.available == 6
            assert result.unavailable == 0
            assert result.evaluated == 6
            assert result.coverage == pytest.approx(1.0)


def test_none_predictions_excluded_from_metrics_but_represented():
    report = _run()
    fold1_naive = report.results[0]
    assert fold1_naive.model == "naive"
    assert fold1_naive.fold == "fold-1"
    # Las predicciones de C son None (sin historial de ajuste en la partición 1): siguen representadas en
    # solicitadas/no disponibles pero nunca entran al cálculo métrico (evaluadas == 3 == solo A).
    assert fold1_naive.unavailable == 3
    assert fold1_naive.evaluated == 3
    assert fold1_naive.mae == pytest.approx(2.0)
    # Las métricas son solo sobre las tres filas evaluadas de A.
    assert fold1_naive.wmape == pytest.approx(6.0 / 45.0)
    # El reporte en sí no lleva filas de predicción: el ejecutor las mantiene fuera;
    # los objetivos no disponibles de C siguen representados solo mediante conteos y el
    # reporte de historia insuficiente de la partición (verificado en test_insufficient_history_*).


def test_missing_dates_reported_in_fold_statuses():
    rows = [row for row in make_evaluation_rows() if row.dt != date(2024, 1, 8)]
    report = run_baseline_evaluation(rows, _config())
    fold1, fold2 = report.fold_statuses
    assert date(2024, 1, 8) in fold1.missing_train_dates
    assert date(2024, 1, 8) in fold2.missing_train_dates
    assert fold1.missing_validation_dates == ()
    assert fold1.missing_test_dates == ()
    # El naive estacional para A@2024-01-15 necesita la fecha de rezago 01-08: ahora no
    # disponible, así que solo 2 de las 3 predicciones de A se evalúan en la partición 1.
    fold1_sn = next(
        result for result in report.results if (result.fold, result.model) == ("fold-1", "seasonal_naive")
    )
    assert fold1_sn.available == 2
    assert fold1_sn.evaluated == 2
    assert fold1_sn.mae == pytest.approx((abs(14 - 7) + abs(16 - 9)) / 2.0)


def test_report_contains_per_product_missing_dates():
    rows = [
        row
        for row in make_evaluation_rows()
        if not (row.product_id == "C" and row.dt == date(2024, 1, 16))
    ]
    report = run_baseline_evaluation(rows, _config())
    for status in report.fold_statuses:
        by_product = {entry.product_id: entry for entry in status.missing_per_product}
        assert isinstance(by_product["C"], MissingDatesForProduct)
        assert date(2024, 1, 16) in by_product["C"].dates
        # La fecha faltante pertenece SOLO al producto que realmente carece de ella.
        assert date(2024, 1, 16) not in by_product["A"].dates
    payload = json.loads(report.to_json())
    for status_payload in payload["fold_statuses"]:
        per_product = {
            entry["product_id"]: entry for entry in status_payload["missing_per_product"]
        }
        assert per_product["C"]["product_id"] == "C"
        assert "2024-01-16" in per_product["C"]["dates"]
        assert "2024-01-16" not in per_product["A"]["dates"]


def test_stockout_metadata_count_preserved():
    report = _run()
    # Partición de prueba: la partición 1 evalúa solo A@01-15 (C no disponible). La partición 2
    # evalúa A@01-18 Y C@01-18 (C ahora tiene filas de validación de ajuste 14..16), así que
    # naive/promedio móvil/ses cuentan 2 filas con quiebre; seasonal_naive sigue evaluando
    # solo a A (los rezagos de 7 días 10..12 de C no están en su historial de ajuste).
    for result in report.results:
        if result.fold == "fold-1":
            assert result.stockout_rows == 1
        elif result.model == "seasonal_naive":
            assert result.stockout_rows == 1
        else:
            assert result.stockout_rows == 2
    validation_report = _run(evaluation_partition=EVALUATION_PARTITION_VALIDATION)
    for result in validation_report.results:
        # A@01-12 (partición 1) y A@01-15 (partición 2) son filas evaluadas con quiebre.
        assert result.stockout_rows == 1


def test_source_revision_and_identifiers_propagate():
    report = _run()
    for result in report.results:
        assert result.evaluation_id == EVALUATION_FIXTURE_EVALUATION_ID
        assert result.source_id == EVALUATION_FIXTURE_SOURCE_ID
        assert result.dataset_revision == EVALUATION_FIXTURE_REVISION
        assert result.target == "observed_sales"
        assert result.product_group == "all_products"
        assert result.evaluation_partition == EVALUATION_PARTITION_TEST
        assert result.config.source_id == EVALUATION_FIXTURE_SOURCE_ID
        assert result.config.dataset_revision == EVALUATION_FIXTURE_REVISION
        assert result.config.baseline_config.moving_average_window == 7
        assert result.config.baseline_config.ses_alpha == pytest.approx(0.3)


def test_deterministic_order_and_repeat_run_equality():
    first = _run()
    second = _run()
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()
    expected_order = [
        (fold, model)
        for fold in ("fold-1", "fold-2")
        for model in MODEL_ORDER
    ]
    assert [(result.fold, result.model) for result in first.results] == expected_order
    assert [status.fold for status in first.fold_statuses] == ["fold-1", "fold-2"]


def test_no_evaluable_fold_is_explicit():
    fold_without_test = FoldEvaluationConfig(
        fold="fold-3",
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 10),
        validation_start=date(2024, 1, 11),
        validation_end=date(2024, 1, 13),
    )
    report = run_baseline_evaluation(
        make_evaluation_rows(), _config(folds=(fold_without_test,))
    )
    (status,) = report.fold_statuses
    assert status.evaluable is False
    assert status.evaluation_rows == 0
    assert status.test_rows == 0
    assert status.validation_rows == 3
    assert len(report.results) == 4
    for result in report.results:
        assert result.requested == 0
        assert result.available == 0
        assert result.evaluated == 0
        assert result.coverage == pytest.approx(0.0)
        assert result.mae is None
        assert result.rmse is None
        assert result.wmape is None
        assert result.bias is None
        assert result.wmape_status == WMAPE_NO_EVALUABLE


def test_empty_rows_produce_explicit_no_evaluable_results():
    report = run_baseline_evaluation([], _config())
    assert len(report.results) == 8
    assert all(result.evaluated == 0 for result in report.results)
    assert all(result.wmape_status == WMAPE_NO_EVALUABLE for result in report.results)
    assert all(result.mae is None for result in report.results)
    assert all(not status.evaluable for status in report.fold_statuses)


def test_zero_denominator_wmape_status_preserved():
    zero_rows = [
        make_daily_row(
            product_id="A",
            dt=date(2024, 1, index).isoformat(),
            sale_amount=0.0,
            revision=EVALUATION_FIXTURE_REVISION,
        )
        for index in range(1, 12)
    ]
    fold = FoldEvaluationConfig(
        fold="zero-fold",
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 8),
        test_start=date(2024, 1, 9),
        test_end=date(2024, 1, 11),
    )
    report = run_baseline_evaluation(zero_rows, _config(folds=(fold,)))
    for result in report.results:
        # Las predicciones cero están disponibles (0.0 no es None) y se evalúan.
        assert result.available == 3
        assert result.evaluated == 3
        assert result.coverage == pytest.approx(1.0)
        assert result.mae == pytest.approx(0.0)
        assert result.rmse == pytest.approx(0.0)
        assert result.bias == pytest.approx(0.0)
        assert result.wmape is None
        assert result.wmape_status == WMAPE_ZERO_ACTUALS


def test_validation_partition_evaluation():
    report = _run(evaluation_partition=EVALUATION_PARTITION_VALIDATION)
    assert all(result.evaluation_partition == EVALUATION_PARTITION_VALIDATION for result in report.results)
    fold1_naive, fold2_naive = report.results[0], report.results[4]
    # La validación de la partición 1 es A 11,12,13 contra la predicción naive 10.
    assert fold1_naive.mae == pytest.approx(2.0)
    assert fold1_naive.wmape == pytest.approx(6.0 / 36.0)
    assert fold1_naive.requested == 3
    assert fold1_naive.coverage == pytest.approx(1.0)
    # La validación de la partición 2 es A 14,15,16 + C 14,15,16 contra la predicción naive 13.
    assert fold2_naive.mae == pytest.approx(2.0)
    assert fold2_naive.wmape == pytest.approx(6.0 / 45.0)
    assert fold2_naive.requested == 6
    assert fold2_naive.coverage == pytest.approx(0.5)


def test_insufficient_history_reported_with_product_ids():
    report = _run()
    for status in report.fold_statuses:
        ids = [entry.product_id for entry in status.insufficient_history]
        assert ids == ["C"]
        entry = status.insufficient_history[0]
        assert entry.observed_days == 0
        assert entry.observed_span_days == 0
        assert entry.required_days == 1
        assert entry.required_span_days == 1
        status_dict = status.to_dict()
        assert status_dict["insufficient_history"][0]["product_id"] == "C"
        assert status_dict["insufficient_history"][0]["observed_days"] == 0


def test_moving_average_window_and_ses_alpha_configurable():
    baseline = BaselineEvaluationConfig(moving_average_window=2, ses_alpha=1.0)
    report = _run(baseline=baseline)
    by_fold_model = {(result.fold, result.model): result for result in report.results}
    # Ventana 2 -> media de los dos últimos valores de AJUSTE (12, 13) = 12.5 para A en la partición 1.
    fold1_ma = by_fold_model[("fold-1", "moving_average")]
    assert fold1_ma.mae == pytest.approx((14 - 12.5 + 15 - 12.5 + 16 - 12.5) / 3.0)
    assert fold1_ma.mae == pytest.approx(2.5)
    # Partición 2 con ventana 2: últimos dos valores de ajuste de A (15, 16) -> 15.5; los de C -> 5.0.
    fold2_ma = by_fold_model[("fold-2", "moving_average")]
    assert fold2_ma.mae == pytest.approx(
        (17 - 15.5 + 18 - 15.5 + 19 - 15.5 + 0.0 + 0.0 + 0.0) / 6.0
    )
    # alpha == 1.0 colapsa SES a la última observación de AJUSTE (13, partición 1 de A).
    assert by_fold_model[("fold-1", "ses")].mae == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    # Partición 2 con alpha=1.0: último valor de ajuste de A 16; último valor de ajuste de C 5.0 (errores 0).
    assert by_fold_model[("fold-2", "ses")].mae == pytest.approx((1.0 + 2.0 + 3.0) / 6.0)
    assert report.results[0].config.baseline_config.moving_average_window == 2
    assert report.results[0].config.baseline_config.ses_alpha == pytest.approx(1.0)


    # --- protocolo de evaluación v2 (corrección del historial de ajuste) -------------


def test_protocol_validation_fit_uses_only_train():
    report = _run(evaluation_partition=EVALUATION_PARTITION_VALIDATION)
    fold1, fold2 = report.fold_statuses
    # El historial de ajuste es SOLO split.train: los conteos y los límites de fecha reales lo prueban.
    assert (fold1.fit_partition, fold2.fit_partition) == ("train", "train")
    assert (fold1.fit_row_count, fold2.fit_row_count) == (10, 13)
    assert (fold1.fit_history_start, fold1.fit_history_end) == (
        date(2024, 1, 1),
        date(2024, 1, 10),
    )
    assert (fold2.fit_history_start, fold2.fit_history_end) == (
        date(2024, 1, 1),
        date(2024, 1, 13),
    )
    for result in report.results:
        assert result.evaluation_protocol_version == EVALUATION_PROTOCOL_VERSION
        assert result.fit_partition == "train"
        assert result.test_excluded_from_fit is True
    # Prueba conductual: el naive de la partición 1 predice el ÚLTIMO valor de TRAIN (10),
    # no el de validación (13): el MAE 2.0 sobre 11,12,13 sería 5.0 bajo fuga.
    assert report.results[0].mae == pytest.approx(2.0)
    assert report.results[0].wmape == pytest.approx(6.0 / 36.0)


def test_protocol_test_fit_uses_train_plus_validation():
    report = _run()
    fold1, fold2 = report.fold_statuses
    # Ajuste = train + validation ordenados: los conteos y los límites reales lo prueban.
    assert (fold1.fit_partition, fold2.fit_partition) == (
        "train+validation",
        "train+validation",
    )
    assert (fold1.fit_row_count, fold2.fit_row_count) == (13, 19)
    assert (fold1.fit_history_start, fold1.fit_history_end) == (
        date(2024, 1, 1),
        date(2024, 1, 13),
    )
    assert (fold2.fit_history_start, fold2.fit_history_end) == (
        date(2024, 1, 1),
        date(2024, 1, 16),
    )
    for result in report.results:
        assert result.fit_partition == "train+validation"
    # Prueba conductual: el naive de la partición 1 predice el último valor de VALIDACIÓN (13)
    # y el naive de la partición 2 el último de validación (16) — no los de solo-train 10/13.
    by_fold_model = {(result.fold, result.model): result for result in report.results}
    assert by_fold_model[("fold-1", "naive")].mae == pytest.approx(2.0)
    assert by_fold_model[("fold-2", "naive")].mae == pytest.approx(1.0)


def test_protocol_test_rows_never_enter_fit():
    report = _run()
    assert all(
        status.test_excluded_from_fit is True for status in report.fold_statuses
    )
    assert all(result.test_excluded_from_fit is True for result in report.results)
    payload = json.loads(report.to_json())
    assert all(
        result["test_excluded_from_fit"] is True for result in payload["results"]
    )
    assert payload["protocol"]["test_excluded_from_fit"] is True
    # El invariante explícito de disjunción y orden ajuste/prueba se cumple sobre los
    # límites de ajuste visibles en el JSON: el ajuste termina estrictamente antes de la prueba.
    for status in report.fold_statuses:
        assert status.fit_history_end < status.test_start


def test_protocol_seasonal_naive_can_use_validation_lag_for_test():
    # Las fechas de prueba 02-18..02-20 necesitan rezagos de 7 días 02-11..02-13, que viven
    # SOLO en validación: un ajuste solo-train (protocolo antiguo) produce cobertura cero
    # del naive estacional para la prueba; el ajuste corregido train+validation debe predecir.
    fold = FoldEvaluationConfig(
        fold="lag-fold",
        train_start=date(2024, 2, 1),
        train_end=date(2024, 2, 10),
        validation_start=date(2024, 2, 11),
        validation_end=date(2024, 2, 13),
        test_start=date(2024, 2, 18),
        test_end=date(2024, 2, 20),
    )
    rows = [
        make_daily_row(
            product_id="A",
            dt=date(2024, 2, index).isoformat(),
            sale_amount=float(index),
            revision=EVALUATION_FIXTURE_REVISION,
        )
        for index in range(1, 21)
    ]
    config = _config(folds=(fold,))
    report = run_baseline_evaluation(rows, config)
    (status,) = report.fold_statuses
    assert status.fit_partition == "train+validation"
    assert status.fit_row_count == 13
    assert status.fit_history_end == date(2024, 2, 13)
    seasonal = next(r for r in report.results if r.model == "seasonal_naive")
    assert seasonal.available == 3
    assert seasonal.evaluated == 3
    assert seasonal.coverage == pytest.approx(1.0)
    # Predicciones 11,12,13 (rezagos de validación) contra reales 18,19,20.
    assert seasonal.mae == pytest.approx(7.0)

    # Prueba conductual de que las filas de validación entran al ajuste para la evaluación
    # de prueba: mutar las ventas de validación cambia la predicción naive al último valor
    # de ajuste mutado (100.0) en lugar de 13.0.
    mutated = [
        make_daily_row(
            product_id=row.product_id,
            dt=row.dt.isoformat(),
            sale_amount=100.0,
            stockout_hours_6_22=row.stockout_hours_6_22,
            revision=row.revision,
        )
        if date(2024, 2, 11) <= row.dt <= date(2024, 2, 13)
        else row
        for row in rows
    ]
    mutated_report = run_baseline_evaluation(mutated, config)
    naive = next(r for r in mutated_report.results if r.model == "naive")
    assert naive.mae == pytest.approx((82.0 + 81.0 + 80.0) / 3.0)


def test_protocol_duplicate_train_validation_keys_rejected():
    rows = list(make_evaluation_rows())
    duplicate = make_daily_row(
        product_id="A",
        dt="2024-01-05",  # A@01-05 ya existe en el fixture
        sale_amount=99.0,
        revision=EVALUATION_FIXTURE_REVISION,
    )
    with pytest.raises(DuplicateKeyError):
        run_baseline_evaluation(rows + [duplicate], _config())


def test_protocol_fit_and_evaluation_dates_and_counts_recorded():
    report = _run()
    for status in report.fold_statuses:
        assert status.evaluation_protocol_version == EVALUATION_PROTOCOL_VERSION
    fold1, fold2 = report.fold_statuses
    assert (fold1.fit_row_count, fold1.evaluation_rows) == (13, 6)
    assert (fold2.fit_row_count, fold2.evaluation_rows) == (19, 6)
    assert fold1.fit_history_start == date(2024, 1, 1)
    assert fold1.fit_history_end == date(2024, 1, 13)
    assert fold2.fit_history_end == date(2024, 1, 16)
    for result in report.results:
        assert result.evaluation_protocol_version == EVALUATION_PROTOCOL_VERSION
        assert result.test_excluded_from_fit is True
        if result.fold == "fold-1":
            assert result.fit_row_count == 13
            assert result.fit_history_end == date(2024, 1, 13)
        else:
            assert result.fit_row_count == 19
            assert result.fit_history_end == date(2024, 1, 16)
        assert result.fit_history_start == date(2024, 1, 1)


def test_protocol_report_json_deterministic_with_envelope():
    first, second = _run(), _run()
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["protocol"] == {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_partition": EVALUATION_PARTITION_TEST,
        "fit_partition": "train+validation",
        "test_excluded_from_fit": True,
        "baseline_parameters_fixed": True,
    }
    for status in payload["fold_statuses"]:
        assert status["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION
        assert status["fit_partition"] == "train+validation"
        assert status["test_excluded_from_fit"] is True
        assert isinstance(status["fit_row_count"], int)
        assert status["fit_history_start"] == "2024-01-01"
        assert status["fit_history_end"] in ("2024-01-13", "2024-01-16")
    for result in payload["results"]:
        assert result["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION
        assert result["fit_partition"] == "train+validation"
        assert result["test_excluded_from_fit"] is True
        assert result["fit_history_start"] == "2024-01-01"
        assert result["fit_history_end"] in ("2024-01-13", "2024-01-16")
        assert "insufficient_fit_history" in result
    # Las corridas con partición de validación llevan el sobre solo-train.
    validation = _run(evaluation_partition=EVALUATION_PARTITION_VALIDATION)
    validation_payload = json.loads(validation.to_json())
    assert validation_payload["protocol"]["fit_partition"] == "train"
    assert all(
        result["fit_partition"] == "train"
        for result in validation_payload["results"]
    )


def test_protocol_insufficient_fit_history_stays_explicit():
    fold = FoldEvaluationConfig(
        fold="hist-fold",
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 10),
        validation_start=date(2024, 1, 11),
        validation_end=date(2024, 1, 13),
        test_start=date(2024, 1, 14),
        test_end=date(2024, 1, 16),
        min_history_days=5,
        min_history_span_days=5,
    )
    rows = list(make_evaluation_rows()) + [
        make_daily_row(
            product_id="D",
            dt=date(2024, 1, index).isoformat(),
            sale_amount=3.0,
            revision=EVALUATION_FIXTURE_REVISION,
        )
        for index in (11, 12, 13, 14, 15, 16)
    ]
    report = run_baseline_evaluation(rows, _config(folds=(fold,)))
    (status,) = report.fold_statuses
    by_id = {entry.product_id: entry for entry in status.insufficient_fit_history}
    assert [entry.product_id for entry in status.insufficient_fit_history] == ["C", "D"]
    # C no tiene filas de ajuste en absoluto -> insuficiencia explícita de historial cero.
    assert (by_id["C"].observed_days, by_id["C"].observed_span_days) == (0, 0)
    assert by_id["C"].required_days == 5
    # D SÍ tiene filas de ajuste (validación 11..13) pero por debajo del mínimo de 5 días: el
    # estado se deriva de las filas de ajuste REALES (3 días), no del train (0).
    assert (by_id["D"].observed_days, by_id["D"].observed_span_days) == (3, 3)
    assert by_id["D"].required_days == 5
    assert by_id["D"].required_span_days == 5
    # El estado heredado derivado de la división sigue intacto y compatible.
    assert [entry.product_id for entry in status.insufficient_history] == ["C", "D"]
    legacy_d = next(e for e in status.insufficient_history if e.product_id == "D")
    assert legacy_d.observed_days == 0  # train-only, unchanged semantics
    for result in report.results:
        assert result.insufficient_fit_history == status.insufficient_fit_history


def test_protocol_no_automatic_winner_selection():
    report = _run()
    assert [result.model for result in report.results] == list(MODEL_ORDER) * 2
    assert {result.model for result in report.results} == set(MODEL_ORDER)
    raw = report.to_json()
    for forbidden in ("winner", "ranked", "best_model"):
        assert forbidden not in raw
    payload = json.loads(raw)
    assert payload["protocol"]["baseline_parameters_fixed"] is True
    for result in payload["results"]:
        assert "winner" not in result
        assert "rank" not in result


# --- validación de entrada ----------------------------------------------------------


def test_runner_rejects_mixed_revisions():
    rows = list(make_evaluation_rows())
    rows.append(
        make_daily_row(product_id="D", dt="2024-01-14", sale_amount=1.0, revision="other-rev")
    )
    with pytest.raises(ValueError, match="Mixed source revisions"):
        run_baseline_evaluation(rows, _config())


def test_runner_rejects_revision_mismatch():
    rows = [
        make_daily_row(
            product_id="A", dt="2024-01-01", sale_amount=1.0, revision="wrong-rev"
        )
    ]
    with pytest.raises(ValueError, match="revision"):
        run_baseline_evaluation(rows, _config())
    empty_revision_rows = [
        make_daily_row(product_id="A", dt="2024-01-01", sale_amount=1.0, revision="")
    ]
    with pytest.raises(ValueError, match="revision"):
        run_baseline_evaluation(empty_revision_rows, _config())


def test_config_rejects_invalid_partition_and_empty_folds_and_label():
    fold = FoldEvaluationConfig(
        fold="f1", train_start=date(2024, 1, 1), train_end=date(2024, 1, 10)
    )
    with pytest.raises(ValueError, match="partition"):
        EvaluationSplitConfig(folds=(fold,), evaluation_partition="train")
    with pytest.raises(ValueError, match="folds"):
        EvaluationSplitConfig(folds=())
    with pytest.raises(ValueError, match="product_group"):
        EvaluationSplitConfig(folds=(fold,), product_group="   ")


def test_config_rejects_missing_duplicate_unknown_models_and_invalid_params():
    with pytest.raises(ValueError, match="missing"):
        BaselineEvaluationConfig(models=("naive", "ses"))
    with pytest.raises(ValueError, match="duplicate"):
        BaselineEvaluationConfig(
            models=("naive", "naive", "seasonal_naive", "moving_average", "ses")
        )
    with pytest.raises(ValueError, match="unknown"):
        BaselineEvaluationConfig(
            models=("naive", "seasonal_naive", "moving_average", "ses", "prophet")
        )
    with pytest.raises(ValueError, match="window"):
        BaselineEvaluationConfig(moving_average_window=0)
    with pytest.raises(ValueError, match="alpha"):
        BaselineEvaluationConfig(ses_alpha=-0.01)
    with pytest.raises(ValueError, match="alpha"):
        BaselineEvaluationConfig(ses_alpha=1.5)


def test_config_rejects_empty_identifiers():
    split = EvaluationSplitConfig(folds=_default_folds())
    baseline = BaselineEvaluationConfig()
    with pytest.raises(ValueError, match="source_id"):
        EvaluationConfig(
            source_id="  ", dataset_revision="r", evaluation_id="e",
            split_config=split, baseline_config=baseline,
        )
    with pytest.raises(ValueError, match="dataset_revision"):
        EvaluationConfig(
            source_id="s", dataset_revision="", evaluation_id="e",
            split_config=split, baseline_config=baseline,
        )
    with pytest.raises(ValueError, match="evaluation_id"):
        EvaluationConfig(
            source_id="s", dataset_revision="r", evaluation_id=" ",
            split_config=split, baseline_config=baseline,
        )


def test_fold_config_rejects_single_boundary_and_empty_label():
    with pytest.raises(ValueError, match="together"):
        FoldEvaluationConfig(
            fold="f1",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 10),
            test_start=date(2024, 1, 11),
        )
    with pytest.raises(ValueError, match="fold"):
        FoldEvaluationConfig(
            fold="   ", train_start=date(2024, 1, 1), train_end=date(2024, 1, 10)
        )
    with pytest.raises(ValueError, match="min_history_days"):
        FoldEvaluationConfig(
            fold="f1",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 10),
            min_history_days=0,
        )


def test_models_normalized_to_required_order():
    baseline = BaselineEvaluationConfig(
        models=["ses", "naive", "moving_average", "seasonal_naive"]
    )
    assert baseline.models == MODEL_ORDER
    report = _run(baseline=baseline)
    assert [result.model for result in report.results] == list(MODEL_ORDER) * 2


# --- serialización JSON ------------------------------------------------------------


def test_report_json_is_deterministic_and_raw_data_free():
    report = _run()
    payload = json.loads(report.to_json())
    assert payload["config"]["source_id"] == EVALUATION_FIXTURE_SOURCE_ID
    assert payload["config"]["dataset_revision"] == EVALUATION_FIXTURE_REVISION
    assert payload["config"]["evaluation_id"] == EVALUATION_FIXTURE_EVALUATION_ID
    assert payload["config"]["evaluation_partition"] == "test"
    assert payload["config"]["product_group"] == "all_products"
    assert payload["config"]["models"] == list(MODEL_ORDER)
    assert payload["config"]["folds"][0]["train_start"] == "2024-01-01"
    assert payload["config"]["folds"][0]["test_end"] == "2024-01-16"
    assert payload["protocol"] == {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_partition": "test",
        "fit_partition": "train+validation",
        "test_excluded_from_fit": True,
        "baseline_parameters_fixed": True,
    }
    assert len(payload["results"]) == 8
    assert len(payload["fold_statuses"]) == 2
    result = payload["results"][0]
    assert result["model"] == "naive"
    assert result["fold"] == "fold-1"
    assert result["evaluation_partition"] == "test"
    assert result["target"] == "observed_sales"
    assert result["train_start"] == "2024-01-01"
    assert result["test_end"] == "2024-01-16"
    assert result["requested"] == 6
    assert result["available"] == 3
    assert result["evaluated"] == 3
    assert result["coverage"] == pytest.approx(0.5)
    assert result["stockout_rows"] == 1
    assert isinstance(result["mae"], float)
    assert result["mae"] == pytest.approx(2.0)
    assert result["wmape_status"] == WMAPE_CALCULABLE
    assert result["configuration"]["dataset_revision"] == EVALUATION_FIXTURE_REVISION
    assert result["configuration"]["moving_average_window"] == 7
    assert result["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION
    assert result["fit_partition"] == "train+validation"
    assert result["fit_row_count"] == 13
    assert result["fit_history_start"] == "2024-01-01"
    assert result["fit_history_end"] == "2024-01-13"
    assert result["test_excluded_from_fit"] is True
    assert result["insufficient_fit_history"] == [
        {
            "product_id": "C",
            "observed_days": 0,
            "observed_span_days": 0,
            "required_days": 1,
            "required_span_days": 1,
        }
    ]
    assert payload["fold_statuses"][0]["evaluable"] is True
    assert payload["fold_statuses"][0]["test_rows"] == 6
    assert payload["fold_statuses"][0]["excluded_out_of_range"] == 7
    assert payload["fold_statuses"][0]["fit_row_count"] == 13
    assert payload["fold_statuses"][0]["fit_history_start"] == "2024-01-01"
    assert payload["fold_statuses"][0]["fit_history_end"] == "2024-01-13"

    raw = report.to_json()
    for forbidden in (
        "hours_sale",
        "hours_stock_status",
        "sale_amount",
        "stock_hour6_22_cnt",
        "stockout_hours_6_22",
        "latent_demand_estimate",
        "prediction",
    ):
        assert forbidden not in raw

    # Round-trip: el parseo JSON reproduce to_dict exactamente; las corridas son byte-estables.
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.to_json() == _run().to_json()


# --- validación de entrada endurecida (refinamiento de la puerta de control) --------


def test_fold_config_rejects_invalid_fold_identifier_types():
    for bad in (None, True, False, 3.5, ["f1"], {"fold": 1}):
        with pytest.raises(ValueError, match="fold identifier"):
            FoldEvaluationConfig(
                fold=bad, train_start=date(2024, 1, 1), train_end=date(2024, 1, 10)
            )


def test_fold_config_rejects_invalid_history_minimums():
    for bad in (True, False, 0, -1, 1.5, "7", None):
        with pytest.raises(ValueError, match="min_history_days"):
            FoldEvaluationConfig(
                fold="f1",
                train_start=date(2024, 1, 1),
                train_end=date(2024, 1, 10),
                min_history_days=bad,
            )
    with pytest.raises(ValueError, match="min_history_span_days"):
        FoldEvaluationConfig(
            fold="f1",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 10),
            min_history_span_days=2.5,
        )


def test_split_config_rejects_non_string_product_group():
    fold = FoldEvaluationConfig(
        fold="f1", train_start=date(2024, 1, 1), train_end=date(2024, 1, 10)
    )
    for bad in (None, 5, True, ["all"]):
        with pytest.raises(ValueError, match="product_group"):
            EvaluationSplitConfig(folds=(fold,), product_group=bad)


def test_split_config_rejects_non_fold_entries():
    with pytest.raises(ValueError, match="FoldEvaluationConfig"):
        EvaluationSplitConfig(folds=(date(2024, 1, 1),))
    with pytest.raises(ValueError, match="FoldEvaluationConfig"):
        EvaluationSplitConfig(folds=(object(),))


def test_split_config_rejects_duplicate_fold_identifiers():
    first = FoldEvaluationConfig(
        fold="fold-1", train_start=date(2024, 1, 1), train_end=date(2024, 1, 10)
    )
    second = FoldEvaluationConfig(
        fold="fold-1", train_start=date(2024, 1, 5), train_end=date(2024, 1, 15)
    )
    with pytest.raises(ValueError, match="duplicate fold"):
        EvaluationSplitConfig(folds=(first, second))


def test_split_config_preserves_input_fold_order():
    fold1, fold2 = _default_folds()
    report = run_baseline_evaluation(
        make_evaluation_rows(), _config(folds=(fold2, fold1))
    )
    assert [result.fold for result in report.results] == ["fold-2"] * 4 + ["fold-1"] * 4
    assert [status.fold for status in report.fold_statuses] == ["fold-2", "fold-1"]


def test_baseline_config_rejects_invalid_window_types():
    for bad in (True, False, 1.5, "7", None, [7]):
        with pytest.raises(ValueError, match="moving_average_window"):
            BaselineEvaluationConfig(moving_average_window=bad)


def test_baseline_config_rejects_invalid_alpha_types():
    for bad in (True, False, "0.3", None, [0.3]):
        with pytest.raises(ValueError, match="ses_alpha"):
            BaselineEvaluationConfig(ses_alpha=bad)
    # Los valores alpha integrales siguen siendo válidos.
    assert BaselineEvaluationConfig(ses_alpha=1).ses_alpha == 1


def test_baseline_config_rejects_non_string_model_entries():
    with pytest.raises(ValueError, match="models"):
        BaselineEvaluationConfig(
            models=("naive", "seasonal_naive", "moving_average", 5)
        )
    # Una entrada no textual junto con un nombre desconocido no debe romper el ordenamiento.
    with pytest.raises(ValueError, match="models"):
        BaselineEvaluationConfig(
            models=("naive", "seasonal_naive", "moving_average", None, "prophet")
        )
    with pytest.raises(ValueError, match="models"):
        BaselineEvaluationConfig(models="naive")
    with pytest.raises(ValueError, match="models"):
        BaselineEvaluationConfig(models=5)


def test_evaluation_config_rejects_wrong_nested_types():
    split = EvaluationSplitConfig(folds=_default_folds())
    baseline = BaselineEvaluationConfig()
    with pytest.raises(ValueError, match="split_config"):
        EvaluationConfig(
            source_id="s",
            dataset_revision="r",
            evaluation_id="e",
            split_config={"folds": ()},
            baseline_config=baseline,
        )
    with pytest.raises(ValueError, match="baseline_config"):
        EvaluationConfig(
            source_id="s",
            dataset_revision="r",
            evaluation_id="e",
            split_config=split,
            baseline_config=None,
        )
