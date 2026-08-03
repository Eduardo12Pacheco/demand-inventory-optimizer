"""Pruebas de líneas base de pronóstico: naive determinista / naive estacional de
7 días / promedio móvil / SES con alpha configurable, ajuste exclusivo con
entrenamiento, predicciones no disponibles explícitas con contabilidad de
cobertura, preservación de metadatos y protecciones de objetivos futuros."""

from __future__ import annotations

from datetime import date

import pytest

from inventory_optimizer.forecasting.baselines import (
    BASELINES,
    REASON_NO_HISTORY,
    REASON_NO_SEASONAL_LAG,
    TargetNotInFutureError,
    fit_baseline,
    fit_moving_average,
    fit_naive,
    fit_seasonal_naive,
    fit_ses,
)
from inventory_optimizer.forecasting.targets import project_targets

from helpers import make_daily_row, make_series


def _train_rows():
    # sale_amount = 1.0..10.0 entre 2024-01-01 y 2024-01-10
    return make_series(product_id="A", start="2024-01-01", days=10)


def _target_rows():
    return project_targets(make_series(product_id="A", start="2024-01-11", days=3))


def test_naive_uses_last_observed_training_value():
    result = fit_naive(_train_rows(), _target_rows())
    assert result.model == "naive"
    assert len(result.rows) == 3
    assert all(row.prediction == pytest.approx(10.0) for row in result.rows)
    assert all(row.prediction_state == "available" for row in result.rows)
    assert [row.dt for row in result.rows] == [
        date(2024, 1, 11),
        date(2024, 1, 12),
        date(2024, 1, 13),
    ]
    assert result.requested == 3
    assert result.available == 3
    assert result.unavailable == 0
    assert result.coverage == pytest.approx(1.0)


def test_naive_unavailable_without_training_history():
    train = _train_rows()
    targets = project_targets(
        [
            make_daily_row(product_id="B", dt="2024-01-11", sale_amount=5.0),
            make_daily_row(product_id="A", dt="2024-01-11", sale_amount=1.0),
        ]
    )
    result = fit_naive(train, targets)
    rows_by_product = {row.product_id: row for row in result.rows}
    assert rows_by_product["A"].prediction == pytest.approx(10.0)
    assert rows_by_product["B"].prediction is None
    assert rows_by_product["B"].prediction_state == "unavailable"
    assert rows_by_product["B"].unavailable_reason == REASON_NO_HISTORY
    assert result.requested == 2
    assert result.available == 1
    assert result.unavailable == 1
    assert result.coverage == pytest.approx(0.5)


def test_seasonal_naive_uses_exact_seven_day_lag_from_training_only():
    train = _train_rows()  # 2024-01-01..01-10 with values 1.0..10.0
    targets = project_targets(
        [
            make_daily_row(product_id="A", dt="2024-01-17", sale_amount=999.0),
            make_daily_row(product_id="A", dt="2024-01-18", sale_amount=999.0),
        ]
    )
    result = fit_seasonal_naive(train, targets)
    # 2024-01-17 - 7 días = 2024-01-10, dentro del entrenamiento: predicción 10.0.
    assert result.rows[0].prediction == pytest.approx(10.0)
    # El real adjunto (999.0) se conserva pero nunca se usa para predecir.
    assert result.rows[0].observed_sales == pytest.approx(999.0)
    # 2024-01-18 - 7 días = 2024-01-11, NO en entrenamiento: explícitamente no disponible.
    assert result.rows[1].prediction is None
    assert result.rows[1].prediction_state == "unavailable"
    assert result.rows[1].unavailable_reason == REASON_NO_SEASONAL_LAG
    assert result.available == 1
    assert result.unavailable == 1


def test_seasonal_naive_reports_no_history_for_unknown_product():
    train = _train_rows()
    targets = project_targets([make_daily_row(product_id="Z", dt="2024-01-17", sale_amount=1.0)])
    (row,) = fit_seasonal_naive(train, targets).rows
    assert row.prediction is None
    assert row.unavailable_reason == REASON_NO_HISTORY


def test_moving_average_uses_last_window_training_values():
    train = _train_rows()  # 1.0..10.0
    result = fit_moving_average(train, _target_rows(), window=3)
    expected = (8.0 + 9.0 + 10.0) / 3.0
    assert all(row.prediction == pytest.approx(expected) for row in result.rows)
    # Una ventana mayor que la historia disponible usa todos los valores observados.
    wide = fit_moving_average(train, _target_rows(), window=10)
    assert all(row.prediction == pytest.approx(5.5) for row in wide.rows)


def test_moving_average_rejects_invalid_window():
    with pytest.raises(ValueError, match="window"):
        fit_moving_average(_train_rows(), _target_rows(), window=0)


def test_ses_forecasts_from_final_level():
    train = _train_rows()
    result = fit_ses(train, _target_rows(), alpha=0.5)
    level = 1.0
    for value in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        level = 0.5 * value + 0.5 * level
    assert all(row.prediction == pytest.approx(level) for row in result.rows)
    # alpha == 1.0 colapsa SES a la última observación.
    last_only = fit_ses(train, _target_rows(), alpha=1.0)
    assert all(row.prediction == pytest.approx(10.0) for row in last_only.rows)
    # alpha == 0.0 conserva la primera observación para siempre.
    first_only = fit_ses(train, _target_rows(), alpha=0.0)
    assert all(row.prediction == pytest.approx(1.0) for row in first_only.rows)


def test_ses_rejects_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        fit_ses(_train_rows(), _target_rows(), alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        fit_ses(_train_rows(), _target_rows(), alpha=1.1)


def test_baselines_never_use_future_target_actuals():
    train = _train_rows()
    targets_plain = project_targets(make_series(product_id="A", start="2024-01-11", days=3))
    targets_mutated = project_targets(
        [
            make_daily_row(product_id="A", dt=dt, sale_amount=999.0)
            for dt in ("2024-01-11", "2024-01-12", "2024-01-13")
        ]
    )
    for name in ("naive", "seasonal_naive", "moving_average", "ses"):
        plain = fit_baseline(name, train, targets_plain)
        mutated = fit_baseline(name, train, targets_mutated)
        assert [row.prediction for row in plain.rows] == [
            row.prediction for row in mutated.rows
        ]
        assert all(
            row.observed_sales == pytest.approx(999.0) for row in mutated.rows
        )


def test_seasonal_naive_lag_on_missing_training_date_is_unavailable():
    # El entrenamiento salta el 2024-01-05: un objetivo en 2024-01-12 necesita esa
    # fecha de rezago exacta y debe quedar explícitamente no disponible en lugar de
    # rellenarse o desplazarse.
    train = [
        row
        for row in make_series(product_id="A", start="2024-01-01", days=9)
        if row.dt != date(2024, 1, 5)
    ]
    targets = project_targets([make_daily_row(product_id="A", dt="2024-01-12", sale_amount=1.0)])
    (row,) = fit_seasonal_naive(train, targets).rows
    assert row.prediction is None
    assert row.unavailable_reason == REASON_NO_SEASONAL_LAG


def test_ses_single_training_row_predicts_that_value():
    train = [make_daily_row(product_id="A", dt="2024-01-01", sale_amount=7.0)]
    targets = _target_rows()
    for alpha in (0.0, 0.3, 1.0):
        result = fit_ses(train, targets, alpha=alpha)
        assert all(row.prediction == pytest.approx(7.0) for row in result.rows)


def test_targets_inside_training_period_are_rejected():
    train = _train_rows()  # training ends 2024-01-10
    targets = project_targets([make_daily_row(product_id="A", dt="2024-01-10", sale_amount=1.0)])
    with pytest.raises(TargetNotInFutureError):
        fit_naive(train, targets)
    with pytest.raises(TargetNotInFutureError):
        fit_seasonal_naive(train, targets)
    with pytest.raises(TargetNotInFutureError):
        fit_moving_average(train, targets)
    with pytest.raises(TargetNotInFutureError):
        fit_ses(train, targets)


def test_forecast_rows_preserve_target_metadata():
    train = _train_rows()
    targets = project_targets(
        [make_daily_row(product_id="A", dt="2024-01-11", sale_amount=4.0, stockout_hours_6_22=2, revision="rev-9")]
    )
    (row,) = fit_naive(train, targets).rows
    assert row.model == "naive"
    assert row.product_id == "A"
    assert row.dt == date(2024, 1, 11)
    assert row.observed_sales == pytest.approx(4.0)
    assert row.stockout_hours_6_22 == 2
    assert row.observation_state == "censored_or_partial"
    assert row.revision == "rev-9"
    assert row.unavailable_reason is None


def test_all_baselines_unavailable_for_product_without_history():
    train = _train_rows()
    targets = project_targets([make_daily_row(product_id="Z", dt="2024-01-11", sale_amount=1.0)])
    for name in ("naive", "seasonal_naive", "moving_average", "ses"):
        (row,) = fit_baseline(name, train, targets).rows
        assert row.prediction is None
        assert row.prediction_state == "unavailable"
        assert row.unavailable_reason == REASON_NO_HISTORY


def test_fit_baseline_dispatches_by_name_and_rejects_unknown():
    train = _train_rows()
    targets = _target_rows()
    assert fit_baseline("naive", train, targets).model == "naive"
    assert set(BASELINES) == {"naive", "seasonal_naive", "moving_average", "ses"}
    with pytest.raises(ValueError, match="Unknown baseline"):
        fit_baseline("prophet", train, targets)
