"""Pruebas de métricas de pronóstico: valores conocidos de MAE/RMSE/WMAPE/sesgo,
WMAPE explícito de denominador cero, manejo explícito de predicciones no
evaluables, conteo de filas con quiebre, cobertura y los campos del reporte tipado."""

from __future__ import annotations

import math
from datetime import date

import pytest

from inventory_optimizer.forecasting.baselines import BaselineForecastResult, ForecastRow, fit_naive
from inventory_optimizer.forecasting.metrics import ForecastMetrics, evaluate_metrics
from inventory_optimizer.forecasting.targets import project_targets

from helpers import make_daily_row


def _row(
    model: str,
    product_id: str | int,
    dt: str,
    observed: float | None,
    prediction: float | None,
    stockout: int = 0,
    state: str = "uncensored",
    revision: str = "rev-1",
) -> ForecastRow:
    return ForecastRow(
        model=model,
        product_id=product_id,
        dt=date.fromisoformat(dt),
        observed_sales=observed,
        prediction=prediction,
        prediction_state="available" if prediction is not None else "unavailable",
        unavailable_reason=None if prediction is not None else "no training history for product",
        stockout_hours_6_22=stockout,
        observation_state=state,
        revision=revision,
    )


def _result(rows: list[ForecastRow], model: str = "naive") -> BaselineForecastResult:
    requested = len(rows)
    available = sum(1 for row in rows if row.prediction is not None)
    return BaselineForecastResult(
        model=model,
        rows=tuple(rows),
        requested=requested,
        available=available,
        unavailable=requested - available,
        coverage=available / requested if requested else 0.0,
    )


def test_metrics_known_mae_rmse_wmape_bias_values():
    rows = [
        _row("naive", "A", "2024-01-11", 10.0, 10.0),
        _row("naive", "A", "2024-01-12", 10.0, 12.0),
        _row("naive", "A", "2024-01-13", 10.0, 9.0),
    ]
    metrics = evaluate_metrics(_result(rows), fold="fold-1")
    assert isinstance(metrics, ForecastMetrics)
    assert metrics.model == "naive"
    assert metrics.fold == "fold-1"
    assert metrics.target == "observed_sales"
    assert metrics.requested == 3
    assert metrics.available == 3
    assert metrics.evaluated == 3
    assert metrics.coverage == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(math.sqrt(5.0 / 3.0))
    assert metrics.wmape == pytest.approx(3.0 / 30.0)
    assert metrics.bias == pytest.approx((0.0 + 2.0 - 1.0) / 3.0)
    assert metrics.wmape_status == "calculable"
    assert metrics.stockout_rows == 0


def test_metrics_zero_wmape_denominator_is_explicit():
    rows = [
        _row("naive", "A", "2024-01-11", 0.0, 3.0),
        _row("naive", "A", "2024-01-12", 0.0, -1.0),
    ]
    metrics = evaluate_metrics(_result(rows))
    assert metrics.wmape is None
    assert metrics.wmape_status == "not_calculable_zero_actuals"
    assert metrics.mae == pytest.approx(2.0)
    assert metrics.rmse == pytest.approx(math.sqrt((9.0 + 1.0) / 2.0))
    assert metrics.bias == pytest.approx(1.0)


def test_metrics_no_evaluable_predictions_is_explicit():
    rows = [
        _row("naive", "A", "2024-01-11", 5.0, None),
        _row("naive", "A", "2024-01-12", None, 7.0),
    ]
    metrics = evaluate_metrics(_result(rows))
    assert metrics.requested == 2
    assert metrics.available == 1
    assert metrics.evaluated == 0
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.stockout_rows == 0
    assert metrics.mae is None
    assert metrics.rmse is None
    assert metrics.wmape is None
    assert metrics.bias is None
    assert metrics.wmape_status == "no_evaluable_predictions"


def test_metrics_counts_stockout_rows_among_evaluated_predictions():
    rows = [
        _row("naive", "A", "2024-01-11", 10.0, 10.0, stockout=3),
        _row("naive", "A", "2024-01-12", 10.0, 9.0, stockout=0),
        # Los metadatos de quiebre en una predicción no disponible NO se cuentan:
        # solo califican las predicciones evaluadas.
        _row("naive", "B", "2024-01-11", 10.0, None, stockout=5),
    ]
    metrics = evaluate_metrics(_result(rows))
    assert metrics.evaluated == 2
    assert metrics.stockout_rows == 1


def test_metrics_wmape_uses_absolute_actuals_in_denominator():
    rows = [
        _row("naive", "A", "2024-01-11", -2.0, 0.0),
        _row("naive", "A", "2024-01-12", 2.0, 0.0),
    ]
    metrics = evaluate_metrics(_result(rows))
    # errores: [2.0, 2.0]; denominador: |-2| + |2| = 4.0
    assert metrics.wmape == pytest.approx(4.0 / 4.0)
    assert metrics.mae == pytest.approx(2.0)
    assert metrics.bias == pytest.approx(-0.0)
    assert metrics.wmape_status == "calculable"


def test_metrics_empty_result_never_fabricates_numbers():
    metrics = evaluate_metrics(_result([]))
    assert metrics.requested == 0
    assert metrics.available == 0
    assert metrics.evaluated == 0
    assert metrics.coverage == pytest.approx(0.0)
    assert metrics.stockout_rows == 0
    assert metrics.mae is None
    assert metrics.rmse is None
    assert metrics.wmape is None
    assert metrics.bias is None
    assert metrics.wmape_status == "no_evaluable_predictions"


def test_metrics_over_a_real_baseline_result():
    train = [
        make_daily_row(product_id="A", dt=f"2024-01-0{index}", sale_amount=float(index))
        for index in range(1, 6)
    ]
    targets = project_targets(
        [
            make_daily_row(product_id="A", dt="2024-01-10", sale_amount=3.0, stockout_hours_6_22=1),
            make_daily_row(product_id="B", dt="2024-01-10", sale_amount=1.0),
        ]
    )
    result = fit_naive(train, targets)
    metrics = evaluate_metrics(result, fold=2)
    assert metrics.model == "naive"
    assert metrics.fold == 2
    assert metrics.target == "observed_sales"
    assert metrics.requested == 2
    assert metrics.available == 1
    assert metrics.evaluated == 1
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.stockout_rows == 1
    assert metrics.mae == pytest.approx(2.0)  # |3.0 - 5.0|
    assert metrics.wmape == pytest.approx(2.0 / 3.0)
    assert metrics.bias == pytest.approx(2.0)
    assert metrics.rmse == pytest.approx(2.0)
    assert metrics.wmape_status == "calculable"
