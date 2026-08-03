"""Pruebas de proyección de objetivos de ventas observadas: preservación de campos,
derivación del estado de observación desde el vector de estado, orden determinista,
rechazo de duplicados y la estimación de demanda latente nunca poblada."""

from __future__ import annotations

from datetime import date

import pytest

from inventory_optimizer.ingestion.fresh_retail import DailySourceRow
from inventory_optimizer.forecasting.targets import (
    DuplicateKeyError,
    TargetRow,
    project_targets,
)

from helpers import make_daily_row, make_series


def test_project_targets_preserves_identity_fields():
    row = make_daily_row(
        product_id="P1", dt="2024-01-15", sale_amount=37.5, stockout_hours_6_22=3, revision="abc123"
    )
    (target,) = project_targets([row])
    assert isinstance(target, TargetRow)
    assert target.product_id == "P1"
    assert target.dt == date(2024, 1, 15)
    assert target.observed_sales == pytest.approx(37.5)
    assert target.stockout_hours_6_22 == 3
    assert target.revision == "abc123"
    assert target.latent_demand_estimate is None


def test_project_targets_observation_state_from_status_vector():
    censored = make_daily_row(product_id="P1", dt="2024-01-15", sale_amount=1.0, stockout_hours_6_22=1)
    uncensored = make_daily_row(product_id="P2", dt="2024-01-15", sale_amount=1.0, stockout_hours_6_22=0)
    (first, second) = project_targets([uncensored, censored])  # input order must not matter
    assert first.product_id == "P1"
    assert first.observation_state == "censored_or_partial"
    assert second.product_id == "P2"
    assert second.observation_state == "uncensored"


def test_project_targets_any_status_one_means_censored():
    # status == 1 solo en el índice 23 (fuera de la ventana [6:22)): la fila es
    # censurada/parcial, pero stockout_hours_6_22 permanece en 0.
    status = [0] * 24
    status[23] = 1
    row = DailySourceRow(
        product_id="P1",
        dt=date(2024, 1, 15),
        sale_amount=2.0,
        hours_sale=(2.0,) + (0.0,) * 23,
        hours_stock_status=tuple(status),
        stock_hour6_22_cnt=0,
        stockout_hours_6_22=0,
        revision="rev-1",
    )
    (target,) = project_targets([row])
    assert target.observation_state == "censored_or_partial"
    assert target.stockout_hours_6_22 == 0


def test_project_targets_sorts_deterministically_and_rejects_duplicates():
    rows = [
        make_daily_row(product_id="B", dt="2024-01-02", sale_amount=2.0),
        make_daily_row(product_id="A", dt="2024-01-02", sale_amount=1.0),
        make_daily_row(product_id="A", dt="2024-01-01", sale_amount=0.5),
    ]
    targets = project_targets(rows)
    assert [(target.product_id, target.dt) for target in targets] == [
        ("A", date(2024, 1, 1)),
        ("A", date(2024, 1, 2)),
        ("B", date(2024, 1, 2)),
    ]
    with pytest.raises(DuplicateKeyError):
        project_targets(rows + [make_daily_row(product_id="A", dt="2024-01-02", sale_amount=9.0)])


def test_project_targets_preserves_zero_sales():
    row = make_daily_row(product_id="P1", dt="2024-01-15", sale_amount=0.0)
    (target,) = project_targets([row])
    assert target.observed_sales == 0.0
    assert target.observed_sales is not None


def test_project_targets_never_creates_latent_estimate():
    targets = project_targets(make_series(product_id="A", start="2024-01-01", days=3))
    assert len(targets) == 3
    assert all(target.latent_demand_estimate is None for target in targets)
