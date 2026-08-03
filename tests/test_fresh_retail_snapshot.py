"""Pruebas de validación a nivel de instantánea global (invariante de 865 productos únicos).

La validación a nivel de fila es separada; este módulo solo ejercita el invariante
global y el parámetro explícito de contexto instantánea-completa / fixture.
"""

from __future__ import annotations

import pytest

from inventory_optimizer.ingestion.fresh_retail import (
    GlobalSnapshotError,
    SnapshotContext,
    parse_daily_row,
    validate_global_snapshot,
)

from helpers import make_raw_row


def _rows(product_count: int) -> list:
    return [
        parse_daily_row(make_raw_row(product_id=f"P{index:04d}"))
        for index in range(product_count)
    ]


def test_fixture_context_does_not_require_865_products():
    summary = validate_global_snapshot(_rows(2), context=SnapshotContext.FIXTURE)
    assert summary.total_rows == 2
    assert summary.unique_products == 2


def test_mixed_numeric_and_string_product_ids_count_once():
    # Normalización numérica: 38 (int64) y "38" son el mismo producto.
    rows = [
        parse_daily_row(make_raw_row(product_id=38)),
        parse_daily_row(make_raw_row(product_id="38")),
        parse_daily_row(make_raw_row(product_id=39)),
    ]
    summary = validate_global_snapshot(rows, context=SnapshotContext.FIXTURE)
    assert summary.total_rows == 3
    assert summary.unique_products == 2


def test_complete_snapshot_accepts_865_unique_products():
    summary = validate_global_snapshot(_rows(865), context=SnapshotContext.COMPLETE)
    assert summary.total_rows == 865
    assert summary.unique_products == 865


def test_complete_snapshot_rejects_fewer_than_865_products():
    with pytest.raises(GlobalSnapshotError):
        validate_global_snapshot(_rows(864), context=SnapshotContext.COMPLETE)


def test_complete_snapshot_rejects_more_than_865_products():
    with pytest.raises(GlobalSnapshotError):
        validate_global_snapshot(_rows(866), context=SnapshotContext.COMPLETE)


def test_complete_snapshot_counts_unique_product_ids():
    duplicates = [
        parse_daily_row(make_raw_row(product_id=f"P{index % 10:04d}"))
        for index in range(865)
    ]
    with pytest.raises(GlobalSnapshotError):
        validate_global_snapshot(duplicates, context=SnapshotContext.COMPLETE)


def test_global_validation_rejects_empty_snapshot():
    with pytest.raises(GlobalSnapshotError):
        validate_global_snapshot([], context=SnapshotContext.FIXTURE)


def test_context_parameter_is_type_safe():
    with pytest.raises(TypeError):
        validate_global_snapshot(_rows(1), context="complete")
