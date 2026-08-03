"""Pruebas offline para el cargador de streaming acotado de FreshRetailNet-50K.

La ruta en vivo de Hugging Face se ejercita con un módulo ``datasets`` falso
inyectado (sin red, sin instalar paquetes); todas las demás pruebas inyectan un
cargador cliente. Ninguna prueba descarga o materializa el conjunto de datos
real.
"""

from __future__ import annotations

import sys
import types

import pytest

from inventory_optimizer.ingestion.fresh_retail import (
    FRESH_RETAIL_PINNED_REVISION,
    FreshRetailDomainError,
    HourlyVectorLengthError,
    UnpinnedRevisionError,
    expand_to_hourly,
)
from inventory_optimizer.ingestion.fresh_retail_stream import (
    FRESH_RETAIL_DATASET_ID,
    FRESH_RETAIL_STREAM_MAX_LIMIT,
    InvalidStreamLimitError,
    NonPinnedRevisionError,
    live_hf_stream_loader,
    stream_fresh_retail_50k,
)

from helpers import make_raw_row


def _loader(rows):
    """Construye un cargador inyectado que registra cada llamada (revision, split)."""
    calls: list[tuple[str, str]] = []

    def loader(revision: str, split: str):
        calls.append((revision, split))
        return iter(rows)

    return loader, calls


def test_stream_rejects_zero_negative_and_non_integer_limits():
    for bad in (0, -1, 10.0, "10", None, True):
        with pytest.raises(InvalidStreamLimitError):
            stream_fresh_retail_50k(limit=bad, client=lambda r, s: iter([]))


def test_stream_hard_max_is_1000_rows():
    assert FRESH_RETAIL_STREAM_MAX_LIMIT == 1000
    with pytest.raises(InvalidStreamLimitError):
        stream_fresh_retail_50k(
            limit=FRESH_RETAIL_STREAM_MAX_LIMIT + 1, client=lambda r, s: iter([])
        )
    # exactamente el tope duro se acepta
    rows = list(
        stream_fresh_retail_50k(
            limit=FRESH_RETAIL_STREAM_MAX_LIMIT, client=lambda r, s: iter([])
        )
    )
    assert rows == []


def test_stream_rejects_empty_whitespace_none_and_non_string_revisions():
    for bad in ("", "   ", None, 123, b"sha"):
        with pytest.raises(UnpinnedRevisionError):
            stream_fresh_retail_50k(
                limit=1, revision=bad, client=lambda r, s: iter([])
            )


def test_stream_rejects_non_pinned_revision_without_calling_client():
    loader, calls = _loader([make_raw_row()])
    with pytest.raises(NonPinnedRevisionError):
        stream_fresh_retail_50k(limit=1, revision="deadbeef" * 5, client=loader)
    assert calls == []


def test_stream_passes_only_pinned_revision_and_explicit_split():
    loader, calls = _loader([make_raw_row(product_id="P0042")])
    rows = list(stream_fresh_retail_50k(limit=1, client=loader))
    assert calls == [(FRESH_RETAIL_PINNED_REVISION, "train")]
    assert [row.product_id for row in rows] == ["P0042"]
    assert rows[0].revision == FRESH_RETAIL_PINNED_REVISION

    loader_eval, calls_eval = _loader([make_raw_row()])
    list(stream_fresh_retail_50k(limit=1, split="eval", client=loader_eval))
    assert calls_eval == [(FRESH_RETAIL_PINNED_REVISION, "eval")]


def test_stream_yields_exactly_limit_rows():
    loader, _ = _loader([make_raw_row(product_id=f"P{i:04d}") for i in range(5)])
    rows = list(stream_fresh_retail_50k(limit=3, client=loader))
    assert [row.product_id for row in rows] == ["P0000", "P0001", "P0002"]


def test_stream_stops_at_limit_without_materializing_source():
    produced: list[int] = []

    def source():
        for i in range(5):
            produced.append(i)
            yield make_raw_row(product_id=f"P{i:04d}")

    rows = list(stream_fresh_retail_50k(limit=3, client=lambda r, s: source()))
    assert [row.product_id for row in rows] == ["P0000", "P0001", "P0002"]
    assert produced == [0, 1, 2]  # la fuente nunca llegó a completarse


def test_stream_consumer_stops_on_infinite_source():
    def infinite_source():
        i = 0
        while True:
            yield make_raw_row(product_id=f"P{i:04d}")
            i += 1

    rows = list(stream_fresh_retail_50k(limit=4, client=lambda r, s: infinite_source()))
    assert len(rows) == 4


def test_stream_yields_all_rows_when_source_is_shorter_than_limit():
    loader, _ = _loader([make_raw_row() for _ in range(2)])
    rows = list(stream_fresh_retail_50k(limit=5, client=loader))
    assert len(rows) == 2


def test_default_stream_call_fails_closed_without_network():
    with pytest.raises(FreshRetailDomainError):
        stream_fresh_retail_50k(limit=5)


def test_stream_propagates_row_validation_errors():
    bad = make_raw_row(hours_sale=[1.0] * 23)
    with pytest.raises(HourlyVectorLengthError):
        list(stream_fresh_retail_50k(limit=5, client=lambda r, s: iter([bad])))


def test_stream_propagates_client_errors_transparently():
    def failing_loader(revision: str, split: str):
        raise ConnectionError("network unreachable")

    with pytest.raises(ConnectionError):
        list(stream_fresh_retail_50k(limit=5, client=failing_loader))


def test_stream_propagates_mid_stream_client_errors():
    def loader(revision: str, split: str):
        def source():
            yield make_raw_row()
            raise ConnectionResetError("stream dropped mid-iteration")

        return source()

    stream = stream_fresh_retail_50k(limit=5, client=loader)
    assert next(stream).product_id == "P0001"
    with pytest.raises(ConnectionResetError):
        next(stream)


def test_streamed_rows_follow_canonical_contract():
    status = [0] * 24
    status[21] = 1
    status[22] = 1
    sales = [float(hour) for hour in range(24)]
    raw = make_raw_row(
        product_id=38,
        dt="2024-03-29",
        hours_sale=sales,
        hours_stock_status=status,
        stock_hour6_22_cnt=1,
    )
    (row,) = list(stream_fresh_retail_50k(limit=1, client=lambda r, s: iter([raw])))
    assert row.product_id == 38
    records = expand_to_hourly(row)
    assert len(records) == 24
    for hour, record in enumerate(records):
        assert record.sales_qty_observed == pytest.approx(sales[hour])
        assert record.stockout_flag == (status[hour] == 1)
        assert record.latent_demand_estimate is None
        assert record.hours_sale_raw == tuple(sales)
        assert record.hours_stock_status_raw == tuple(status)
    assert records[20].sales_observation_state == "uncensored"
    assert records[21].sales_observation_state == "censored_or_partial"
    # el índice 22 es una hora de quiebre pero queda FUERA de la ventana semiabierta [6:22)
    assert records[22].sales_observation_state == "censored_or_partial"
    assert row.stockout_hours_6_22 == 1  # solo se cuenta el índice 21


def test_live_hf_loader_uses_exact_dataset_call(monkeypatch):
    captured: dict = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return iter([make_raw_row()])

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    rows = live_hf_stream_loader(FRESH_RETAIL_PINNED_REVISION, "train")
    assert captured["args"] == (FRESH_RETAIL_DATASET_ID,)
    assert captured["kwargs"] == {
        "revision": FRESH_RETAIL_PINNED_REVISION,
        "split": "train",
        "streaming": True,
    }
    assert next(iter(rows))["product_id"] == "P0001"


def test_live_hf_loader_fails_closed_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(FreshRetailDomainError):
        live_hf_stream_loader(FRESH_RETAIL_PINNED_REVISION, "train")


def test_live_hf_loader_rejects_non_pinned_revision_before_calling_datasets(monkeypatch):
    called: list[tuple] = []

    def fake_load_dataset(*args, **kwargs):
        called.append((args, kwargs))
        return iter([])

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    for bad in ("deadbeef" * 5, "", "   ", None, 123):
        with pytest.raises((NonPinnedRevisionError, UnpinnedRevisionError)):
            live_hf_stream_loader(bad, "train")
    assert called == []  # el datasets.load_dataset falso nunca debe invocarse


def test_stream_composes_with_live_loader_offline(monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: iter(
        [make_raw_row() for _ in range(2)]
    )
    monkeypatch.setitem(sys.modules, "datasets", fake)

    rows = list(stream_fresh_retail_50k(limit=2, client=live_hf_stream_loader))
    assert len(rows) == 2
    assert all(row.revision == FRESH_RETAIL_PINNED_REVISION for row in rows)
