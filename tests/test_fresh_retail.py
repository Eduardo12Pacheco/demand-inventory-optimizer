"""Pruebas de ingesta FreshRetailNet-50K a nivel de fila.

Cubre la transformación canónica, la preservación de crudos y cada invariante
requerido a nivel de fila. La validación global del conteo de productos (865
product_id únicos) vive en test_fresh_retail_snapshot.py.
"""

from __future__ import annotations

import pytest

from inventory_optimizer.ingestion.fresh_retail import (
    FRESH_RETAIL_PINNED_REVISION,
    DailySumMismatchError,
    HourlyVectorLengthError,
    RowValidationError,
    StockCounterError,
    StockStatusValueError,
    UnpinnedRevisionError,
    acquire_fresh_retail_50k,
    expand_to_hourly,
    parse_daily_row,
)

from helpers import make_raw_row


def test_hourly_sale_vector_must_be_length_24():
    with pytest.raises(HourlyVectorLengthError):
        parse_daily_row(make_raw_row(hours_sale=[1.0] * 23))


def test_hourly_status_vector_must_be_length_24():
    with pytest.raises(HourlyVectorLengthError):
        parse_daily_row(make_raw_row(hours_stock_status=[0] * 25))


def test_stock_status_values_must_be_0_or_1():
    status = [0] * 24
    status[10] = 2
    with pytest.raises(StockStatusValueError):
        parse_daily_row(make_raw_row(hours_stock_status=status))


def test_stock_status_none_entry_is_rejected_as_domain_error():
    # Una entrada None debe aparecer como StockStatusValueError, nunca un TypeError crudo.
    status: list = [0] * 24
    status[10] = None
    with pytest.raises(StockStatusValueError):
        parse_daily_row(make_raw_row(hours_stock_status=status))


def test_inconsistent_stock_counter_rejected():
    # Un vector de estado todo ceros implica sum(hours_stock_status[6:22]) == 0.
    with pytest.raises(StockCounterError):
        parse_daily_row(make_raw_row(stock_hour6_22_cnt=5))


def test_index_22_excluded_from_stock_counter():
    # Regresión: desambigua el semiabierto [6:22) del inclusivo [6:23].
    # El índice 21 es 0 mientras el índice 22 es 1, así que solo la ventana
    # semiabierta puede satisfacer stock_hour6_22_cnt == sum(hours_stock_status[6:22]).
    status = [0] * 24
    status[22] = 1
    row = parse_daily_row(
        make_raw_row(hours_stock_status=status, stock_hour6_22_cnt=0)
    )
    assert row.stockout_hours_6_22 == 0
    records = expand_to_hourly(row)
    assert records[21].stockout_flag is False
    assert records[22].stockout_flag is True
    assert records[22].sales_qty_observed == pytest.approx(1.0)
    assert all(record.stock_hour6_22_cnt_raw == 0 for record in records)


def test_real_shaped_row_parses_and_preserves_sales():
    # Forma oficial de la fuente: product_id int64, clave dt, unos de estado en
    # los índices 21/22/23 y stock_hour6_22_cnt=1 (solo el índice 21 cae dentro
    # de la ventana semiabierta [6:22); los índices 22 y 23 NO se cuentan).
    status = [0] * 24
    status[21] = 1
    status[22] = 1
    status[23] = 1
    sales = [float(hour) for hour in range(24)]
    row = parse_daily_row(
        make_raw_row(
            product_id=38,
            dt="2024-03-29",
            hours_sale=sales,
            hours_stock_status=status,
            stock_hour6_22_cnt=1,
        )
    )
    assert row.product_id == 38
    assert row.dt.isoformat() == "2024-03-29"
    assert row.stockout_hours_6_22 == 1
    records = expand_to_hourly(row)
    for hour in (21, 22, 23):
        assert records[hour].stockout_flag is True
        assert records[hour].sales_qty_observed == pytest.approx(float(hour))
    assert records[20].stockout_flag is False
    assert records[22].stock_hour6_22_cnt_raw == 1
    assert records[22].stockout_hours_6_22 == 1


def test_inconsistent_daily_sum_rejected():
    # La fila por defecto suma 24.0; un sale_amount de 25.0 excede la tolerancia numérica.
    with pytest.raises(DailySumMismatchError):
        parse_daily_row(make_raw_row(sale_amount=25.0))


def test_unpinned_revision_rejected_at_acquisition():
    with pytest.raises(UnpinnedRevisionError):
        acquire_fresh_retail_50k(revision=None, client=lambda revision: [])
    with pytest.raises(UnpinnedRevisionError):
        acquire_fresh_retail_50k(revision="   ", client=lambda revision: [])


def test_unpinned_revision_rejected_at_parse():
    with pytest.raises(UnpinnedRevisionError):
        parse_daily_row(make_raw_row(), revision="")


def test_status_and_sales_transformation_is_canonical():
    status = [0] * 24
    status[3] = 1
    status[19] = 1
    sales = [float(hour) for hour in range(24)]
    records = expand_to_hourly(
        parse_daily_row(make_raw_row(hours_sale=sales, hours_stock_status=status))
    )
    assert len(records) == 24
    for hour, record in enumerate(records):
        assert record.sales_qty_observed == pytest.approx(sales[hour])
        assert record.stockout_flag == (status[hour] == 1)
        assert record.sales_observation_state == (
            "censored_or_partial" if status[hour] == 1 else "uncensored"
        )


def test_observed_sales_not_masked_on_stockout_hours():
    # Una hora con quiebre conserva su valor de ventas observado: sin
    # sustitución con NaN/cero.
    status = [0] * 24
    status[5] = 1
    sales = [2.5 if hour == 5 else 1.0 for hour in range(24)]
    records = expand_to_hourly(
        parse_daily_row(make_raw_row(hours_sale=sales, hours_stock_status=status))
    )
    stockout = records[5]
    assert stockout.stockout_flag is True
    assert stockout.sales_qty_observed == pytest.approx(2.5)
    assert stockout.hours_sale_raw == tuple(sales)
    assert stockout.hours_stock_status_raw == tuple(status)


def test_latent_demand_estimate_initially_none():
    for record in expand_to_hourly(parse_daily_row(make_raw_row())):
        assert record.latent_demand_estimate is None


def test_raw_fields_preserved_on_every_hourly_record():
    status = [0] * 24
    status[8] = 1  # una hora de quiebre dentro de la ventana semiabierta [6:22)
    row = parse_daily_row(
        make_raw_row(product_id="P0007", dt="2024-03-02", hours_stock_status=status)
    )
    for record in expand_to_hourly(row):
        assert record.product_id == "P0007"
        assert record.date.isoformat() == "2024-03-02"
        assert record.hours_stock_status_raw == tuple(status)
        assert record.stock_hour6_22_cnt_raw == 1
        assert record.stockout_hours_6_22 == 1
        assert record.sales_observation_state in ("censored_or_partial", "uncensored")


def test_dt_key_is_official_and_date_is_legacy_alias():
    assert parse_daily_row(make_raw_row(dt="2024-05-01")).dt.isoformat() == "2024-05-01"
    legacy = make_raw_row(dt="2024-05-01")
    legacy.pop("dt")
    legacy["date"] = "2024-05-02"
    assert parse_daily_row(legacy).dt.isoformat() == "2024-05-02"


def test_row_requires_dt_or_date_key():
    raw = make_raw_row()
    raw.pop("dt")
    with pytest.raises(RowValidationError):
        parse_daily_row(raw)


def test_numeric_product_id_normalization():
    assert parse_daily_row(make_raw_row(product_id=38)).product_id == 38
    assert parse_daily_row(make_raw_row(product_id="38")).product_id == 38
    assert parse_daily_row(make_raw_row(product_id="P0007")).product_id == "P0007"


def test_stockout_hours_6_22_is_derived_not_an_input():
    # No existe un campo crudo stockout_hours_6_22 en la fuente; el valor
    # canónico proviene de la ventana de estado validada.
    row = parse_daily_row(make_raw_row(stock_hour6_22_cnt=0))
    assert row.stockout_hours_6_22 == 0
    assert expand_to_hourly(row)[0].stockout_hours_6_22 == 0


def test_revision_is_pinned_to_expected_sha():
    assert FRESH_RETAIL_PINNED_REVISION == "08c1fab7f9257bc73679d415d65d644165d351d4"


def test_acquire_defaults_to_pinned_revision_with_client():
    seen: list[str] = []

    def client(revision: str):
        seen.append(revision)
        return [make_raw_row(product_id="P0042")]

    rows = acquire_fresh_retail_50k(client=client)
    assert seen == [FRESH_RETAIL_PINNED_REVISION]
    assert [row.product_id for row in rows] == ["P0042"]


def test_daily_sum_tolerance_accepts_binary_rounding():
    row = parse_daily_row(make_raw_row(hours_sale=[0.1] * 24, sale_amount=2.4))
    assert row.sale_amount == pytest.approx(2.4)
