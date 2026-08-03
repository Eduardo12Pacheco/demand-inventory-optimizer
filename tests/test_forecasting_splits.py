"""Pruebas de divisiones temporales: límites inclusivos explícitos, orden estricto,
particiones walk-forward expansivas, reportes de fechas faltantes e historia
insuficiente, rechazo de duplicados y preservación de ids de producto."""

from __future__ import annotations

from datetime import date

import pytest

from inventory_optimizer.forecasting.splits import (
    DuplicateKeyError,
    ExpandingWindowFold,
    InsufficientHistory,
    InvalidBoundaryOrderError,
    MissingDatesForProduct,
    TemporalSplit,
    expanding_window_folds,
    split_temporal,
)

from helpers import make_daily_row, make_series


def test_split_uses_explicit_inclusive_boundaries():
    rows = make_series(product_id="A", start="2024-01-01", days=15)
    split = split_temporal(
        rows,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 7),
        validation_start=date(2024, 1, 8),
        validation_end=date(2024, 1, 10),
        test_start=date(2024, 1, 11),
        test_end=date(2024, 1, 15),
    )
    assert isinstance(split, TemporalSplit)
    assert split.train == tuple(rows[:7])
    assert split.validation == tuple(rows[7:10])
    assert split.test == tuple(rows[10:])
    assert split.train_start == date(2024, 1, 1)
    assert split.train_end == date(2024, 1, 7)
    assert split.validation_start == date(2024, 1, 8)
    assert split.validation_end == date(2024, 1, 10)
    assert split.test_start == date(2024, 1, 11)
    assert split.test_end == date(2024, 1, 15)
    assert split.excluded_out_of_range == 0


def test_split_has_no_temporal_leakage():
    rows = make_series(product_id="A", start="2024-01-01", days=30)
    split = split_temporal(
        rows,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 10),
        validation_start=date(2024, 1, 11),
        validation_end=date(2024, 1, 20),
        test_start=date(2024, 1, 21),
        test_end=date(2024, 1, 30),
    )
    assert all(row.dt <= date(2024, 1, 10) for row in split.train)
    assert all(date(2024, 1, 11) <= row.dt <= date(2024, 1, 20) for row in split.validation)
    assert all(date(2024, 1, 21) <= row.dt <= date(2024, 1, 30) for row in split.test)
    assert (
        max(row.dt for row in split.train)
        < min(row.dt for row in split.validation)
        < min(row.dt for row in split.test)
    )
    assert set(split.train).isdisjoint(set(split.validation))
    assert set(split.validation).isdisjoint(set(split.test))


def test_split_rejects_duplicate_product_date_keys():
    rows = make_series(product_id="A", start="2024-01-01", days=3)
    rows.append(make_daily_row(product_id="A", dt="2024-01-02", sale_amount=99.0))
    with pytest.raises(DuplicateKeyError, match="Duplicate"):
        split_temporal(rows, train_start=date(2024, 1, 1), train_end=date(2024, 1, 3))


def test_split_rejects_invalid_boundary_order():
    rows = make_series(product_id="A", start="2024-01-01", days=10)
    with pytest.raises(InvalidBoundaryOrderError):
        split_temporal(
            rows,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 7),
            validation_start=date(2024, 1, 7),  # overlaps inclusive train end
            validation_end=date(2024, 1, 10),
        )
    with pytest.raises(InvalidBoundaryOrderError):
        split_temporal(rows, train_start=date(2024, 1, 5), train_end=date(2024, 1, 1))
    with pytest.raises(InvalidBoundaryOrderError):
        split_temporal(
            rows,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 7),
            test_start=date(2024, 1, 7),  # must follow train end strictly
            test_end=date(2024, 1, 10),
        )
    with pytest.raises(InvalidBoundaryOrderError):
        split_temporal(
            rows,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 7),
            validation_start=date(2024, 1, 8),
            validation_end=date(2024, 1, 5),
        )


def test_split_reports_missing_dates_without_filling():
    rows = make_series(product_id="A", start="2024-01-01", days=5)
    rows = [row for index, row in enumerate(rows) if index != 2]  # gap at 2024-01-03
    split = split_temporal(rows, train_start=date(2024, 1, 1), train_end=date(2024, 1, 5))
    assert split.missing_train_dates == (date(2024, 1, 3),)
    # Ninguna fila sintética puede aparecer: la fecha faltante debe permanecer ausente en todas partes.
    assert all(row.dt != date(2024, 1, 3) for row in split.train)
    assert split.validation == ()
    assert split.validation_start is None
    assert split.validation_end is None
    assert split.test == ()
    assert split.test_start is None
    assert split.test_end is None
    assert split.missing_validation_dates == ()
    assert split.missing_test_dates == ()


def test_split_reports_per_product_missing_dates():
    rows_a = make_series(product_id="A", start="2024-01-01", days=5)
    rows_b = [row for index, row in enumerate(make_series(product_id="B", start="2024-01-01", days=5)) if index != 4]
    split = split_temporal(
        rows_a + rows_b, train_start=date(2024, 1, 1), train_end=date(2024, 1, 5)
    )
    by_product = {entry.product_id: entry for entry in split.missing_per_product}
    assert isinstance(by_product["A"], MissingDatesForProduct)
    assert by_product["A"].dates == ()
    assert by_product["B"].dates == (date(2024, 1, 5),)


def test_split_reports_insufficient_history_days():
    rows_a = make_series(product_id="A", start="2024-01-01", days=5)
    rows_b = make_series(product_id="B", start="2024-01-01", days=2)
    split = split_temporal(
        rows_a + rows_b,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 5),
        min_history_days=3,
        min_history_span_days=3,
    )
    assert split.insufficient_history == (
        InsufficientHistory(
            product_id="B",
            observed_days=2,
            observed_span_days=2,
            required_days=3,
            required_span_days=3,
        ),
    )


def test_split_reports_insufficient_history_by_span():
    rows = [
        make_daily_row(product_id="A", dt="2024-01-01", sale_amount=1.0),
        make_daily_row(product_id="A", dt="2024-01-10", sale_amount=5.0),
    ]
    split = split_temporal(
        rows,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 10),
        min_history_days=2,
        min_history_span_days=14,
    )
    assert split.insufficient_history == (
        InsufficientHistory(
            product_id="A",
            observed_days=2,
            observed_span_days=10,
            required_days=2,
            required_span_days=14,
        ),
    )


def test_split_excludes_and_counts_out_of_range_rows():
    rows = make_series(product_id="A", start="2023-12-30", days=5)
    split = split_temporal(rows, train_start=date(2024, 1, 1), train_end=date(2024, 1, 3))
    assert [row.dt for row in split.train] == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert split.excluded_out_of_range == 2


def test_split_product_present_only_in_test_gets_zero_history_report():
    rows = make_series(product_id="A", start="2024-01-01", days=5) + make_series(
        product_id="C", start="2024-01-11", days=2
    )
    split = split_temporal(
        rows,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 5),
        validation_start=date(2024, 1, 6),
        validation_end=date(2024, 1, 10),
        test_start=date(2024, 1, 11),
        test_end=date(2024, 1, 12),
        min_history_days=1,
        min_history_span_days=1,
    )
    # C tiene cero días de entrenamiento: se reporta explícitamente, nunca se descarta en silencio.
    by_product = {entry.product_id: entry for entry in split.insufficient_history}
    assert by_product["C"].observed_days == 0
    assert by_product["C"].observed_span_days == 0
    # El reporte de fechas faltantes por producto de C cubre TODO el lapso solicitado.
    missing_c = next(
        entry for entry in split.missing_per_product if entry.product_id == "C"
    )
    assert missing_c.dates == (
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 6),
        date(2024, 1, 7),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
    )
    assert [row.product_id for row in split.test] == ["C", "C"]


def test_split_preserves_product_ids_and_sorts_deterministically():
    rows = [
        make_daily_row(product_id="P0002", dt="2024-01-02", sale_amount=2.0),
        make_daily_row(product_id=38, dt="2024-01-02", sale_amount=3.0),
        make_daily_row(product_id="P0001", dt="2024-01-02", sale_amount=1.0),
        make_daily_row(product_id="P0001", dt="2024-01-01", sale_amount=0.5),
        make_daily_row(product_id=38, dt="2024-01-01", sale_amount=2.5),
    ]
    split = split_temporal(rows, train_start=date(2024, 1, 1), train_end=date(2024, 1, 2))
    ordered = [(row.product_id, row.dt) for row in split.train]
    assert ordered == [
        (38, date(2024, 1, 1)),
        (38, date(2024, 1, 2)),
        ("P0001", date(2024, 1, 1)),
        ("P0001", date(2024, 1, 2)),
        ("P0002", date(2024, 1, 2)),
    ]
    assert {row.product_id for row in split.train} == {38, "P0001", "P0002"}


# --- particiones walk-forward de ventana expansiva -------------------------------


def test_expanding_window_folds_keep_train_start_fixed_and_expand():
    rows = make_series(product_id="A", start="2024-01-01", days=10)
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=2,
        folds=3,
        step_days=1,
    )
    assert isinstance(folds, tuple)
    assert all(isinstance(fold, ExpandingWindowFold) for fold in folds)
    assert len(folds) == 3
    fold1, fold2, fold3 = folds
    assert (fold1.fold_index, fold2.fold_index, fold3.fold_index) == (1, 2, 3)
    assert fold1.split.train_start == date(2024, 1, 1)
    assert fold1.split.train_start == fold2.split.train_start == fold3.split.train_start
    assert fold1.split.train_end == date(2024, 1, 3)
    assert fold2.split.train_end == date(2024, 1, 4)
    assert fold3.split.train_end == date(2024, 1, 5)
    assert fold1.split.validation_start == date(2024, 1, 4)
    assert fold1.split.validation_end == date(2024, 1, 5)
    assert fold2.split.validation_start == date(2024, 1, 5)
    assert fold2.split.validation_end == date(2024, 1, 6)
    assert fold3.split.validation_start == date(2024, 1, 6)
    assert fold3.split.validation_end == date(2024, 1, 7)
    assert len(fold1.split.train) == 3
    assert len(fold2.split.train) == 4
    assert len(fold3.split.train) == 5


def test_expanding_window_folds_never_leak_validation_rows_into_train():
    rows = make_series(product_id="A", start="2024-01-01", days=10)
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=2,
        folds=3,
        step_days=1,
    )
    for fold in folds:
        train_dates = {row.dt for row in fold.split.train}
        validation_dates = {row.dt for row in fold.split.validation}
        assert train_dates.isdisjoint(validation_dates)
        assert all(row.dt <= fold.split.train_end for row in fold.split.train)
        assert all(
            fold.split.validation_start <= row.dt <= fold.split.validation_end
            for row in fold.split.validation
        )


def test_expanding_window_folds_reuse_training_rows_across_folds():
    rows = make_series(product_id="A", start="2024-01-01", days=10)
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=2,
        folds=3,
        step_days=1,
    )
    assert set(folds[0].split.train) < set(folds[2].split.train)
    # Las ventanas de validación pueden solaparse cuando step_days < validation_days; la
    # reutilización es esperada y documentada, solo el entrenamiento debe crecer de forma monótona.
    assert {row.dt for row in folds[0].split.validation} & {
        row.dt for row in folds[1].split.validation
    } == {date(2024, 1, 5)}


def test_expanding_window_folds_optional_test_horizon():
    rows = make_series(product_id="A", start="2024-01-01", days=12)
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=2,
        folds=2,
        step_days=1,
        test_start=date(2024, 1, 9),
        test_days=3,
    )
    expected_test = [date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
    # Partición 1: 12 filas - 3 train - 2 validation - 3 test = 4 excluidas
    # (días de hueco 01-06..01-08 más el 2024-01-12). La partición 2 entrena con 4 días,
    # así que solo quedan los días de hueco 01-07/01-08 más el 2024-01-12: 3 excluidas.
    expected_excluded = {1: 4, 2: 3}
    for fold in folds:
        assert [row.dt for row in fold.split.test] == expected_test
        assert all(row.dt > fold.split.validation_end for row in fold.split.test)
        assert fold.split.excluded_out_of_range == expected_excluded[fold.fold_index]


def test_expanding_window_folds_report_insufficient_history_per_fold():
    rows = make_series(product_id="A", start="2024-01-01", days=6) + make_series(
        product_id="B", start="2024-01-04", days=3
    )
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=1,
        folds=2,
        step_days=1,
    )
    fold1, fold2 = folds
    # El entrenamiento de la partición 1 es [01-01..01-03]: B aún no tiene filas ahí,
    # así que B no tiene historia de entrenamiento aunque aparezca en la ventana de
    # validación de la partición 1.
    assert [entry.product_id for entry in fold1.split.insufficient_history] == ["B"]
    assert fold1.split.insufficient_history[0].observed_days == 0
    # El entrenamiento de la partición 2 se expande a [01-01..01-04]: B ahora tiene un día de entrenamiento.
    assert fold2.split.insufficient_history == ()


def test_expanding_window_folds_report_missing_dates_per_fold():
    rows = [row for index, row in enumerate(make_series(product_id="A", start="2024-01-01", days=6)) if index != 2]
    folds = expanding_window_folds(
        rows,
        start=date(2024, 1, 1),
        initial_train_days=3,
        validation_days=1,
        folds=2,
        step_days=1,
    )
    for fold in folds:
        assert date(2024, 1, 3) in fold.split.missing_train_dates
        assert all(row.dt != date(2024, 1, 3) for row in fold.split.train)
        assert all(row.dt != date(2024, 1, 3) for row in fold.split.validation)


def test_expanding_window_folds_reject_invalid_parameters():
    rows = make_series(product_id="A", start="2024-01-01", days=10)
    with pytest.raises(ValueError):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=0, validation_days=2, folds=2, step_days=1
        )
    with pytest.raises(ValueError):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=3, validation_days=0, folds=2, step_days=1
        )
    with pytest.raises(ValueError):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=3, validation_days=2, folds=0, step_days=1
        )
    with pytest.raises(ValueError):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=3, validation_days=2, folds=2, step_days=0
        )
    with pytest.raises(ValueError, match="together"):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=3, validation_days=2, folds=2, step_days=1, test_days=3
        )
    with pytest.raises(ValueError, match="together"):
        expanding_window_folds(
            rows, start=date(2024, 1, 1), initial_train_days=3, validation_days=2, folds=2, step_days=1, test_start=date(2024, 1, 9)
        )
    with pytest.raises(InvalidBoundaryOrderError):
        expanding_window_folds(
            rows,
            start=date(2024, 1, 1),
            initial_train_days=3,
            validation_days=2,
            folds=2,
            step_days=1,
            test_start=date(2024, 1, 5),  # se solapa con la última ventana de validación
            test_days=2,
        )
