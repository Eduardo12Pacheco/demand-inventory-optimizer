"""Adaptador de ingesta FreshRetailNet-50K.

Esquema horario canónico fijado a la revisión de fuente
``08c1fab7f9257bc73679d415d65d644165d351d4``.

Contrato esperado de la fila diaria de la fuente (una fila por producto/fecha)::

    product_id: int or str            # int64 en la fuente oficial; las formas
                                      # numéricas se normalizan a int
    dt: cadena de fecha ISO (YYYY-MM-DD)  # clave oficial de la fuente; 'date'
                                          # se acepta como alias heredado/fixture
    sale_amount: número               # total diario en unidades originales
    hours_sale: 24 números            # ventas horarias observadas, unidades originales
    hours_stock_status: 24 enteros    # 0 = en stock, 1 = hora con quiebre
    stock_hour6_22_cnt: int           # contador de la fuente; debe ser igual a
                                      # sum(hours_stock_status[6:22])

La ventana de stock es semiabierta [6, 22): posiciones 6..21 (16 horas); el
índice 22 NO se cuenta. stock_hour6_22_cnt cuenta las horas con status==1 en esa
ventana. No existe un campo de entrada crudo stockout_hours_6_22; el
stockout_hours_6_22 canónico se deriva de la ventana de estado validada.

Mapeo canónico (una fila de la fuente -> 24 registros horarios)::

    sales_qty_observed      = hours_sale[h]                     (unidades originales)
    stockout_flag           = hours_stock_status[h] == 1
    latent_demand_estimate  = None                              (nunca se estima)
    sales_observation_state = "censored_or_partial" cuando status == 1,
                              "uncensored" en caso contrario
    hours_sale_raw          = vector crudo completo de 24 ventas (preservado)
    hours_stock_status_raw  = vector crudo completo de 24 estados (preservado)
    stock_hour6_22_cnt_raw  = contador crudo de la fuente (preservado)
    stockout_hours_6_22     = sum(hours_stock_status[6:22])     (derivado)

Las ventas observadas NUNCA se enmascaran: una hora con quiebre conserva su valor
observado y se marca con ``sales_observation_state="censored_or_partial"``. Las
ventas observadas no se etiquetan como "demanda real"; la demanda latente queda
fuera de alcance en el momento de la ingesta. La validación a nivel de fila es
independiente del invariante global de instantánea (865 product_id únicos), que
solo se aplica a instantáneas completas mediante un contexto explícito.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

FRESH_RETAIL_PINNED_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
"""Revisión de fuente de FreshRetailNet-50K fijada (nunca vacía/sin fijar)."""

HOURS_PER_DAY = 24
WINDOW_START_HOUR = 6
WINDOW_EXCLUSIVE_END_HOUR = 22  # semiabierta [6, 22): índices 6..21, 16 horas
WINDOW_HOURS = WINDOW_EXCLUSIVE_END_HOUR - WINDOW_START_HOUR  # 16

SALES_OBSERVATION_CENSORED = "censored_or_partial"
SALES_OBSERVATION_UNCENSORED = "uncensored"

SALE_AMOUNT_REL_TOL = 1e-9
SALE_AMOUNT_ABS_TOL = 1e-6

SNAPSHOT_EXPECTED_UNIQUE_PRODUCTS = 865


# --- errores de dominio -----------------------------------------------------


class FreshRetailDomainError(Exception):
    """Clase base para los errores de ingesta de FreshRetailNet-50K."""


class UnpinnedRevisionError(FreshRetailDomainError):
    """La revisión de la fuente está vacía/sin definir; la adquisición requiere una revisión fijada."""


class RowValidationError(FreshRetailDomainError):
    """Una fila diaria de la fuente viola el esquema de FreshRetailNet-50K."""


class HourlyVectorLengthError(RowValidationError):
    """hours_sale / hours_stock_status deben contener exactamente 24 valores."""


class StockStatusValueError(RowValidationError):
    """Los valores de hours_stock_status deben ser 0 (en stock) o 1 (quiebre)."""


class StockCounterError(RowValidationError):
    """stock_hour6_22_cnt no coincide con sum(hours_stock_status[6:22])."""


class DailySumMismatchError(RowValidationError):
    """sale_amount no coincide con sum(hours_sale) más allá de la tolerancia numérica."""


class GlobalSnapshotError(FreshRetailDomainError):
    """Se viola un invariante global de la instantánea (p. ej. cobertura de productos)."""


# --- registros --------------------------------------------------------------


@dataclass(frozen=True)
class DailySourceRow:
    """Una fila diaria validada de la fuente con sus valores crudos preservados."""

    product_id: str | int
    dt: date
    sale_amount: float
    hours_sale: tuple[float, ...]
    hours_stock_status: tuple[int, ...]
    stock_hour6_22_cnt: int
    stockout_hours_6_22: int
    revision: str


@dataclass(frozen=True)
class HourlyRecord:
    """Un registro horario canónico de observación de demanda.

    ``latent_demand_estimate`` es siempre ``None`` en el momento de la ingesta;
    la recuperación de demanda latente es una etapa posterior separada y nunca
    se realiza aquí.
    """

    product_id: str | int
    date: date
    hour: int
    sales_qty_observed: float
    stockout_flag: bool
    hours_sale_raw: tuple[float, ...]
    hours_stock_status_raw: tuple[int, ...]
    stock_hour6_22_cnt_raw: int
    stockout_hours_6_22: int
    sales_observation_state: Literal["censored_or_partial", "uncensored"]
    latent_demand_estimate: None = None


@dataclass(frozen=True)
class GlobalSnapshotSummary:
    """Resultado de una validación global de instantánea."""

    total_rows: int
    unique_products: int


# --- fijación de revisión ---------------------------------------------------


def validate_revision(revision: str | None) -> str:
    """Devuelve una revisión recortada no vacía o lanza :class:`UnpinnedRevisionError`."""
    if revision is None or not revision.strip():
        raise UnpinnedRevisionError(
            "FreshRetailNet-50K acquisition requires a pinned source revision; "
            f"got {revision!r}. Use FRESH_RETAIL_PINNED_REVISION or an explicit SHA."
        )
    return revision.strip()


# --- ayudantes de parseo ----------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_status_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_product_id(value: Any) -> str | int:
    """Normaliza los identificadores de producto para que las formas numéricas se comparen de forma consistente.

    Los valores int64 siguen siendo ints; las cadenas con aspecto numérico
    ("38") colapsan al mismo int; el resto de cadenas no vacías se conservan
    tal cual.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RowValidationError(f"product_id must be an int or str, got {value!r}.")
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text:
        raise RowValidationError("product_id must be a non-empty string.")
    try:
        return int(text)
    except ValueError:
        return text


_MISSING = object()
"""Centinela para "ningún valor inválido encontrado" (None puede ser un valor inválido)."""


def _check_stock_counters(status_vector: tuple[int, ...], stock_cnt: int) -> None:
    expected = sum(status_vector[WINDOW_START_HOUR:WINDOW_EXCLUSIVE_END_HOUR])
    if stock_cnt != expected:
        raise StockCounterError(
            "Stock counter inconsistent with hours_stock_status in the half-open "
            f"window [{WINDOW_START_HOUR}, {WINDOW_EXCLUSIVE_END_HOUR}) (indices "
            f"{WINDOW_START_HOUR}..{WINDOW_EXCLUSIVE_END_HOUR - 1}, {WINDOW_HOURS} "
            f"hours): got stock_hour6_22_cnt={stock_cnt}, expected "
            f"sum(hours_stock_status[6:22])={expected}."
        )


def _check_daily_sum(sale_vector: tuple[float, ...], sale_amount: float) -> None:
    total = math.fsum(sale_vector)
    if not math.isclose(
        total, sale_amount, rel_tol=SALE_AMOUNT_REL_TOL, abs_tol=SALE_AMOUNT_ABS_TOL
    ):
        raise DailySumMismatchError(
            f"sale_amount={sale_amount} differs from sum(hours_sale)={total} beyond "
            f"tolerance (rel={SALE_AMOUNT_REL_TOL}, abs={SALE_AMOUNT_ABS_TOL})."
        )


# --- validación y parseo a nivel de fila ------------------------------------


def parse_daily_row(
    raw: Mapping[str, Any],
    *,
    revision: str = FRESH_RETAIL_PINNED_REVISION,
) -> DailySourceRow:
    """Valida una fila cruda diaria de la fuente y devuelve una fila tipada y validada."""
    pinned = validate_revision(revision)

    def _field(name: str) -> Any:
        if name not in raw:
            raise RowValidationError(
                f"Missing required field {name!r} in FreshRetailNet-50K source row."
            )
        return raw[name]

    product_id = _normalize_product_id(_field("product_id"))

    if "dt" in raw:
        raw_date = raw["dt"]
    elif "date" in raw:
        raw_date = raw["date"]  # alias heredado/de fixture
    else:
        raise RowValidationError(
            "Missing required date key 'dt' (or legacy alias 'date')."
        )
    try:
        row_date = date.fromisoformat(str(raw_date))
    except (TypeError, ValueError) as exc:
        raise RowValidationError(
            f"dt must be an ISO YYYY-MM-DD string, got {raw_date!r}."
        ) from exc

    sale_amount = _field("sale_amount")
    if not _is_number(sale_amount):
        raise RowValidationError(f"sale_amount must be numeric, got {sale_amount!r}.")
    sale_amount = float(sale_amount)

    hours_sale = _field("hours_sale")
    if not isinstance(hours_sale, Sequence) or len(hours_sale) != HOURS_PER_DAY:
        raise HourlyVectorLengthError(
            f"hours_sale must have exactly {HOURS_PER_DAY} entries, "
            f"got {len(hours_sale)}."
        )
    if any(not _is_number(value) for value in hours_sale):
        raise RowValidationError("hours_sale entries must be numeric.")
    sale_vector = tuple(float(value) for value in hours_sale)

    hours_stock_status = _field("hours_stock_status")
    if not isinstance(hours_stock_status, Sequence) or len(hours_stock_status) != HOURS_PER_DAY:
        raise HourlyVectorLengthError(
            f"hours_stock_status must have exactly {HOURS_PER_DAY} entries, "
            f"got {len(hours_stock_status)}."
        )
    bad_status = next(
        (value for value in hours_stock_status if not _is_status_value(value)),
        _MISSING,
    )
    if bad_status is not _MISSING:
        raise StockStatusValueError(
            "hours_stock_status entries must be 0 (in stock) or 1 (stockout), "
            f"got {bad_status!r}."
        )
    status_vector = tuple(int(value) for value in hours_stock_status)

    stock_cnt = _field("stock_hour6_22_cnt")
    if not _is_non_negative_int(stock_cnt):
        raise RowValidationError(
            f"stock_hour6_22_cnt must be a non-negative int, got {stock_cnt!r}."
        )
    stock_cnt = int(stock_cnt)
    _check_stock_counters(status_vector, stock_cnt)
    _check_daily_sum(sale_vector, sale_amount)
    stockout_hours_6_22 = sum(
        status_vector[WINDOW_START_HOUR:WINDOW_EXCLUSIVE_END_HOUR]
    )

    return DailySourceRow(
        product_id=product_id,
        dt=row_date,
        sale_amount=sale_amount,
        hours_sale=sale_vector,
        hours_stock_status=status_vector,
        stock_hour6_22_cnt=stock_cnt,
        stockout_hours_6_22=stockout_hours_6_22,
        revision=pinned,
    )


# --- transformación canónica ------------------------------------------------


def expand_to_hourly(row: DailySourceRow) -> tuple[HourlyRecord, ...]:
    """Expande una fila diaria validada en 24 registros horarios canónicos.

    Nunca enmascara las ventas observadas: las horas con quiebre conservan su
    valor crudo y se marcan con ``sales_observation_state="censored_or_partial"``.
    """
    return tuple(
        HourlyRecord(
            product_id=row.product_id,
            date=row.dt,
            hour=hour,
            sales_qty_observed=row.hours_sale[hour],
            stockout_flag=row.hours_stock_status[hour] == 1,
            hours_sale_raw=row.hours_sale,
            hours_stock_status_raw=row.hours_stock_status,
            stock_hour6_22_cnt_raw=row.stock_hour6_22_cnt,
            stockout_hours_6_22=row.stockout_hours_6_22,
            sales_observation_state=(
                SALES_OBSERVATION_CENSORED
                if row.hours_stock_status[hour] == 1
                else SALES_OBSERVATION_UNCENSORED
            ),
        )
        for hour in range(HOURS_PER_DAY)
    )


# --- adquisición ------------------------------------------------------------


RawRowLoader = Callable[[str], Iterable[Mapping[str, Any]]]


def _default_client(revision: str) -> Iterable[Mapping[str, Any]]:
    raise FreshRetailDomainError(
        "No FreshRetailNet-50K download client configured. Pass an explicit "
        "`client=` loader (e.g. a huggingface_hub snapshot loader pinned to "
        f"revision {revision}) or ingest saved fixtures via parse_daily_row. "
        "Offline tests never download or expand the dataset."
    )


def acquire_fresh_retail_50k(
    revision: str = FRESH_RETAIL_PINNED_REVISION,
    *,
    client: RawRowLoader | None = None,
) -> list[DailySourceRow]:
    """Adquiere filas de FreshRetailNet-50K en una revisión fijada explícita.

    La revisión por defecto es :data:`FRESH_RETAIL_PINNED_REVISION` y se
    rechaza cuando está vacía o sin definir. El cliente por defecto se niega a
    descargar; proporcione ``client`` para adquisición por red o use
    ``parse_daily_row`` sobre fixtures guardados.
    """
    pinned = validate_revision(revision)
    loader = client if client is not None else _default_client
    return [parse_daily_row(raw, revision=pinned) for raw in loader(pinned)]


# --- validación global de instantánea ---------------------------------------


class SnapshotContext(Enum):
    """Contexto explícito para la validación global de instantánea.

    COMPLETE: una instantánea de producción completa; se aplica el invariante
    de 865 productos únicos. FIXTURE: un fixture offline pequeño; el
    invariante se omite.
    """

    COMPLETE = "complete"
    FIXTURE = "fixture"


def validate_global_snapshot(
    rows: Iterable[DailySourceRow],
    *,
    context: SnapshotContext,
    expected_unique_products: int = SNAPSHOT_EXPECTED_UNIQUE_PRODUCTS,
) -> GlobalSnapshotSummary:
    """Valida los invariantes globales de la instantánea (las comprobaciones por fila ocurren por fila).

    Con ``context=SnapshotContext.COMPLETE`` la instantánea debe contener
    exactamente ``expected_unique_products`` (por defecto 865) valores
    ``product_id`` únicos. Los contextos de fixture están exentos del
    invariante de conteo y solo requieren filas no vacías. El contexto es
    obligatorio y se valida su tipo para que una aserción global nunca pueda
    omitirse accidentalmente.
    """
    if not isinstance(context, SnapshotContext):
        raise TypeError(f"context must be a SnapshotContext, got {context!r}.")
    materialized = list(rows)
    if not materialized:
        raise GlobalSnapshotError("Global snapshot validation received no rows.")
    unique_products = {row.product_id for row in materialized}
    summary = GlobalSnapshotSummary(
        total_rows=len(materialized),
        unique_products=len(unique_products),
    )
    if (
        context == SnapshotContext.COMPLETE
        and len(unique_products) != expected_unique_products
    ):
        raise GlobalSnapshotError(
            f"Complete FreshRetailNet-50K snapshot expected {expected_unique_products} "
            f"unique product_ids, found {len(unique_products)}."
        )
    return summary
