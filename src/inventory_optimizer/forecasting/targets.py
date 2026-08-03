"""Proyección de objetivos de ventas observadas desde filas diarias validadas.

El objetivo es ``observed_sales = sale_amount``: una observación de la fuente
normalizada globalmente que puede estar censurada durante quiebres de stock.
NUNCA se etiqueta como demanda real, y ``latent_demand_estimate`` permanece
ausente/``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

from inventory_optimizer.forecasting._ordering import (
    DuplicateKeyError,
    validate_unique_and_sort,
)
from inventory_optimizer.ingestion.fresh_retail import (
    DailySourceRow,
    SALES_OBSERVATION_CENSORED,
    SALES_OBSERVATION_UNCENSORED,
)

ObservationState = Literal["censored_or_partial", "uncensored"]


@dataclass(frozen=True)
class TargetRow:
    """Un objetivo de ventas observadas para un producto/fecha.

    ``observation_state`` se deriva del vector de estado de la fuente: la fila
    es ``censored_or_partial`` cuando CUALQUIER hora tiene status == 1, si no
    ``uncensored``. ``latent_demand_estimate`` es siempre ``None``.
    """

    product_id: str | int
    dt: date
    observed_sales: float
    stockout_hours_6_22: int
    observation_state: ObservationState
    revision: str
    latent_demand_estimate: None = None


def project_targets(rows: Iterable[DailySourceRow]) -> tuple[TargetRow, ...]:
    """Proyecta filas diarias validadas en objetivos de ventas observadas.

    La salida se ordena de forma determinista por ``(product_id, dt)``; las
    claves duplicadas lanzan :class:`DuplicateKeyError`. ``sale_amount`` se
    traslada tal cual como ``observed_sales`` — nunca se recorta, imputa,
    enmascara ni se re-etiqueta como demanda real.
    """
    ordered = validate_unique_and_sort(rows)
    return tuple(
        TargetRow(
            product_id=row.product_id,
            dt=row.dt,
            observed_sales=row.sale_amount,
            stockout_hours_6_22=row.stockout_hours_6_22,
            observation_state=(
                SALES_OBSERVATION_CENSORED
                if any(hour == 1 for hour in row.hours_stock_status)
                else SALES_OBSERVATION_UNCENSORED
            ),
            revision=row.revision,
        )
        for row in ordered
    )
