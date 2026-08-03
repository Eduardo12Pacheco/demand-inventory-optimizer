"""Adquisición de streaming acotada para FreshRetailNet-50K.

Transmite como máximo :data:`FRESH_RETAIL_STREAM_MAX_LIMIT` filas diarias
validadas desde ``datasets`` de Hugging Face en la revisión fijada exacta,
reutilizando el contrato canónico de fila de
:mod:`inventory_optimizer.ingestion.fresh_retail`
(:func:`fresh_retail.parse_daily_row`). Este módulo está deliberadamente
separado para que la dependencia opcional ``datasets`` siga siendo perezosa:
importar código offline nunca requiere acceso a la red ni al paquete.

Garantías
    * La única revisión que jamás se entrega a un cliente es
      :data:`FRESH_RETAIL_PINNED_REVISION`; las revisiones vacías, con espacios,
      no textuales o no fijadas se rechazan antes de cualquier llamada al cliente.
    * ``limit`` debe ser un entero positivo no mayor que 1000; la iteración se
      detiene exactamente en ``limit`` y nunca materializa la fuente completa.
    * Una llamada por defecto (sin ``client`` explícito) no realiza acceso a la
      red y falla cerrado con un error de dominio; el cliente en vivo nunca se
      invoca en silencio.
    * La validación de filas es idéntica a la ruta basada en listas, y los
      errores del cliente o de validación se propagan sin cambios (transparentes).
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Mapping

from .fresh_retail import (
    FRESH_RETAIL_PINNED_REVISION,
    DailySourceRow,
    FreshRetailDomainError,
    UnpinnedRevisionError,
    parse_daily_row,
)

FRESH_RETAIL_DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
"""Identificador exacto del conjunto de datos de Hugging Face (nunca se modifica)."""

FRESH_RETAIL_STREAM_MAX_LIMIT = 1000
"""Máximo duro de filas que puede producir un único stream acotado."""


# --- errores de dominio -------------------------------------------------------


class InvalidStreamLimitError(FreshRetailDomainError):
    """El límite de streaming debe ser un entero positivo no mayor que el tope duro."""


class NonPinnedRevisionError(FreshRetailDomainError):
    """La adquisición por streaming solo acepta la revisión fijada auditada."""


# --- validación ---------------------------------------------------------------


def validate_stream_limit(limit: Any) -> int:
    """Devuelve ``limit`` cuando es un int positivo como máximo el tope duro."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise InvalidStreamLimitError(
            f"Streaming limit must be a positive integer, got {limit!r}."
        )
    if limit > FRESH_RETAIL_STREAM_MAX_LIMIT:
        raise InvalidStreamLimitError(
            f"Streaming limit {limit} exceeds the hard maximum of "
            f"{FRESH_RETAIL_STREAM_MAX_LIMIT} rows per stream."
        )
    return limit


def validate_stream_revision(revision: str | None) -> str:
    """Devuelve la revisión cuando es exactamente el SHA fijado.

    Se rechazan los valores vacíos, con espacios, no textuales y cualquier otro
    valor no fijado; la ruta de streaming nunca entrega una revisión distinta a
    un cliente.
    """
    if not isinstance(revision, str) or not revision.strip():
        raise UnpinnedRevisionError(
            "Streaming FreshRetailNet-50K acquisition requires a pinned source "
            f"revision string; got {revision!r}. Use FRESH_RETAIL_PINNED_REVISION."
        )
    if revision != FRESH_RETAIL_PINNED_REVISION:
        raise NonPinnedRevisionError(
            f"Streaming acquisition only accepts the audited fixed revision "
            f"{FRESH_RETAIL_PINNED_REVISION!r}, got {revision!r}."
        )
    return revision


# --- clientes -----------------------------------------------------------------


StreamRowLoader = Callable[[str, str], Iterable[Mapping[str, Any]]]
"""Protocolo del cargador de streaming: ``loader(revision, split) -> iterable de filas crudas``.

El cargador recibe la revisión fijada ya validada y un nombre de split explícito,
y debe devolver un iterable de filas crudas de la fuente (mappings) sin
materializar el conjunto de datos completo.
"""


def live_hf_stream_loader(revision: str, split: str) -> Iterable[Mapping[str, Any]]:
    """Transmite filas crudas de FreshRetailNet-50K a través de ``datasets`` de Hugging Face.

    Valida que ``revision`` sea exactamente :data:`FRESH_RETAIL_PINNED_REVISION`
    antes que cualquier otra cosa, y luego llama a ``datasets.load_dataset`` con
    el ID exacto del conjunto de datos (:data:`FRESH_RETAIL_DATASET_ID`), la
    revisión fijada, el ``split`` explícito y ``streaming=True``. El paquete
    ``datasets`` se importa de forma perezosa; cuando no está instalado se lanza
    un error de dominio conservando el ``ImportError`` como causa. El uso directo
    nunca puede cargar una revisión mutable o no fijada.
    """
    pinned = validate_stream_revision(revision)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FreshRetailDomainError(
            "Streaming FreshRetailNet-50K requires the optional 'datasets' "
            "package. Install it (e.g. `uv sync --extra streaming`) or pass an "
            "injected client for offline tests."
        ) from exc
    return load_dataset(
        FRESH_RETAIL_DATASET_ID,
        revision=pinned,
        split=split,
        streaming=True,
    )


# --- streaming acotado --------------------------------------------------------


def stream_fresh_retail_50k(
    limit: int,
    *,
    split: str = "train",
    revision: str = FRESH_RETAIL_PINNED_REVISION,
    client: StreamRowLoader | None = None,
) -> Iterator[DailySourceRow]:
    """Transmite como máximo ``limit`` filas diarias validadas en la revisión fijada.

    ``limit`` debe ser un entero positivo no mayor que
    :data:`FRESH_RETAIL_STREAM_MAX_LIMIT` (1000). La revisión debe ser
    exactamente :data:`FRESH_RETAIL_PINNED_REVISION`. Una llamada por defecto
    sin ``client`` explícito no realiza acceso a la red y lanza un error de
    dominio; pase :func:`live_hf_stream_loader` para streaming real o un
    cargador inyectado en pruebas offline. La iteración se detiene exactamente
    en ``limit`` sin materializar la fuente completa, y los errores del cliente
    o de validación de filas se propagan sin cambios.
    """
    bounded = validate_stream_limit(limit)
    pinned = validate_stream_revision(revision)
    if client is None:
        raise FreshRetailDomainError(
            "No FreshRetailNet-50K streaming client configured; a default "
            "stream never performs network access. Pass an explicit `client=` "
            "(e.g. live_hf_stream_loader) for datasets.load_dataset streaming, "
            "or an injected loader in offline tests."
        )
    return _bounded_stream(loader=client, limit=bounded, split=split, revision=pinned)


def _bounded_stream(
    *,
    loader: StreamRowLoader,
    limit: int,
    split: str,
    revision: str,
) -> Iterator[DailySourceRow]:
    """Produce filas validadas desde ``loader``, deteniéndose exactamente en ``limit``.

    ``islice`` garantiza que la fuente nunca se avanza más allá de ``limit`` y
    que la fuente completa nunca se materializa. La validación de filas es la
    misma :func:`fresh_retail.parse_daily_row` canónica que usa la ruta basada
    en listas.
    """
    for raw in islice(loader(revision, split), limit):
        yield parse_daily_row(raw, revision=revision)
