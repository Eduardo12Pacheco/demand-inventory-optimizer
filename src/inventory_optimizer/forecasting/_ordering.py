"""Ayudantes privados compartidos de ordenamiento para colecciones de filas de pronóstico.

Se mantienen privados: solo la API pública de pronóstico garantiza semánticas de
orden y unicidad. Se soportan filas con atributos ``product_id`` y ``dt``.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Protocol, TypeVar

_RowT = TypeVar("_RowT", bound="RowLike")


class RowLike(Protocol):
    """Cualquier cosa con un product_id y una fecha diaria."""

    product_id: str | int
    dt: date


class DuplicateKeyError(Exception):
    """Se lanza cuando las filas contienen claves (product_id, dt) repetidas."""


def product_sort_key(product_id: str | int) -> tuple[int, str]:
    """Clave de orden determinista y segura ante tipos mixtos para ids de producto.

    Los ints se ordenan antes que las cadenas no numéricas; dentro de cada grupo
    las formas textuales se ordenan lexicográficamente. Las cadenas con aspecto
    numérico colapsan a ints durante la normalización de la ingesta, así que el
    int 38 y la cadena "38" no pueden aparecer ambos.
    """
    if isinstance(product_id, int):
        return (0, str(product_id))
    return (1, str(product_id))


def validate_unique_and_sort(rows: Iterable[_RowT]) -> list[_RowT]:
    """Rechaza claves (product_id, dt) duplicadas y ordena de forma determinista.

    Los duplicados lanzan :class:`DuplicateKeyError` listando las claves
    infractoras; las filas nunca se deduplican en silencio. La salida se ordena
    por ``(product_id, dt)`` y nunca se mezcla.
    """
    materialized = list(rows)
    seen: dict[tuple[str | int, date], list[int]] = {}
    for index, row in enumerate(materialized):
        seen.setdefault((row.product_id, row.dt), []).append(index)
    duplicates = {key: indexes for key, indexes in seen.items() if len(indexes) > 1}
    if duplicates:
        sample = ", ".join(
            f"{key[0]!r}@{key[1].isoformat()}"
            for key in sorted(duplicates, key=lambda item: (product_sort_key(item[0]), item[1]))
        )
        raise DuplicateKeyError(
            f"Duplicate (product_id, dt) keys are not allowed: {sample}."
        )
    materialized.sort(key=lambda row: (product_sort_key(row.product_id), row.dt))
    return materialized
