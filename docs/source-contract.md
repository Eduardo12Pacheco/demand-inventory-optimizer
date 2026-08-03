# Contrato de fuente de FreshRetailNet-50K

> Contrato de fuente basado en evidencia para el conjunto de datos
> FreshRetailNet-50K tal como lo usan el adaptador de ingesta de este repositorio
> y el cargador de streaming acotado. Auditado contra la tarjeta del conjunto de
> datos fijada el 2026-08-02; la evidencia de la tarjeta se cita desde la
> revisión fijada abajo. Este documento es factual y conservador: nunca llama a
> las ventas observadas "demanda real".

## Identidad del conjunto de datos

| Elemento | Valor |
| --- | --- |
| ID del conjunto de datos | `Dingdong-Inc/FreshRetailNet-50K` |
| Página del conjunto de datos | https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K |
| Revisión fijada (SHA) | `08c1fab7f9257bc73679d415d65d644165d351d4` |
| Tarjeta del conjunto de datos fijada | https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/blob/08c1fab7f9257bc73679d415d65d644165d351d4/README.md |
| Desarrollador de datos | Dingdong-Inc |
| Lanzamiento | Versión 1.0, 05/08/2025 |
| Paper | https://arxiv.org/abs/2505.16319 |
| Repositorio de línea base | https://github.com/Dingdong-Inc/frn-50k-baseline |

## Licencia

- El front matter de la tarjeta fijada declara `license: cc-by-4.0` y afirma:
  "This dataset is ready for commercial/non-commercial use."
- La sección "License/Terms of Use" de la tarjeta afirma: "This dataset is
  licensed under the Creative Commons Attribution 4.0 International License (CC
  BY 4.0) available at https://creativecommons.org/licenses/by/4.0/legalcode."
- Bajo CC BY 4.0 el licenciatario puede compartir y adaptar el material — incluso
  con fines comerciales — siempre que se dé atribución, se incluya un enlace a la
  licencia y se indiquen los cambios. La licencia no permite restricciones
  adicionales más allá de sus propios términos; los derechos de terceros sobre
  los datos no están cubiertos por la licencia.
- La sección "Intended use" de la tarjeta: "The FreshRetailNet-50K Dataset is
  intended to be freely used by the community to continue to improve latent
  demand recovery and demand forecasting techniques. However, for each dataset
  an user elects to use, the user is responsible for checking if the dataset
  license is fit for the intended purpose."

## Atribución y cita

Texto de atribución a usar siempre que este conjunto de datos se comparta,
adapte o reporte (CC BY 4.0 exige atribución, un enlace a la licencia y aviso de
cualquier cambio):

> FreshRetailNet-50K dataset, version 1.0 (05/08/2025), developed and
> published by Dingdong-Inc
> (https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), licensed
> under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).

Ejemplo de aviso de cambio al adaptar o modificar los datos:

> This work adapts FreshRetailNet-50K (Dingdong-Inc, CC BY 4.0,
> https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K). Changes:
> [describe what was added, removed, or transformed].

Cita oficial, citada desde la tarjeta fijada:

```bibtex
@article{2025freshretailnet-50k,
      title={FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail},
      author={Yangyang Wang, Jiawei Gu, Li Long, Xin Li, Li Shen, Zhouyu Fu, Xiangjun Zhou, Xu Jiang},
      year={2025},
      eprint={2505.16319},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.16319},
}
```

## Análisis y uso permitidos

- La tarjeta lista el caso de uso como "Developers researching latent demand
  recovery and demand forecasting techniques" y afirma explícitamente que el
  conjunto de datos está listo para uso comercial y no comercial.
- Este repositorio está diseñado y previsto para usar el conjunto de datos en
  ingesta, validación, pronóstico básico de ventas observadas y análisis de
  quiebres. El pronóstico básico se implementa con líneas base deterministas; el
  pronóstico avanzado y la recuperación de demanda latente permanecen pendientes
  (ver README). Las ventas observadas nunca se re-etiquetan como demanda real;
  la recuperación de demanda latente permanece fuera de alcance en el momento de
  la ingesta.

## Condiciones de redistribución y salvedades

- Compartir y adaptar están permitidos bajo CC BY 4.0 con: atribución a
  Dingdong-Inc, un enlace a la licencia y aviso de cualquier cambio.
- Salvedades:
  1. Los derechos de terceros sobre los datos no están cubiertos por la licencia
     CC BY 4.0.
  2. La tarjeta fijada, el paper y el repositorio de línea base discrepan en los
     conteos de SKU y en la semántica del estado (ver "Discrepancias" abajo).
  3. Según la propia guía de la tarjeta, cada usuario debe verificar que la
     licencia se ajuste a su propósito previsto.
- Este repositorio no redistribuye el conjunto de datos; solo lo documenta y
  consume.

## Política de no persistencia / fixtures

- No se almacenan, confirman ni redistribuyen filas en vivo de FreshRetailNet-50K
  en este repositorio. Los fixtures offline de prueba son filas sintéticas
  pequeñas que replican la forma oficial solo para pruebas de validación; no se
  agrega ningún fixture redistribuible de datos en vivo.
- La ingesta corre ya sea transmitida (acotada, bajo demanda) o desde fixtures
  sintéticos; el repositorio nunca persiste una descarga en vivo completa o
  parcial.

## Esquema oficial relevante para este adaptador

"Data Fields" de la tarjeta fijada (extracto; tipos tal como se publicaron en la
tarjeta):

| Campo | Tipo | Descripción de la tarjeta |
| --- | --- | --- |
| `product_id` | int64 | "The encoded product id" |
| `dt` | string | "The date" |
| `sale_amount` | float64 | "The daily sales amount after global normalization (Multiplied by a specific coefficient)" |
| `hours_sale` | Sequence(float64) | "The hourly sales amount after global normalization (Multiplied by a specific coefficient)" |
| `stock_hour6_22_cnt` | int32 | "The number of out-of-stock hours between 6:00 and 22:00" |
| `hours_stock_status` | Sequence(int32) | "The hourly out-of-stock status" |

Campos de la tarjeta que este adaptador no usa: `city_id`, `store_id`,
`management_group_id`, `first_category_id`, `second_category_id`,
`third_category_id`, `discount`, `holiday_flag`, `activity_flag`, `precpt`,
`avg_temperature`, `avg_humidity`, `avg_wind_level`.

Mapeo del adaptador (contrato aceptado, sin cambios):

- `sales_qty_observed = hours_sale[h]` en las unidades normalizadas de la
  fuente, nunca enmascarado ni reemplazado.
- `stockout_flag = hours_stock_status[h] == 1` (estado 1 = quiebre de stock).
- `stockout_hours_6_22 = sum(hours_stock_status[6:22])`; el contador oficial
  `stock_hour6_22_cnt` cuenta las horas de quiebre entre las 6:00 y las 22:00 y
  debe ser igual a esa suma de ventana semiabierta.
- `latent_demand_estimate = None` en el momento de la ingesta.

## Conteos de la fuente

Según la salida "How to use it" de la tarjeta fijada:

- `train`: 4,500,000 filas
- `eval`: 350,000 filas

## Discrepancias

1. **Conteo de SKU.** El resumen de la tarjeta fijada dice que el conjunto de
   datos abarca "865 perishable SKUs", mientras que el resumen del paper (arXiv
   v5 en https://arxiv.org/abs/2505.16319) dice "863 perishable SKUs". El
   invariante de instantánea completa del repositorio sigue a la tarjeta fijada
   (865 valores únicos de `product_id`).
2. **Semántica del estado.** El paper define una ecuación en la que el estado 1 =
   stock disponible y 0 = quiebre, mientras que la descripción del campo de la
   tarjeta fijada ("The hourly out-of-stock status") y la muestra auditada
   establecen el estado 1 = quiebre. Este adaptador sigue la tarjeta y la muestra
   auditada: estado 1 = quiebre.
3. **Repositorio de línea base.** El README de la línea base
   (https://github.com/Dingdong-Inc/frn-50k-baseline) carga el conjunto de datos
   sin una revisión fijada, y
   `latent_demand_recovery/exp/data/generate_data.py` aplica
   `hours_sale = np.where(hours_stock_status==1, np.nan, hours_sale_origin)` —
   un enmascaramiento destructivo de las ventas observadas usado para la
   preparación de datos de recuperación de demanda latente, no semántica de
   ingesta. Este repositorio nunca enmascara las ventas observadas en el momento
   de la ingesta.

## Garantías del cargador

El cargador de streaming acotado (`fresh_retail_stream.stream_fresh_retail_50k`):

- requiere un `limit` entero positivo y aplica un máximo duro de 1.000 filas por
  stream;
- acepta solo la revisión fijada
  `08c1fab7f9257bc73679d415d65d644165d351d4` (las revisiones vacías, con
  espacios, no textuales y cualquier otra se rechazan antes de cualquier llamada
  al cliente);
- usa por defecto `split="train"` y siempre pasa un split explícito al cliente;
- transmite a través de `datasets.load_dataset` con `streaming=True`,
  deteniéndose exactamente en el límite sin materializar la fuente completa;
- no realiza acceso a la red en una llamada por defecto y falla cerrado con un
  error de dominio — el cliente en vivo nunca se invoca en silencio (pase un
  `client=` explícito, p. ej. `live_hf_stream_loader`, para streaming en vivo);
- valida cada fila con el mismo contrato canónico `parse_daily_row` que la ruta
  basada en listas y propaga los errores del cliente/red de forma transparente;
- mantiene `datasets` como dependencia opcional e importada de forma perezosa
  (instalar con `uv sync --extra streaming`).

## Prueba de humo en vivo

La prueba de humo del mantenedor está acotada a exactamente 1.000 filas (el
máximo duro del cargador de streaming). Requiere el paquete opcional `datasets`
vía `uv sync --extra streaming`:

```bash
uv run python - <<'PY'
from inventory_optimizer.ingestion.fresh_retail_stream import (
    FRESH_RETAIL_PINNED_REVISION,
    live_hf_stream_loader,
    stream_fresh_retail_50k,
)
rows = list(stream_fresh_retail_50k(limit=1000, client=live_hf_stream_loader))
assert len(rows) == 1000
assert all(row.revision == FRESH_RETAIL_PINNED_REVISION for row in rows)
print("live smoke OK:", len(rows), "rows at", rows[0].revision)
PY
```

Salida esperada en caso de éxito:

```text
live smoke OK: 1000 rows at 08c1fab7f9257bc73679d415d65d644165d351d4
```

Resultado de la ejecución: **superada con exactamente 1.000 filas** en la
revisión fijada `08c1fab7f9257bc73679d415d65d644165d351d4`. El proceso de humo
usó directorios de caché temporales de Hugging Face y los eliminó al salir; no se
persistió en el repositorio ninguna fila en vivo ni fixture redistribuible.
