# Ejecutor de evaluación de líneas base offline

> Lo que se puede ejecutar hoy: una evaluación determinista de las cuatro líneas
> base sobre particiones temporales explícitas, con un reporte tipado y
> serializable a JSON. Están registrados una evaluación offline sintética y una
> evaluación real acotada bajo `baseline-evaluation-v2`. Aquí no se ejecutan una
> evaluación de instantánea completa, entrenamiento de modelos ni selección de
> ganador.

## Camino rápido

1. Construir filas validadas (filas sintéticas offline, p. ej. el fixture de
   `tests/helpers.py`) y una `EvaluationConfig` con particiones explícitas.
2. Llamar a `run_baseline_evaluation(rows, config)`.
3. Leer `EvaluationReport.results` (una fila por partición/modelo) y
   `EvaluationReport.fold_statuses` (reportes de calidad de datos), o
   serializar con `report.to_json()`.

Ejemplo offline mínimo (fechas y configuración explícitas, sin red):

```python
from datetime import date

from inventory_optimizer.forecasting import (
    BaselineEvaluationConfig,
    EvaluationConfig,
    EvaluationReport,
    EvaluationSplitConfig,
    FoldEvaluationConfig,
    run_baseline_evaluation,
)

# Filas sintéticas validadas (fixture offline; nunca una descarga en vivo).
rows = [...]  # Iterable[DailySourceRow] en dataset_revision "test-rev-eval-1"

config = EvaluationConfig(
    source_id="fresh-retailnet-50k-dev",
    dataset_revision="test-rev-eval-1",  # debe ser igual a la revisión de cada fila
    evaluation_id="eval-dev-001",
    split_config=EvaluationSplitConfig(
        folds=(
            FoldEvaluationConfig(
                fold="fold-1",
                train_start=date(2024, 1, 1), train_end=date(2024, 1, 10),
                validation_start=date(2024, 1, 11), validation_end=date(2024, 1, 13),
                test_start=date(2024, 1, 14), test_end=date(2024, 1, 16),
            ),
            FoldEvaluationConfig(
                fold="fold-2",
                train_start=date(2024, 1, 1), train_end=date(2024, 1, 13),
                validation_start=date(2024, 1, 14), validation_end=date(2024, 1, 16),
                test_start=date(2024, 1, 17), test_end=date(2024, 1, 19),
            ),
        ),
        evaluation_partition="test",  # "validation" o "test", nunca "train"
        product_group="all_products",
    ),
    baseline_config=BaselineEvaluationConfig(),  # naive, seasonal_naive, moving_average, ses
)

report: EvaluationReport = run_baseline_evaluation(rows, config)
for result in report.results:
    print(result.fold, result.model, result.mae, result.coverage)
print(report.to_json())  # determinista, json de la stdlib, sin filas crudas
```

## Detalles

| Tema | Decisión |
|-------|----------|
| API del ejecutor | `run_baseline_evaluation(rows, config)` en `src/inventory_optimizer/forecasting/evaluation.py`, exportada desde `inventory_optimizer.forecasting` |
| Particiones | Lista explícita de `FoldEvaluationConfig`, convertida a través del `split_temporal` existente — todas las comprobaciones de división (límites inclusivos, orden estricto, rechazo de duplicados, reporte de fechas faltantes e historia insuficiente) siguen siendo autoritativas; los identificadores de partición duplicados se rechazan y se preserva el orden de entrada de las particiones |
| Partición de evaluación | Exactamente una de `"validation"` o `"test"`; train nunca se evalúa. El historial de ajuste sigue el protocolo versionado (`baseline-evaluation-v2`): la evaluación de validación ajusta SOLO `split.train` (la validación es solo una comprobación); la evaluación de prueba ajusta el `split.train + split.validation` ordenado y con duplicados verificados; las filas de prueba nunca entran al historial de ajuste y los parámetros de las líneas base están fijos (sin auto-ajuste) |
| Líneas base | Los cuatro modelos requeridos corren en cada partición (`naive`, `seasonal_naive`, `moving_average` con `window` explícito, `ses` con `alpha` explícito), ajustados EXCLUSIVAMENTE con el historial de ajuste del protocolo (`split.train` para la evaluación de validación; `split.train + split.validation` para la de prueba); las listas de modelos faltantes/duplicadas/desconocidas, las entradas no textuales y los parámetros inválidos (ventana bool/no int, alpha bool/no numérico) se rechazan con errores claros |
| Métricas | `evaluate_metrics` sobre pares evaluados, llevadas con el id de partición: MAE, RMSE, WMAPE (con estado explícito de denominador cero / sin evaluables), sesgo, conteos de solicitadas/disponibles/evaluadas, cobertura, conteo de filas con quiebre |
| Forma del reporte | Filas deterministas en orden de partición y luego en el orden de modelos requerido; estado a nivel de partición con conteos de filas, fechas faltantes globales y por producto, ids de producto con historia insuficiente, conteos fuera de rango, y campos de ajuste del protocolo por partición/resultado (`fit_partition`, `fit_row_count`, `fit_history_start`/`fit_history_end`, `test_excluded_from_fit`, `insufficient_fit_history`) |
| Configuración | Las configs tipadas rechazan los tipos incorrectos con claridad: ids de partición y mínimos de historia no textuales o booleanos, etiquetas no textuales, entradas que no sean `FoldEvaluationConfig` o ids de partición duplicados, entradas de modelo no textuales y tipos anidados incorrectos fallan con `ValueError` en la construcción |
| Serialización | `to_dict()` / `to_json()` solo con `json` de la stdlib, `sort_keys=True`, fechas ISO, sin `DailySourceRow`, vectores horarios ni filas de predicción |
| Reproducibilidad | Filas + config idénticas ⇒ reporte tipado idéntico y cadena JSON idéntica |

## Protocolo de evaluación (versionado)

`EVALUATION_PROTOCOL_VERSION = "baseline-evaluation-v2"` identifica el protocolo
corregido de historial de ajuste. Cada reporte lleva un sobre de protocolo a
nivel de reporte más los campos de ajuste por partición/resultado; todas las
fechas se serializan como cadenas ISO y el JSON permanece sin datos crudos y
determinista.

| Campo | Significado |
| --- | --- |
| `evaluation_protocol_version` | Identidad del protocolo (`baseline-evaluation-v2`) |
| `evaluation_partition` | `"validation"` o `"test"` (nunca `"train"`) |
| `fit_partition` | `"train"` para la evaluación de validación, `"train+validation"` para la de prueba (particiones del protocolo; un conjunto vacío contribuye cero filas) |
| `fit_row_count` | Número real de filas pasadas a `fit_baseline` (por partición/resultado; nunca un agregado inventado cuando las particiones difieren) |
| `fit_history_start` / `fit_history_end` | Límites de fecha reales de las filas de ajuste (cadenas ISO; `null` para historial de ajuste vacío) |
| `test_excluded_from_fit` | Siempre `true`: las filas de prueba nunca entran al historial de ajuste; las claves de ajuste y de prueba son disjuntas y el historial de ajuste termina antes de que comience la prueba (aplicado antes de ajustar) |
| `insufficient_fit_history` | Estado tipado por producto derivado de las filas de ajuste REALES y los mínimos configurados de la partición; el `insufficient_history` existente basado en train permanece intacto |
| `baseline_parameters_fixed` | Siempre `true`: `moving_average_window` / `ses_alpha` explícitos desde `BaselineEvaluationConfig`; sin tuning, ranking ni selección de ganador |

Bajo este protocolo `seasonal_naive` puede usar una observación de validación
como su rezago de 7 días cuando se evalúa la prueba; el ajuste antiguo de solo
train hacía esos rezagos no disponibles (el reporte real de diagnóstico antiguo
mostró cobertura 0 del naive estacional por esa razón — ese reporte permanece
solo como diagnóstico y está superado). El target sigue siendo exactamente
`observed_sales`; los metadatos de quiebre son solo diagnóstico; la demanda
latente nunca se estima y los quiebres no se corrigen ni se enmascaran.

## Estado actual vs. estado futuro

| | Ahora (este hito) | Luego (no ejecutado aquí) |
|---|---|---|
| Filas de entrada | Fixtures sintéticos pequeños más un prefijo real acotado del cargador de streaming | Una instantánea real completa en la revisión fijada |
| Evaluación | Evaluación de desarrollo offline más una evaluación real de 1.000 filas bajo el protocolo corregido `baseline-evaluation-v2`; el reporte v1 anterior permanece solo como diagnóstico | El mismo ejecutor sobre una instantánea fijada completa |
| Alcance | Cuatro líneas base sobre `observed_sales` | Pronóstico avanzado, intervalos/cuantiles, recuperación de demanda latente, simulación, optimización, tableros |

## Evaluaciones reales acotadas

Dos evaluaciones reales acotadas están registradas para la misma revisión de
fuente. Ambas cargaron exactamente 1.000 filas con `streaming=True`, usaron dos
particiones de prueba explícitas y escribieron solo reportes JSON agregados y
sin datos crudos.

El reporte anterior es solo de diagnóstico:

`data/evaluations/freshretailnet-50k-bounded-real-1000.json`

Usó la regla antigua de ajuste solo-train, así que `seasonal_naive` tuvo
cobertura de prueba cero porque sus rezagos de 7 días estaban en validación. Se
preserva sin cambios y no es el benchmark final.

La corrida con el protocolo corregido es el benchmark acotado final:

`data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json`

Usa `baseline-evaluation-v2`: la validación es solo una comprobación con ajuste
de train; la prueba ajusta el historial ordenado y con duplicados verificados de
`train + validation`, nunca incluyendo filas de prueba. `seasonal_naive` tiene
cobertura 1.0 en ambas particiones de prueba. El reporte registra los conteos de
filas de ajuste y los límites de fecha, la versión del protocolo, los parámetros
fijos de las líneas base y la exclusión de la prueba. No contiene filas crudas,
vectores horarios, predicciones individuales ni caché local de Hugging Face.
Ninguno de los dos reportes representa el conjunto de datos completo.

## Captura del visor

El gráfico de comparación y la tabla estática del visor se derivan SOLO del
reporte v2 congelado (`freshretailnet-50k-bounded-real-1000-v2.json`); el
reporte v1 de diagnóstico nunca se carga ni se muestra. Las capturas de
`docs/assets/` se regeneran contra la app en ejecución local (sin recortes ni
edición posterior):

```bash
uv sync --extra demo
uv run streamlit run app/streamlit_app.py --server.headless true
# en otra terminal, con la app arriba:
uv run --with playwright python scripts/capture_demo_screenshot.py \
  --width 1440 --height 900 --viewport-only \
  --output docs/assets/demo-viewer.png
uv run --with playwright python scripts/capture_demo_screenshot.py \
  --width 1440 --output docs/assets/demo-viewer-full.png
```

La captura principal es el viewport nativo de 1440×900; la secundaria es la
página completa a 1440 px de ancho. El script espera a que la app responda en
`/_stcore/health` antes de capturar y no modifica ningún dato del reporte.

## Contrato de métricas y limitaciones

- El target es exactamente `observed_sales = sale_amount` — ventas de la fuente
  observadas y posiblemente censuradas. Las métricas NUNCA se calculan contra la
  demanda latente/real; `latent_demand_estimate` permanece `None` y sin uso.
- Las limitaciones de quiebre/censura se preservan, no se ocultan: los metadatos
  de quiebre (`stockout_hours_6_22`, estado de observación) se llevan en cada
  fila evaluada y se cuentan, pero la cantidad latente durante quiebres no se
  estima.
- Una fila diaria es `censored_or_partial` cuando alguna hora tiene estado == 1;
  tales observaciones se preservan tal cual y nunca se enmascaran.
- Las predicciones `None` (historia faltante o rezago estacional faltante)
  permanecen representadas en los conteos de solicitadas/no disponibles, se
  excluyen del cálculo métrico y nunca fabrican números. Una predicción de
  exactamente `0.0` SÍ está disponible y se evalúa.
- Sin selección automática de mejor modelo: cada fila de partición/modelo
  permanece visible y ningún ganador se ranquea. Los parámetros de las líneas
  base están fijos antes de la prueba — sin auto-ajuste.

## Explícitamente fuera de alcance

- Modelos de pronóstico avanzados, pronósticos por intervalos/cuantiles,
  cobertura de intervalos de predicción.
- Recuperación de demanda latente y cualquier re-etiquetado de ventas observadas
  como demanda real.
- Simulación de inventario, optimización de políticas y tableros.
- Evaluación en vivo de instantánea completa, descarga del conjunto de datos
  completo o tratar el prefijo acotado como un benchmark de todo el conjunto.

## Validación

```bash
uv run pytest -q
uv run python -m compileall -q src/
uv lock --check
```
