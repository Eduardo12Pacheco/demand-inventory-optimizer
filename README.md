# Demand Inventory Optimizer

> Un producto de datos integral, primero en Python, para pronóstico de demanda
> comparable globalmente, planificación de inventario y simulación y optimización de quiebres de stock.

## Estado del proyecto

**Pronóstico temporal básico implementado (sintético más una evaluación real acotada, con protocolo corregido).** El adaptador de ingesta de FreshRetailNet-50K (revisión de fuente fijada en `08c1fab7f9257bc73679d415d65d644165d351d4`) y sus pruebas de validación están implementados y pasan, al igual que el cargador de streaming acotado (prueba de humo real de exactamente 1.000 filas en la revisión fijada; no se persiste ninguna fila en vivo). Las divisiones temporales deterministas, solo con stdlib, los objetivos de ventas observadas, cuatro líneas base explicables, las métricas de error de pronóstico y un ejecutor de evaluación de líneas base reproducible y offline están implementados y probados bajo `baseline-evaluation-v2`: la evaluación de validación ajusta solo `train`; la evaluación de prueba ajusta `train + validation` ordenados; las filas de prueba nunca se ajustan. La evaluación real anterior de 1.000 filas sigue siendo solo diagnóstica porque usó el ajuste antiguo de solo entrenamiento; una nueva evaluación de 1.000 filas bajo el protocolo corregido queda registrada como el benchmark acotado en `data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json`. Un visor de Streamlit de solo lectura para ese benchmark está implementado (`app/streamlit_app.py`, extra opcional `demo`) y está etiquetado como visor de benchmark acotado — no es un modelo de producción ni una política de inventario. Aún quedan pendientes la evaluación de snapshot completa, la simulación, la optimización, el pronóstico avanzado y la recuperación de demanda latente. Las ventas observadas nunca se reetiquetan como demanda real, no hay ajuste automático de parámetros y no se realiza ninguna selección automática de mejor modelo. Este README es el contrato vivo del proyecto.

## Problema

Las decisiones de inventario son una compensación constante: demasiado poco stock
significa quiebres y ventas perdidas; demasiado, costos de mantenimiento y
desperdicio. La mayoría de los pronósticos de demanda lucen mejor de lo que son
porque se evalúan con fuga de información o sobre divisiones irreales, y las
líneas base simples nunca se reportan. El resultado son políticas de inventario
ajustadas a un pronóstico excesivamente optimista.

Este proyecto construirá un pipeline reproducible y comparable de pronóstico y
optimización de inventario sobre un conjunto de datos público, con evaluación
temporal honesta y compensaciones de negocio visibles.

## Objetivo del producto

Construir una plataforma de datos pequeña con estilo de producción que responda
preguntas como:

- ¿Qué tan bien pronostican la demanda las líneas base simples, y cuánto mejor es cualquier modelo avanzado?
- ¿Qué política de inventario (punto de reorden, nivel de orden hasta) minimiza el costo total en un nivel de servicio elegido?
- ¿Cómo se compensan los quiebres y los costos de mantenimiento al cambiar el nivel de servicio?
- ¿Dónde duele realmente el error de pronóstico en el negocio?

El producto debe hacer explícita la compensación de negocio: el nivel de servicio,
el costo de quiebre y el costo de mantenimiento son insumos de la decisión, no
constantes ocultas.

## Usuarios previstos

- Planificadores de inventario y analistas de operaciones que exploran compensaciones de políticas.
- Estudiantes y candidatos que demuestran habilidades de pronóstico y optimización.
- Cualquier persona que evalúe métodos de pronóstico comparables sobre datos públicos.

## Qué hará el sistema

1. **Seleccionar un conjunto de datos público durante la auditoría de fuente** (candidatos: competencias minoristas de demanda como M5 Accuracy y Store Sales/Favorita, o un conjunto equivalente comparable — la elección final la decide la auditoría).
2. **Ingerir y validar** los datos con un esquema documentado.
3. **Dividir temporalmente** con walk-forward, evaluación fuera de muestra y sin fuga (por grupo producto/tienda y horizonte temporal final).
4. **Establecer líneas base** (naive, naive estacional, promedio móvil, suavizamiento exponencial) antes de cualquier modelo avanzado.
5. **Reportar el error de pronóstico** (MAE, RMSE, WMAPE, sesgo, cobertura de intervalo).
6. **Simular políticas de inventario** (por ejemplo, punto de reorden/orden hasta) contra la demanda realizada con costos explícitos de quiebre y mantenimiento.
7. **Optimizar** los parámetros de la política para un nivel de servicio u objetivo de costo declarado.
8. **Exponer** los resultados mediante una demo interactiva de escenarios.
9. **Reportar** supuestos, limitaciones y comportamiento ante fallas.

## Visión de la demo

La demo final debe entenderse en menos de un minuto y mostrar más que un gráfico
de camino feliz:

- Una vista de pronóstico con reales, línea base y modelo avanzado sobre el horizonte fuera de muestra.
- Una vista de comparación de errores: qué método gana, por cuánto y en qué productos.
- Un simulador de escenarios con controles deslizantes para nivel de servicio, plazo de entrega y supuestos de costo.
- Una frontera de quiebre vs. costo que muestre la compensación de un vistazo.
- Una página de calidad de datos que muestre la última ingesta exitosa, el estado de la fuente y las limitaciones conocidas.
- Un estado de resultado vacío o de falla que explique qué pasó en lugar de fallar en silencio.

El proyecto tendrá un despliegue público cuando se entiendan las restricciones de
hospedaje. Hasta entonces, se requerirá una demo local reproducible y un recorrido
grabado corto.

## Alcance

### Producto mínimo viable

- Un conjunto de datos público auditado con un esquema documentado.
- Ingesta reproducible usando fixtures guardados para desarrollo offline.
- Divisiones temporales walk-forward con un horizonte fuera de muestra.
- Líneas base simples con métricas estándar de error de pronóstico.
- Un simulador de políticas de inventario con costos explícitos de quiebre y mantenimiento.
- Una demo de escenarios interactiva en Streamlit.
- Pruebas automatizadas para la lógica de división, métricas y simulación.
- Un README que documente el linaje de datos, las métricas, las limitaciones y la configuración.

### Versión 2

- Un modelo avanzado medido (por ejemplo, gradient boosting o un método probabilístico) solo después de reportar las líneas base.
- Pronósticos de intervalo y cuantiles con evaluación de cobertura.
- Optimización de políticas bajo restricciones (presupuesto, nivel de servicio).
- Conjuntos de datos de benchmark adicionales y comparación entre conjuntos.
- Servicio FastAPI y despliegue local con Docker.

### Explícitamente fuera de alcance

- Datos de demanda privados o propietarios.
- Afirmaciones de superioridad universal de cualquier método único.
- Optimizar costos que no estén declarados y defendidos.
- Integración con sistemas reales de aprovisionamiento.
- Construir un pronóstico ML opaco sin línea base y evaluación.

## Stack técnico propuesto

| Capa | Elección inicial | Propósito |
| --- | --- | --- |
| Lenguaje | Python | Implementación central y análisis |
| Entorno | `uv` | Gestión reproducible de dependencias y entorno |
| Calidad | Ruff, pytest, Pandera | Formato/lint, pruebas y contratos de datos |
| Ingesta | Adaptadores de fuente en Python, `httpx` | Recopilar datos sin acoplar el dominio a una sola fuente |
| Almacenamiento crudo | Fixtures JSON y Parquet | Preservar evidencia de fuente y permitir desarrollo offline |
| Analítica | pandas y DuckDB | Limpieza, análisis SQL y consultas analíticas locales |
| Líneas base | stdlib de Python (este hito) | Naive, naive estacional, promedio móvil, SES — deterministas y sin dependencias |
| Extensión ML | scikit-learn, luego | Mejoras de pronóstico medidas después de la línea base |
| Optimización | scipy, OR-Tools luego | Búsqueda de parámetros de política bajo restricciones |
| Demo | Streamlit para el MVP | Demostración pública rápida, legible e interactiva |
| API | FastAPI y Pydantic, luego | Frontera de servicio tipada para la aplicación |
| Entrega | Docker y GitHub Actions, luego | Ejecución reproducible y comprobaciones automatizadas |

Las elecciones tecnológicas están escalonadas deliberadamente: las líneas base y
las divisiones honestas vienen antes que los modelos, y la compensación de negocio
se define antes de la optimización.

## Habilidades demostradas

### Ciencia de datos

- Encuadre del problema y definición de métricas (WMAPE, sesgo, cobertura).
- Validación temporal: divisiones walk-forward y prevención de fuga.
- Evaluación de modelos con líneas base primero.
- Incertidumbre y pronóstico por intervalos (en la versión 2).

### Ingeniería de datos

- Adaptadores de fuente e ingesta resiliente.
- Capas de datos cruda, limpia y analítica.
- Diseño de esquemas y contratos de datos.
- Manejo de fallas, frescura e idempotencia.
- Analítica SQL con DuckDB.

### IA y ML

- Disciplina de líneas base simples: los modelos avanzados deben justificar su costo.
- Pronóstico probabilístico como extensión medida.
- Reporte honesto fuera de muestra por producto y grupo de tienda.

### Optimización y análisis de negocio

- Simulación de políticas de inventario (punto de reorden, orden hasta).
- Modelos de costo explícitos: quiebres, mantenimiento y nivel de servicio.
- Fronteras de compensación en lugar de respuestas de un solo número.

### Software y full stack

- Módulos Python tipados y fronteras de servicio.
- Experiencia de usuario interactiva con Streamlit.
- Pruebas, documentación, configuración, logging y ejecución reproducible.
- Preparación para CI y despliegue.

### Comunicación profesional

- Declaración clara del problema y valor para las partes interesadas.
- Métricas, compensaciones, limitaciones y análisis de fallas.
- Un video corto de demo y una publicación de LinkedIn enfocada en el resultado, no en la lista de herramientas.

## Plan de evaluación

El proyecto reportará evidencia medible, incluida:

- Error de pronóstico de líneas base (MAE, RMSE, WMAPE, sesgo) sobre el horizonte fuera de muestra.
- Mejora de cualquier modelo avanzado sobre la mejor línea base, usando las mismas divisiones.
- Tasa de quiebre simulada y costo total en los niveles de servicio declarados.
- Frontera de costo: costo total a lo largo de los niveles de servicio (por ejemplo, 90/95/98 por ciento).
- Reproducibilidad desde un entorno limpio usando fixtures.
- Al menos una falla inducida intencionalmente (por ejemplo, días de ventas faltantes) y la respuesta del sistema.

Un resultado fuerte es una comparación honesta y comparable con compensaciones de
negocio visibles — no un modelo único que afirma ser el mejor en todas partes.

## Principios de datos y ética

- Usar solo conjuntos de datos públicos cuyas licencias permitan análisis, reproducción y publicación.
- Atribuir las fuentes de los datos y respetar sus términos.
- Documentar las limitaciones reales del conjunto de datos; no presentarlo como específico de Ecuador a menos que la auditoría diga lo contrario.
- Declarar explícitamente todos los supuestos de costo y nivel de servicio.
- Mantener un conjunto de datos de fixture para que el proyecto siga siendo ejecutable si una fuente cambia o desaparece.
- Nunca presentar resultados simulados como recomendaciones reales de negocio para una empresa específica.

## Estructura de repositorio planificada

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── inventory_optimizer/
│       ├── ingestion/
│       ├── forecasting/
│       ├── simulation/
│       ├── optimization/
│       └── config.py
├── tests/
├── data/
│   └── sample/
├── notebooks/
├── app/
├── docs/
└── .github/
    └── workflows/
```

Los conjuntos de datos crudos grandes, las credenciales y el estado de ejecución local no deben confirmarse.

## Estado de implementación

### Ingesta de FreshRetailNet-50K (implementada, probada)

Una fila diaria de la fuente validada se expande en 24 registros horarios
canónicos. La revisión de la fuente está fijada; las revisiones vacías o sin
fijar se rechazan.

| Campo canónico | Definición |
| --- | --- |
| `sales_qty_observed` | `hours_sale[h]`, unidades originales — nunca se enmascara ni reemplaza |
| `stockout_flag` | `hours_stock_status[h] == 1` |
| `latent_demand_estimate` | `None` (nunca se estima en la ingesta) |
| `sales_observation_state` | `'censored_or_partial'` cuando el estado es 1, si no `'uncensored'` |
| `stockout_hours_6_22` | `sum(hours_stock_status[6:22])` — derivado de la ventana semiabierta validada, no es un campo de la fuente |

Los valores crudos de la fuente se preservan en cada registro: `hours_sale_raw`,
`hours_stock_status_raw`, `stock_hour6_22_cnt_raw` (el contador de la fuente debe
ser igual a `sum(hours_stock_status[6:22])` — ventana semiabierta, índices 6..21;
el índice 22 no se cuenta). Las filas de la fuente usan la clave oficial `dt`
(con `date` aceptado como alias heredado) y valores numéricos de `product_id`,
normalizados de forma consistente (int64 y `"38"` son el mismo producto). La
validación a nivel de fila rechaza vectores horarios que no tengan longitud 24,
valores de estado de stock distintos de 0/1, contadores de stock inconsistentes
con el vector de estado y `sale_amount` que no coincida con `sum(hours_sale)` más
allá de la tolerancia numérica. Una validación global de instantánea separada
exige exactamente 865 valores únicos de `product_id` para instantáneas completas
y acepta contextos de fixture explícitos sin ese requisito.

### Cargador de streaming acotado (implementado, probado con humo)

`fresh_retail_stream.stream_fresh_retail_50k` transmite como máximo 1.000 filas
validadas desde `datasets.load_dataset` de Hugging Face (`streaming=True`) en la
revisión fijada exacta, sin materializar la fuente completa. Una llamada por
defecto no realiza acceso a la red y falla cerrado con un error de dominio; el
streaming en vivo requiere un `client=live_hf_stream_loader` explícito (extra
opcional: `uv sync --extra streaming`). La prueba de humo real consumió
exactamente 1.000 filas y no persistió ningún dato en vivo.

**Auditoría de licencia: completa (CC BY 4.0).** Ver
[docs/source-contract.md](docs/source-contract.md) para el contrato de fuente
completo: ID exacto del conjunto de datos, revisión fijada y URL de la tarjeta
fijada, términos de licencia, atribución y cita, uso permitido, salvedades de
redistribución, esquema oficial, conteos de la fuente y las discrepancias entre
la tarjeta fijada, el paper y el repositorio de línea base. **No se almacena
ningún dato en vivo de FreshRetailNet-50K en este repositorio**; los fixtures
offline son filas sintéticas pequeñas.

| Área | Estado |
| --- | --- |
| Adaptador fijado, validación de filas, transformación canónica, preservación de crudos | Implementado + probado |
| Validación global de instantánea (865 productos, contexto explícito completo/fixture) | Implementado + probado |
| Cargador de streaming acotado (límite ≤ 1.000, revisión fijada, falla cerrada por defecto) | Implementado + probado offline + prueba de humo de 1.000 filas superada |
| Divisiones temporales, objetivos de ventas observadas, cuatro líneas base, métricas de pronóstico | Implementado + probado (sintético offline) |
| Ejecutor de evaluación de líneas base offline reproducible (configs explícitas de partición/línea base, reporte determinista + JSON) | Implementado + probado (sintético offline) |
| Simulación, optimización, demo interactiva | No iniciado (por diseño) |
| Visor de benchmark acotado de solo lectura (extra `demo` de Streamlit, solo agregados) | Implementado + probado (arranque AppTest + interacción de partición) |
| Recuperación de demanda latente | No iniciado (fuera de alcance) |

### Pronóstico temporal básico (implementado, probado)

Pronóstico determinista, solo con stdlib, sobre ventas observadas, con pruebas
sintéticas offline pequeñas bajo `tests/` (sin fixtures de datos en vivo). Las
APIs públicas viven en `src/inventory_optimizer/forecasting/` (`splits.py`,
`targets.py`, `baselines.py`, `metrics.py`, `evaluation.py`).

**Definición del objetivo.** `observed_sales = sale_amount`. Las ventas son
observaciones de la fuente normalizadas globalmente y pueden estar censuradas
durante quiebres; NUNCA se llaman demanda real. `latent_demand_estimate`
permanece ausente/`None` y nunca se usa.

**Divisiones temporales.** `split_temporal` usa límites explícitos de día de
calendario inclusivos (`train_start`/`train_end`, opcionales `validation_*` y
`test_*`); los conjuntos ausentes son tuplas vacías explícitas con límites
`None` — sin horizontes ocultos, sin fechas falsas. Las filas nunca se mezclan,
cada conjunto se ordena por `(product_id, dt)`, y las claves duplicadas
`(product_id, dt)` lanzan un error en lugar de deduplicarse en silencio. Las
fechas de calendario faltantes dentro de los rangos solicitados se reportan
(globalmente y por producto, sobre todo el lapso solicitado) y nunca se
rellenan: nunca aparecen filas sintéticas. Los productos con muy poca historia
de entrenamiento observada aparecen como registros `InsufficientHistory`
explícitos (días/lapso observados vs. mínimos requeridos). Las filas fuera de
los rangos solicitados se excluyen y se cuentan en `excluded_out_of_range`.
`expanding_window_folds` construye particiones walk-forward a partir de
`initial_train_days`, `validation_days`, `folds` y `step_days` explícitos, más
un horizonte de prueba explícito opcional (`test_start` + `test_days`): el
inicio del entrenamiento permanece fijo mientras el final del entrenamiento se
expande, la validación sigue inmediatamente, y ninguna fila futura de
validación/prueba entra jamás al entrenamiento de una partición (reutilizar
filas entre particiones de entrenamiento expansivas es esperado). Cada partición
expone sus propios límites de fecha y sus propios reportes de fechas faltantes e
historia insuficiente.

**Líneas base.** Cuatro líneas base explicables, cada una ajustada
EXCLUSIVAMENTE con las filas de entrenamiento pasadas y prediciendo solo filas
target futuras (se rechazan las fechas target en o antes de la última fecha de
entrenamiento de un producto):

| Línea base | Predicción |
| --- | --- |
| `naive` | Último `sale_amount` observado por producto |
| `seasonal_naive` | Valor observado exactamente 7 días de calendario antes de la fecha target, buscado solo en filas de entrenamiento |
| `moving_average(window)` | Media de los últimos `window` valores de entrenamiento observados por producto (todos los valores observados cuando la historia es más corta) |
| `ses(alpha)` | Nivel final de suavizamiento exponencial simple ajustado solo sobre las observaciones de entrenamiento de cada producto (`alpha` en `[0, 1]`) |

La forma de resultado común (`ForecastRow` / `BaselineForecastResult`) preserva
`product_id`, la fecha target, `observed_sales`, `stockout_hours_6_22`, el
estado de observación y la revisión de la fuente, y reporta conteos de
solicitadas/disponibles/no disponibles más la cobertura (`available /
requested`). Las predicciones no disponibles permanecen `None` con una razón
explícita (historia faltante o rezago estacional faltante); nunca se eliminan,
recortan, imputan, enmascaran ni re-etiquetan.

**Métricas.** `evaluate_metrics` reporta sobre pares evaluados (predicción y
`observed_sales` ambos presentes) con las fórmulas exactas:

- `MAE = mean(|actual − prediction|)`
- `RMSE = sqrt(mean((actual − prediction)²))`
- `WMAPE = sum(|actual − prediction|) / sum(|actual|)` — `None` numérico con estado `not_calculable_zero_actuals` cuando el denominador es cero
- `bias = mean(prediction − actual)`

El reporte tipado incluye modelo, identificador de partición, etiqueta target
exactamente `observed_sales`, conteos de solicitadas/disponibles/evaluadas,
cobertura, conteo de filas con quiebre (filas evaluadas con `stockout_hours_6_22
> 0`), las cuatro métricas y un estado WMAPE explícito (`calculable`,
`not_calculable_zero_actuals` o `no_evaluable_predictions`). Sin predicciones
evaluables, las métricas son `None` en lugar de números fabricados.

**Limitaciones de quiebre/censura.** Una fila diaria es `censored_or_partial`
cuando alguna hora tiene estado == 1; tales observaciones se preservan tal cual
y nunca se enmascaran (sin cambios respecto del contrato de ingesta), y las
filas con estado == 1 permanecen preservadas en cada resultado de pronóstico. La
cantidad latente durante quiebres no se estima ni modela aquí.

**Aún no implementado (por diseño):** modelos de pronóstico avanzados,
pronósticos por intervalos y cuantiles, y recuperación de demanda latente.

### Ejecutor de evaluación de líneas base offline (implementado, probado)

Evaluación offline determinista de las cuatro líneas base sobre particiones
temporales explícitas, con un reporte tipado y JSON sin datos crudos. El
contrato completo y un ejemplo mínimo de uso offline viven en
[docs/evaluation.md](docs/evaluation.md).

| Decisión | Valor |
| --- | --- |
| Configuración | `EvaluationConfig` (source id / revisión de conjunto de datos / id de evaluación no vacíos) + `EvaluationSplitConfig` (particiones explícitas, `evaluation_partition` en `validation`/`test`, `product_group` agregado) + `BaselineEvaluationConfig` (los cuatro modelos requeridos en orden determinista, `moving_average_window` y `ses_alpha` explícitos) |
| Particiones | Convertidas a través del `split_temporal` existente; cada comprobación de división y reporte de calidad de datos sigue siendo autoritativo; sin horizontes ocultos ni conteos de particiones inventados |
| Objetivos | Solo la partición configurada se proyecta vía `project_targets`; train nunca se evalúa. El historial de ajuste sigue el protocolo versionado (`baseline-evaluation-v2`): la evaluación de validación ajusta SOLO `split.train`; la evaluación de prueba ajusta el `split.train + split.validation` ordenado y con duplicados verificados; las filas de prueba nunca entran al historial de ajuste (claves de ajuste/prueba disjuntas, el ajuste termina antes de que comience la prueba) |
| Líneas base | `naive`, `seasonal_naive`, `moving_average(window)`, `ses(alpha)` ajustadas EXCLUSIVAMENTE con el historial de ajuste del protocolo (`split.train` para la evaluación de validación; `split.train + split.validation` para la de prueba); las entradas de modelo faltantes/duplicadas/desconocidas o no textuales y los parámetros inválidos (no int/bool) fallan claramente |
| Métricas | `evaluate_metrics` por partición/modelo con id de partición: MAE/RMSE/WMAPE/sesgo, estado WMAPE explícito, conteos de solicitadas/disponibles/evaluadas, cobertura, conteo de filas con quiebre |
| Reporte | `results` deterministas (orden de particiones, luego el orden de modelos requerido) + `fold_statuses` (conteos de filas, fechas faltantes globales y por producto, ids de producto con historia insuficiente, conteos fuera de rango); sobre de protocolo a nivel de reporte (`evaluation_protocol_version`, `evaluation_partition`, `fit_partition`, `test_excluded_from_fit`, `baseline_parameters_fixed: true`); campos de ajuste por partición/resultado (`fit_partition`, `fit_row_count`, `fit_history_start`/`fit_history_end`, `test_excluded_from_fit`, `insufficient_fit_history`); target exactamente `observed_sales`; configuración reproducible en cada fila |
| Serialización | `to_dict()`/`to_json()` solo con `json` de la stdlib, `sort_keys=True`, fechas ISO; sin filas crudas, vectores horarios ni filas de predicción; filas + config idénticas ⇒ JSON idéntico |

Garantías de honestidad: las predicciones `None` siguen representadas (conteos de
solicitadas/no disponibles) y se excluyen del cálculo métrico; los estados WMAPE
de reales cero y sin evaluables son explícitos; los metadatos de quiebre se
preservan y cuentan; NO hay selección automática de mejor modelo — cada fila de
partición/modelo permanece visible. Los parámetros de las líneas base están
fijos antes de la prueba (sin auto-ajuste); las filas de prueba nunca se ajustan;
la demanda latente nunca se estima y los quiebres no se corrigen ni se enmascaran.

Dos evaluaciones reales acotadas están registradas para la misma fuente fijada.
El primer reporte, `data/evaluations/freshretailnet-50k-bounded-real-1000.json`,
es solo de diagnóstico porque usó el ajuste antiguo de solo entrenamiento y la
cobertura de prueba del naive estacional fue 0. Permanece sin cambios. El
benchmark acotado corregido es
`data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json`: exactamente
1.000 filas transmitidas, dos particiones de prueba explícitas,
`fit_partition=train+validation`, prueba excluida del ajuste y cobertura del
naive estacional 1.00 en ambas particiones. Ambos reportes no llevan datos
crudos y ninguno representa una instantánea completa del conjunto de datos.

### Visor de benchmark acotado (demo)

Un visor de Streamlit de solo lectura y solo agregados renderiza exactamente el
reporte de benchmark v2 congelado en una ruta fija relativa al repositorio
(`data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json`). Es un
**visor de benchmark acotado, no un modelo de producción ni una política de
inventario**; no realiza solicitudes de datos en vivo, nunca carga el reporte
v1 de diagnóstico y no expone controles de carga/ruta/URL.

![Visor de benchmark acotado](docs/assets/demo-viewer.png)

Captura de página completa (secundaria, opcional):
[demo-viewer-full.png](docs/assets/demo-viewer-full.png).

**Qué muestra (todo derivado del reporte en tiempo de ejecución, nada
hardcodeado):**

- Resumen visual: hechos de escala (1.000 filas transmitidas, 12 productos,
  2 particiones, 4 líneas base), el hallazgo de no-ganador
  («Este benchmark acotado no selecciona automáticamente un ganador.») y una
  tarjeta de conclusión por partición con el menor valor de cada métrica,
  derivado de los datos actuales del reporte.
- Bloque de protocolo temporal: 2 particiones, cada una
  `entrenamiento → validación → prueba` con fechas y conteos; el historial de
  ajuste de la prueba = entrenamiento + validación ordenados (conteos de
  ajuste por partición 769 / 923); las filas de prueba nunca entran al
  historial de ajuste; parámetros de línea base fijos (ventana de promedio
  móvil de 7 días; alpha de suavizamiento exponencial de 0.3), sin ajuste
  automático.
- Un selector de métrica (MAE / RMSE / WMAPE) y un gráfico de comparación por
  partición construido solo con HTML/CSS propio (sin librerías de gráficos):
  barras con longitud relativa al mayor valor de cada partición, valor exacto
  de 3 decimales junto a cada barra, un marcador «menor valor» sobre el mínimo
  de cada partición y una tabla equivalente oculta para lectores de pantalla.
- Un filtro de partición (`Todas las particiones` / `Partición 1` /
  `Partición 2`) y una tabla de resultados agregados en el orden fijo del
  reporte — nunca ordenados por ranking.
- Observaciones calculadas de los resultados actuales por partición y métrica,
  con la afirmación explícita de que el benchmark no selecciona ningún
  ganador automático.
- Datos de quiebres de stock del reporte (593 filas con quiebre; 708 filas con
  alguna hora en estado de quiebre; filas con quiebre en la prueba por
  partición: 34 y 33), con bandas de proporción y la salvedad de censura: las
  ventas observadas siguen siendo ventas observadas, no existe estimación de
  demanda latente y los quiebres no se corrigen ni se enmascaran.
- Metodología: flujo del pipeline (fuente → transmisión acotada → particiones
  temporales → líneas base → métricas → análisis de limitaciones) y un bloque
  explicativo por línea base con sus parámetros reales del reporte.
- Una nota histórica del protocolo claramente etiquetada: el reporte v1 de
  diagnóstico (que el visor nunca lee) registró una cobertura de prueba del
  naive estacional de 0 bajo el ajuste antiguo de solo entrenamiento; la
  cobertura v2 es 1.00 para cada modelo. El cambio es el protocolo de historial
  de ajuste, no una mejora artificial del modelo.
- Limitaciones y advertencias directamente del reporte (el visor las muestra
  traducidas al español; el JSON congelado nunca se modifica), incluido el
  prefijo acotado, el historial parcial del producto 548 y la ausencia de
  cualquier política de inventario o modelo avanzado.
- Procedencia y frontera: el conjunto de datos, su revisión fijada, el ID de
  evaluación, el rango observado y las claves técnicas exactas viven en
  «Detalles técnicos y procedencia»; el visor es un benchmark acotado, no un
  modelo de producción ni una política de inventario.

El visor es responsivo (rejillas colapsadas a 1024 px y 640 px, tabla con
scroll horizontal contenido), respeta `prefers-reduced-motion` (anula las
animaciones), tiene foco visible para navegación por teclado, enlaces de salto
al contenido y equivalente textual accesible del gráfico. El sistema visual
comprometido (superficies, tokens, tipografía, decisiones de layout) está
documentado en [DESIGN.md](DESIGN.md).

El visor falla cerrado: un reporte faltante o malformado muestra un error
visible y claro en lugar de sustituir valores que podrían falsear el benchmark.

**Resultados derivados del reporte (v2, 77 observaciones de prueba por
partición):**

| Partición | Modelo | MAE | RMSE | WMAPE | Sesgo | Cobertura | Disponibles | Solicitadas | Evaluadas | Filas con quiebre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Partición 1 | Último valor | 0.574 | 0.918 | 0.581 | 0.039 | 1.00 | 77 | 77 | 77 | 34 |
| Partición 1 | Naive estacional | 0.586 | 0.935 | 0.593 | 0.147 | 1.00 | 77 | 77 | 77 | 34 |
| Partición 1 | Promedio móvil | 0.557 | 0.819 | 0.564 | 0.147 | 1.00 | 77 | 77 | 77 | 34 |
| Partición 1 | Suavizamiento exponencial | 0.538 | 0.771 | 0.545 | 0.137 | 1.00 | 77 | 77 | 77 | 34 |
| Partición 2 | Último valor | 0.610 | 0.874 | 0.570 | 0.021 | 1.00 | 77 | 77 | 77 | 33 |
| Partición 2 | Naive estacional | 0.518 | 0.749 | 0.484 | −0.110 | 1.00 | 77 | 77 | 77 | 33 |
| Partición 2 | Promedio móvil | 0.481 | 0.645 | 0.449 | −0.110 | 1.00 | 77 | 77 | 77 | 33 |
| Partición 2 | Suavizamiento exponencial | 0.515 | 0.662 | 0.481 | −0.046 | 1.00 | 77 | 77 | 77 | 33 |

**Interpretación prudente.** Por partición y métrica los mínimos difieren —
el suavizamiento exponencial es el más bajo en MAE/WMAPE/RMSE en la
Partición 1 y el promedio móvil en la Partición 2 — y ningún método gana
todas las métricas. Este benchmark acotado no selecciona automáticamente un
ganador; una muestra de 1.000 filas, 12 productos y 2 particiones es evidencia
sobre el comportamiento de las líneas base en este prefijo, no un ranking
general.

**Detalles técnicos y procedencia** (claves exactas, separadas de la lectura
pública):

- Reporte congelado (solo lectura, ruta fija):
  `data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json`.
- Conjunto de datos: `Dingdong-Inc/FreshRetailNet-50K`; revisión:
  `08c1fab7f9257bc73679d415d65d644165d351d4`; ID de evaluación:
  `freshretailnet-50k-bounded-real-1000-v2`.
- Protocolo: `baseline-evaluation-v2`; partición evaluada `test`;
  `fit_partition` = `train+validation`; `test_excluded_from_fit` = `true`;
  `baseline_parameters_fixed` = `true`.
- Claves de partición: `real-fold-1`, `real-fold-2`.
- Claves de modelo: `naive`, `seasonal_naive`, `moving_average`, `ses`;
  ventana de promedio móvil 7 días; alpha de suavizamiento exponencial 0.3.
- Objetivo evaluado: `observed_sales`.
- v1 de diagnóstico (nunca se carga):
  `freshretailnet-50k-bounded-real-1000.json`.

**Inicio rápido:**

```bash
uv sync --extra demo
uv run streamlit run app/streamlit_app.py
```

El visor está cubierto por `tests/test_benchmark_report.py` (lector puro del
reporte) y `tests/test_streamlit_app.py` (AppTest de Streamlit: arranque,
interacción del filtro de partición y del selector de métrica, marcadores de
mínimo, equivalente accesible, secciones nuevas y la oración exacta de
no-ganador).

### Validación local

```bash
uv run pytest -q
uv run python -m compileall -q src/
git diff --check
uv lock --check
```

El visor de demo agrega sus propias comprobaciones:

```bash
uv sync --extra demo          # instala streamlit (extra opcional)
uv run pytest -q tests/test_benchmark_report.py tests/test_streamlit_app.py
uv run streamlit run app/streamlit_app.py   # el arranque real en modo headless se verifica con --server.headless true
```

## Roadmap

- [x] Auditar las licencias de los conjuntos de datos públicos candidatos (completa: FreshRetailNet-50K, CC BY 4.0 — ver docs/source-contract.md).
- [x] Seleccionar el conjunto de datos (FreshRetailNet-50K) y definir el esquema canónico.
- [x] Crear el primer adaptador de ingesta y los fixtures offline de prueba pequeños.
- [x] Implementar el cargador de streaming acotado (pruebas offline y prueba de humo de 1.000 filas superadas).
- [x] Ejecutar la prueba de humo de streaming en vivo acotada (exactamente 1.000 filas; sin datos en vivo persistidos).
- [x] Implementar las divisiones temporales y el arnés de validación.
- [x] Implementar y evaluar líneas base simples.
- [x] Implementar el ejecutor de evaluación de líneas base offline reproducible (configs tipadas, particiones, métricas, JSON).
- [x] Ejecutar la evaluación real de diagnóstico acotada (1.000 filas; reporte sin datos crudos; ajuste antiguo solo-train).
- [x] Corregir el protocolo de evaluación (`baseline-evaluation-v2`: la validación ajusta solo train; la prueba ajusta train+validation ordenados; la prueba nunca se ajusta; parámetros fijos).
- [x] Ejecutar un benchmark real de líneas base acotado corregido (1.000 filas; reporte v2 sin datos crudos).
- [x] Construir el visor de benchmark acotado de solo lectura (extra `demo` de Streamlit; solo agregados; cubierto por AppTest; captura de pantalla en docs/assets).
- [ ] Evaluar líneas base sobre una instantánea real fijada completa vía el cargador de streaming.
- [ ] Implementar el simulador de políticas de inventario y el modelo de costos.
- [ ] Agregar comprobaciones de calidad de datos y pruebas de rutas de falla.
- [ ] Construir la demo interactiva de escenarios (vista de pronóstico, simulador de políticas, frontera de compensaciones).
- [ ] Agregar un modelo avanzado solo después de reportar las líneas base.
- [ ] Agregar optimización de políticas bajo restricciones.
- [ ] Desplegar la demo o publicar un recorrido grabado.
- [ ] Escribir el caso de estudio técnico final y la publicación de LinkedIn.

## Definición de hecho

El proyecto está listo para publicación de portafolio cuando:

- Un nuevo colaborador pueda ejecutarlo a partir de las instrucciones documentadas.
- La demo responda una pregunta clara de inventario con evidencia trazable.
- Los pronósticos se evalúen en divisiones temporales contra líneas base simples.
- La compensación de negocio (nivel de servicio versus costo) sea visible y ajustable.
- Las pruebas y el CI pasen en la revisión revisada.
- El README explique la arquitectura, las compensaciones, las limitaciones y los próximos pasos.
- Un reclutador pueda entender el problema, el resultado y el enlace de la demo en menos de dos minutos.

## Licencia

El código original de este repositorio se publica bajo la licencia MIT (ver
`LICENSE`). Los datos, capturas raw, manifests, fixtures y demás contenido de
terceros no quedan cubiertos por MIT: conservan sus términos independientes
documentados — FreshRetailNet-50K es CC BY 4.0 (ver `docs/source-contract.md`) —
y no se otorga ningún derecho implícito sobre material de terceros.
