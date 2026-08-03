# Guion de demo — 75 a 90 segundos, siete tiempos

Visor de benchmark acotado de solo lectura (`uv run streamlit run app/streamlit_app.py`).
Público: un revisor técnico. Tono: factual, sobrio, sin promoción. Todos los
números citados son los valores exactos del reporte congelado v2.

---

**Tiempo 1 — Resumen del visor (0–10s)**

> "Este visor presenta un benchmark acotado de líneas base reales de
> FreshRetailNet-50K, v2: solo lectura y solo agregados, desde un único reporte
> JSON local. Nunca toca el conjunto de datos, nunca carga el reporte v1 de
> diagnóstico y nunca hace una solicitud de datos en vivo. El resumen muestra la
> escala exacta: 1.000 filas transmitidas, 12 productos, 2 particiones y
> 4 líneas base, con la advertencia central arriba: este benchmark acotado no
> selecciona automáticamente un ganador."

**Tiempo 2 — Protocolo temporal (10–25s)**

> "El bloque de protocolo muestra por qué los resultados son creíbles: cada
> partición es entrenamiento → validación → prueba, del 2024-03-28 al
> 2024-06-11 y del 2024-03-28 al 2024-06-25. El historial de ajuste de la prueba
> es exactamente entrenamiento más validación — la Partición 1 ajusta 769 filas
> y la Partición 2 ajusta 923 — y las filas de prueba nunca entran al historial
> de ajuste. Los parámetros están fijos: promedio móvil de 7 días y
> suavizamiento exponencial con alpha 0.3, sin ajuste automático."

**Tiempo 3 — Comparación por partición (25–40s)**

> "El selector de métrica cambia el gráfico entre MAE, RMSE y WMAPE; la barra
> resaltada marca el menor valor de cada partición y la tabla equivalente
> accesible repite cada valor. Leyendo la Partición 1, el suavizamiento
> exponencial tiene el MAE más bajo (0.538), el RMSE más bajo (0.771) y el
> WMAPE más bajo (0.545); en la Partición 2, el promedio móvil (0.481, 0.645 y
> 0.449). Hay 77 observaciones de prueba por partición y cobertura 1.00 en las
> ocho filas."

**Tiempo 4 — La corrección naive estacional v1 → v2 (40–50s)**

> "La nota histórica del protocolo: la v1 de diagnóstico registró una cobertura
> de prueba del naive estacional de 0 bajo el ajuste antiguo de solo
> entrenamiento. La v2, con el protocolo corregido, registra cobertura 1.00
> para cada modelo en cada partición. La diferencia es el protocolo de
> historial de ajuste (entrenamiento + validación ordenados), no una mejora
> artificial del modelo."

**Tiempo 5 — Quiebres de stock (50–62s)**

> "La sección de quiebres usa los conteos del reporte: 593 filas con quiebre y
> 708 filas con alguna hora en estado de quiebre sobre las 1.000; en la prueba,
> 34 filas en la Partición 1 y 33 en la Partición 2. Las ventas observadas
> siguen siendo ventas observadas: no hay estimación de demanda latente y los
> quiebres no se corrigen ni se enmascaran."

**Tiempo 6 — Limitaciones (62–75s)**

> "Las limitaciones están impresas en el visor: 1.000 filas, 12 productos, el
> producto 548 con historial parcial. Esto no es el conjunto de datos completo,
> no es una política de inventario y ningún modelo está listo para producción.
> El filtro de partición y la tabla estática de ocho filas muestran que no hay
> ranking: el orden es fijo."

**Tiempo 7 — Cierre: sin ganador universal (75–90s)**

> "El modelo con el menor valor cambia según la partición y la métrica — el
> suavizamiento exponencial en la Partición 1, el promedio móvil en la
> Partición 2 — y ningún método gana todo. Este benchmark acotado no
> selecciona automáticamente un ganador. Para ejecutarlo: `uv sync --extra
> demo` y luego `uv run streamlit run app/streamlit_app.py`. La evidencia
> completa — código, pruebas y reporte — está documentada en el repositorio."

---

Nota: las claves técnicas exactas (particiones `real-fold-1`/`real-fold-2`;
modelos `naive`, `seasonal_naive`, `moving_average`, `ses`; protocolo
`baseline-evaluation-v2`; ruta del reporte) se presentan por separado en la
sección «Detalles técnicos y procedencia» del visor, fuera de la lectura
pública.
