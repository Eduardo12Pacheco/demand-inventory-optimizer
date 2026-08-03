# Sistema visual del visor de benchmark acotado

> Documento de diseño del visor de Streamlit de solo lectura
> (`app/streamlit_app.py`). Describe el sistema visual **comprometido en el
> código** de este hito: tokens, layout, decisiones de gráficos, accesibilidad
> y las restricciones de verdad de los datos que el diseño no puede violar.
> Es un documento de diseño editorial-científico: los datos congelados mandan
> y el diseño existe para hacerlos legibles, nunca para adornarlos.

## 1. Lectura del proyecto

**Lectura del diseño:** instrumento científico de lectura para un revisor técnico,
con un lenguaje editorial-científico sobrio: superficies de blanco cálido, tinta
verde-neutra, un único acento verde azulado apagado, datos en monoespaciada,
líneas finas en lugar de tarjetas y movimiento mínimo, solo cuando explica estado.

| Dial | Valor | Por qué |
| --- | --- | --- |
| VARIANCE | 3 | Simetría de grillas y alineaciones estrictas; el contenido es el protagonista |
| MOTION | 2 | Un único momento de entrada por componente; anulado bajo `prefers-reduced-motion` |
| DENSITY | 6 | Datos densos, pero con aire: las secciones respiran con espacio y reglas finas |

## 2. Tokens (paleta y tipografía)

Tokens declarados en `:root` dentro de `app/streamlit_app.py` (la única fuente
de verdad visual; no hay hojas externas):

| Token | Valor | Uso |
| --- | --- | --- |
| `--paper` | `#F6F4EE` | Fondo de la aplicación |
| `--paper-2` | `#FBFAF6` | Superficies elevadas: hechos, tarjetas, tablas, bloques de protocolo |
| `--ink` | `#1E2622` | Texto principal |
| `--ink-2` | `#55605A` | Texto secundario, etiquetas y leyendas |
| `--line` | `#DCD7CA` | Reglas de sección y bordes |
| `--line-soft` | `#F0ECDF` | Hairlines internos |
| `--accent` | `#2E6B5E` | Único acento: marcadores de mínimo, foco, hallazgo |
| `--accent-soft` | `#E4ECE8` | Fondo del hallazgo de no-ganador |
| `--mono` | pila de monoespaciadas del sistema | Todos los números y valores de datos |

Reglas de uso:

- **Un solo acento.** El verde azulado apagado (`--accent`) se usa solo para:
  el marcador «menor valor», el hallazgo de no-ganador, los nombres de
  partición en el protocolo, las barras mínimas del gráfico y el foco visible.
  Sin degradados, sin sombras decorativas, sin resplandores.
- **Datos siempre en mono.** Cada número público (conteos, métricas, fechas,
  porcentajes) se renderiza con `--mono`; la tipografía de lectura es la pila
  sans del sistema.
- **Tarjetas solo para hechos y conclusiones.** El resto de la página usa
  líneas finas (`--line-soft`) y espacio: las limitaciones, observaciones y
  detalles nunca se encierran en cajas.

## 3. Layout y secciones

Contenedor centrado con ancho máximo de 1180 px; orden de secciones fijo:

1. **Salto de contenido + navegación de anclas** (`.skip-link`, `.viewer-nav`):
   sticky, una línea, enlaces a las siete secciones.
2. **Resumen** (`.summary`): chips de modo («Solo lectura · Solo agregados»),
   título, subtítulo con la escala (1.000 filas · 12 productos · 2 particiones
   · 4 líneas base), grilla de hechos (`.facts`), hallazgo de no-ganador
   (`.finding`) y tarjetas de conclusión por partición (`.conclusions`).
3. **Protocolo temporal** (`.proto`): un carril proporcional por partición
   (entrenamiento → validación → prueba con anchos según días de calendario),
   leyenda con fechas y conteos, y el carril de historial de ajuste
   (entrenamiento + validación; la prueba nunca entra).
4. **Resultados por partición y modelo**: dos selectores (filtro de partición,
   métrica), gráfico de comparación por partición (`.chart`) y tabla estática
   de agregados en el orden fijo del reporte.
5. **Observaciones** (`.obs-list`): declaraciones calculadas del reporte.
6. **Quiebres de stock** (`.stockout-section`): hechos + bandas de proporción
   (`.bands`).
7. **Metodología** (`.methodology-section`): flujo del pipeline (`.flow`) y
   bloques por línea base (`.model-grid`).
8. **Limitaciones** (`.limit-grid`) y **advertencias** (`.warnings-list`).
9. **Nota histórica** (`.hist-note`), **detalles técnicos y procedencia** y pie
   de página con la oración exacta de no-ganador.

## 4. Decisión de gráficos: HTML/CSS nativo

El gráfico de comparación se construye **solo con HTML/CSS/SVG propios** (divs
de barras con `width` proporcional al mayor valor de cada partición, valores de
3 decimales junto a cada barra). Es una decisión comprometida: **no se agregan
dependencias de gráficos** (nada de Plotly/Altair/Matplotlib). Razones:

- El visor debe seguir siendo auditable y ejecutable con el extra `demo` mínimo.
- Ocho filas de datos no justifican una biblioteca; el renderizado propio es
  determinista y verificable por las pruebas (AppTest inspecciona el markup).
- El equivalente textual accesible (`.sr-only`) repite cada valor y cada mínimo
  para lectores de pantalla.

Reglas del gráfico: un marcador «menor valor» por partición y métrica, pegado
al valor exacto; longitud relativa al máximo de la partición; ambos rangos de
prueba con fechas y conteos en los encabezados de columna; la barra mínima usa
`--accent` y el resto un gris cálido (`#C9C4B4`).

## 5. Responsive

| Ancho | Cambios |
| --- | --- |
| ≤ 1024 px | Conclusiones a 1 columna; procedencia a 1 columna; filas de barra más angostas |
| ≤ 640 px | Hechos a 2 columnas; gráfico, bloques de modelo, limitaciones y leyenda del protocolo a 1 columna; filas de barra y bandas a rejilla compacta; flujo vertical con flechas rotadas |

La tabla estática mantiene su scroll horizontal dentro del contenedor de
Streamlit (nunca desborda la página). La tabla oculta equivalente usa
`table-layout: fixed` y ancho de 1 px para que su contenido `nowrap` no genere
overflow horizontal en viewports pequeños.

## 6. Movimiento

Un único momento por componente, solo entrada (`viewer-rise`, `chart-in`,
`bar-grow`: opacidad + desplazamiento o ancho, 0.45–0.55 s, con delays
escalonados en el resumen). Sin bucles, sin parallax, sin scroll-jacking.

`@media (prefers-reduced-motion: reduce)` anula **todas** las animaciones y
transiciones (`animation: none !important`), incluido `scroll-behavior`.
Verificado en navegador real con emulación de `prefers-reduced-motion`.

## 7. Accesibilidad

- **Foco visible:** `outline: 2px solid var(--accent)` con `outline-offset: 2px`
  en enlaces, botones, selectores y tabla bajo `:focus-visible`.
- **Equivalente textual del gráfico:** tabla `.sr-only` con aria-label, una fila
  por (partición, modelo) más el mínimo de cada partición.
- **ARIA:** el diagrama de protocolo expone `role="group"` y `aria-label` con el
  detalle completo por carril (`role="img"`); las bandas de quiebre marcan el
  relleno como decorativo y la proporción se repite como texto.
- **Salto de contenido:** enlace «Saltar al contenido» al inicio del DOM.
- **Contraste:** tinta `#1E2622` sobre `#F6F4EE` (AA holgado); el acento
  `#2E6B5E` con blanco se usa solo para textos grandes o etiquetas.

## 8. Restricciones de verdad de los datos (datos congelados)

El sistema visual no puede violar estas restricciones; son contratos, no
sugerencias:

- **Nunca se muestran valores hardcodeados:** todo número proviene del reporte
  congelado en tiempo de ejecución; `None` se muestra como «—», nunca se
  inventa.
- **El JSON congelado nunca se modifica:** la prosa en inglés del reporte se
  traduce en la capa de presentación (`REPORT_TEXT_TRANSLATIONS`); si falta una
  traducción, el visor falla cerrado.
- **Sin ganador:** el diseño marca el *menor valor por partición y métrica*,
  nunca un ganador global; la oración exacta de no-ganador aparece en el
  resumen y en el pie.
- **Solo agregados:** nunca se pintan filas crudas, vectores horarios ni
  predicciones individuales.
- **Sin gráficos que sugieran precisión falsa:** las barras son relativas al
  mayor valor de la partición, los valores exactos de 3 decimales van junto a
  cada barra y la leyenda explica la regla de longitud.
- **El reporte v1 de diagnóstico nunca se carga ni se muestra como resultado;**
  solo se menciona en la nota histórica y en los detalles técnicos.
- **Frontera benchmark vs. producción:** el diseño distingue siempre el visor
  de benchmark acotado de un modelo de producción o una política de inventario;
  ninguna sección lo presenta como recomendación operativa.
