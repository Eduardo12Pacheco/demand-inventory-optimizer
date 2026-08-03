"""Pruebas a nivel AppTest para el visor de benchmark de Streamlit de solo lectura (UI en español).

Estas pruebas requieren el extra opcional ``demo`` (``uv sync --extra demo``);
nunca se omiten en silencio cuando Streamlit no está disponible en el entorno
bajo el que se ejecuta la suite.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from inventory_optimizer.demo.benchmark_report import (
    REPORT_PATH,
    REPORT_RELATIVE_PATH,
    load_report,
)

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP_PATH = APP_DIR / "streamlit_app.py"
EXACT_NO_WINNER_SENTENCE = "Este benchmark acotado no selecciona automáticamente un ganador."
# Nombres públicos en español de los modelos (sin claves técnicas crudas en la UI principal).
MODEL_LABELS = [
    "Último valor",
    "Naive estacional",
    "Promedio móvil",
    "Suavizamiento exponencial",
]
FOLD_LABELS = ["Partición 1", "Partición 2"]
RAW_KEYS = [
    "`real-fold-1`",
    "`real-fold-2`",
    "`naive`",
    "`seasonal_naive`",
    "`moving_average`",
    "`ses`",
]
SPANISH_MAIN_LABELS = [
    "Conjunto de datos",
    "Revisión",
    "ID de evaluación",
    "Filas",
    "Productos",
    "Rango observado",
    "Protocolo temporal",
    "Entrenamiento",
    "Validación",
    "Prueba",
    "Resultados por partición y modelo",
    "Observaciones",
    "Datos de quiebres de stock",
    "Limitaciones",
    "Advertencias del reporte",
    "Detalles técnicos y procedencia",
]
# El reporte v2 congelado nunca debe cambiar de contenido; cualquier edición al JSON rompe
# esta prueba a propósito. Ver docs/evaluation.md para el protocolo de corrección.
FROZEN_REPORT_SHA256 = "005177b09eacdd4a6820ce5585b4a22ab470b27e85f2569fcb2bf450747f82e3"


def make_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


def _all_texts(at: AppTest) -> str:
    """Markdown más captions, como el visor renderiza su copia pública."""
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_app_starts_without_exceptions() -> None:
    at = make_app()
    at.run()
    assert not at.exception, at.exception
    title = " ".join(t.value for t in at.title)
    assert "Visor de benchmark acotado de pronósticos" in title
    texts = _all_texts(at)
    assert "freshretailnet-50k-bounded-real-1000-v2" in texts
    assert "08c1fab7f9257bc73679d415d65d644165d351d4" in texts


def test_spanish_main_labels_are_visible() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    for label in SPANISH_MAIN_LABELS:
        assert label in texts, label


def test_initial_table_shows_all_folds_in_fixed_model_order() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    # Los resultados se renderizan como una tabla estática (no virtualizada) para que
    # cada fila agregada siempre se pinte; no se usa ningún dataframe virtualizado.
    assert not at.dataframe
    assert at.table, "expected the static results table"
    df = at.table[0].value
    assert list(df.columns) == [
        "Partición",
        "Modelo",
        "MAE",
        "RMSE",
        "WMAPE",
        "Sesgo",
        "Cobertura",
        "Disponibles",
        "Solicitadas",
        "Evaluadas",
        "Filas con quiebre",
    ]
    assert len(df) == 8
    assert list(df["Partición"])[:4] == ["Partición 1"] * 4
    assert list(df["Partición"])[4:] == ["Partición 2"] * 4
    assert list(df["Modelo"])[:4] == MODEL_LABELS
    assert list(df["Modelo"])[4:] == MODEL_LABELS
    # Solo columnas agregadas; la cobertura y los conteos vienen del reporte.
    assert list(df["Cobertura"]) == ["1.00"] * 8
    assert list(df["Disponibles"]) == ["77"] * 8
    assert list(df["Solicitadas"]) == ["77"] * 8
    assert list(df["Evaluadas"]) == ["77"] * 8


def test_fold_selector_filters_the_table_in_spanish() -> None:
    at = make_app()
    at.run()
    # Selector de partición (heredado) primero, selector de métrica (nuevo) después.
    assert len(at.selectbox) == 2
    assert at.selectbox[0].label == "Filtrar por partición"
    assert at.selectbox[0].options == ["Todas las particiones", "Partición 1", "Partición 2"]
    assert at.selectbox[1].label == "Métrica para comparar"
    assert at.selectbox[1].options == ["MAE", "RMSE", "WMAPE"]
    at.selectbox[0].select("Partición 2").run()
    assert not at.exception, at.exception
    df = at.table[0].value
    assert len(df) == 4
    assert set(df["Partición"]) == {"Partición 2"}
    at.selectbox[0].select("Partición 1").run()
    assert set(at.table[0].value["Partición"]) == {"Partición 1"}


def test_exact_no_winner_sentence_is_visible() -> None:
    at = make_app()
    at.run()
    texts = _all_texts(at)
    assert EXACT_NO_WINNER_SENTENCE in texts
    assert texts.count(EXACT_NO_WINNER_SENTENCE) == 2  # banner de resultados + pie de página


def test_spanish_observations_are_computed_from_the_report() -> None:
    at = make_app()
    at.run()
    texts = _all_texts(at)
    # Calculadas por partición/métrica desde el JSON, con nombres de partición y modelo en español:
    # el suavizamiento exponencial gana la partición 1, el promedio móvil la partición 2.
    assert "En la Partición 1, el suavizamiento exponencial tiene el MAE más bajo (0.538)." in texts
    assert "En la Partición 1, el suavizamiento exponencial tiene el WMAPE más bajo (0.545)." in texts
    assert "En la Partición 2, el promedio móvil tiene el MAE más bajo (0.481)." in texts
    assert "En la Partición 2, el promedio móvil tiene el WMAPE más bajo (0.449)." in texts
    assert "Ningún modelo individual tiene el error más bajo en todas las métricas" in texts
    assert "Líneas base evaluadas: Último valor, Naive estacional, Promedio móvil y Suavizamiento exponencial." in texts


def test_no_raw_fold_or_model_keys_in_primary_ui() -> None:
    """Las claves crudas solo se permiten dentro de la sección etiquetada de detalles técnicos."""
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    primary = texts.split("Detalles técnicos y procedencia")[0]
    for raw in RAW_KEYS:
        assert raw not in primary, raw
    assert "real-fold-1" not in primary
    assert "real-fold-2" not in primary


def test_technical_identifiers_are_isolated_in_details_section() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    assert "Detalles técnicos y procedencia" in texts
    details = texts.split("Detalles técnicos y procedencia")[1]
    for raw in RAW_KEYS:
        assert raw in details, raw
    for identifier in (
        "`baseline-evaluation-v2`",
        "`train+validation`",
        "`test_excluded_from_fit`",
        "`baseline_parameters_fixed`",
        "`inventory_optimizer.bounded_real_evaluation/v1`",
        REPORT_RELATIVE_PATH,
        "08c1fab7f9257bc73679d415d65d644165d351d4",
        "`observed_sales`",
    ):
        assert identifier in details, identifier


def test_no_operar_leer_in_public_copy() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    assert "Operar/Leer" not in texts
    assert "Solo lectura · Solo agregados" in texts


def test_spanish_model_names_appear_in_table_and_observations() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    df = at.table[0].value
    assert list(df["Modelo"])[:4] == MODEL_LABELS
    texts = _all_texts(at)
    # Los nombres en español son las únicas referencias a modelos en la UI principal.
    assert "Suavizamiento exponencial" in texts
    assert "Promedio móvil" in texts
    assert "Naive estacional" in texts
    assert "Último valor" in texts


def test_limitations_and_warnings_render_in_spanish_not_english() -> None:
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    raw = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    # La prosa inglesa congelada nunca debe aparecer textual; el marcador técnico
    # ``bounded_real_evaluation`` solo se permite dentro de su renderización en español.
    english_strings = [
        s for s in (*raw["limitations"], *raw["warnings"])
        if s != "bounded_real_evaluation"
    ]
    for english in english_strings:
        assert english not in texts, english
    for spanish in (
        "La evaluación está acotada al prefijo transmitido, no es una instantánea completa.",
        "El prefijo de entrada puede tener historiales de producto incompletos y fechas de calendario faltantes.",
        "No se incluyen pronósticos avanzados ni recuperación de demanda latente.",
        "Modo de evaluación real acotada.",
        "Este reporte cubre como máximo las primeras 1.000 filas de entrenamiento transmitidas "
        "y no representa el conjunto de datos completo.",
        "Las métricas evalúan únicamente las ventas observadas; no son métricas de demanda latente ni de demanda real.",
        "Las filas con metadatos de quiebre se conservan; la censura puede afectar la interpretación "
        "del error sobre ventas observadas.",
        "No se realizó ninguna selección automática de mejor modelo.",
    ):
        assert spanish in texts, spanish


def test_translation_map_covers_exactly_the_frozen_report_text() -> None:
    """El mapa de la capa de presentación debe cubrir cada limitación/advertencia v2 actual.

    Comprobación AST estática sobre el código de la app: si el reporte congelado gana una
    nueva cadena de prosa, esta prueba falla hasta que el mapa en español la cubra (falla cerrada).
    """
    import ast

    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    mapping = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REPORT_TEXT_TRANSLATIONS"
            for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
    assert mapping is not None, "REPORT_TEXT_TRANSLATIONS missing from app source"
    raw = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    expected = {*raw["limitations"], *raw["warnings"]}
    assert set(mapping) == expected
    assert all(isinstance(value, str) and value for value in mapping.values())


def test_spanish_explanatory_claims_are_negated_where_required() -> None:
    at = make_app()
    at.run()
    texts = _all_texts(at)
    # Afirmaciones de alcance, ventas observadas, quiebres y preparación para producción, en español.
    assert "Alcance limitado: solo 1.000 filas, 12 productos y 2 particiones." in texts
    assert "Las ventas observadas no son demanda real ni estimación de demanda latente recuperada." in texts
    assert "Los quiebres de stock no fueron corregidos ni enmascarados." in texts
    assert "Ningún modelo está listo para producción." in texts
    # Ninguna afirmación prohibida sin negación en ninguna parte.
    for forbidden in ("best model", "production-ready forecast", "optimal inventory policy", "demand recovered"):
        assert forbidden not in texts.lower(), forbidden


def test_app_shows_protocol_and_stockout_facts() -> None:
    at = make_app()
    at.run()
    texts = _all_texts(at)
    for expected in (
        "fit_partition",
        "entrenamiento + validación",
        "las filas de prueba nunca entran al historial de ajuste",
        "769",
        "923",
        "593",
        "708",
        "34",
        "33",
        "baseline-evaluation-v2",
        "no es una política de inventario",
        "test_excluded_from_fit",
        "observed_sales",
    ):
        assert expected in texts, expected


def test_report_derived_metrics_are_preserved_in_the_table() -> None:
    at = make_app()
    at.run()
    df = at.table[0].value
    assert list(df["MAE"]) == ["0.574", "0.586", "0.557", "0.538", "0.610", "0.518", "0.481", "0.515"]
    assert list(df["WMAPE"]) == ["0.581", "0.593", "0.564", "0.545", "0.570", "0.484", "0.449", "0.481"]
    assert list(df["RMSE"]) == ["0.918", "0.935", "0.819", "0.771", "0.874", "0.749", "0.645", "0.662"]
    assert list(df["Filas con quiebre"]) == ["34", "34", "34", "34", "33", "33", "33", "33"]


def test_report_path_stays_v2_and_json_contents_are_unchanged() -> None:
    """La ruta de la app/reporte sigue siendo el archivo v2 congelado y sus bytes nunca cambian."""
    assert REPORT_RELATIVE_PATH == "data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json"
    assert REPORT_PATH.name == "freshretailnet-50k-bounded-real-1000-v2.json"
    assert REPORT_PATH.is_file()
    raw = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert raw["evaluation_id"] == "freshretailnet-50k-bounded-real-1000-v2"
    assert raw["schema"] == "inventory_optimizer.bounded_real_evaluation/v1"
    assert hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest() == FROZEN_REPORT_SHA256
    # La app renderiza exactamente esa ruta de reporte (visible, relativa al repositorio,
    # dentro de la sección de detalles técnicos).
    at = make_app()
    at.run()
    assert not at.exception
    texts = _all_texts(at)
    assert REPORT_RELATIVE_PATH in texts


def test_screenshot_script_exposes_required_options() -> None:
    """La utilidad de captura soporta ruta de salida, control de viewport/página completa y zoom."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "capture_demo_screenshot.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--output", "--width", "--height", "--viewport-only", "--zoom"):
        assert option in result.stdout, option


def test_screenshot_script_rejects_invalid_zoom() -> None:
    """--zoom debe ser un número positivo finito; los valores inválidos fallan rápido."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "capture_demo_screenshot.py"
    for bad_value in ("0", "-0.5", "abc"):
        result = subprocess.run(
            [sys.executable, str(script), "--zoom", bad_value],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0, bad_value
        stderr = result.stderr.lower()
        assert "positive" in stderr or "invalid float" in stderr, (bad_value, result.stderr)


DETAILS_HEADING = "Detalles técnicos y procedencia"


def test_spanish_thousands_separators_in_public_counts() -> None:
    """Cada conteo entero público se renderiza con separadores de miles en español.

    Los conteos >= 1000 deben usar ``.`` (1,000 -> 1.000) en el subtítulo, los hechos,
    el protocolo, las columnas de conteo de la tabla, los mosaicos de quiebre, las
    observaciones y las limitaciones; las métricas científicas mantienen su formato
    decimal sin cambios.
    """
    at = make_app()
    at.run()
    assert not at.exception, at.exception
    texts = _all_texts(at)
    # Subtítulo y hechos.
    assert "1.000 filas transmitidas · 12 productos · 2 particiones · 4 líneas base" in texts
    assert "1.000" in texts  # hecho "Filas"
    assert "90 días" in texts  # hecho "Rango observado"
    # Observaciones y limitaciones.
    assert "Alcance limitado: solo 1.000 filas, 12 productos y 2 particiones." in texts
    assert "Acotado a 1.000 filas transmitidas, 12 productos y 2 particiones." in texts
    # Conteos de filas/ajuste del protocolo (ya por debajo de 1,000, deben pasar igual por el formateador).
    assert "769 filas" in texts and "923 filas" in texts
    # Hechos de quiebres de stock.
    assert "593" in texts and "708" in texts
    assert "34 / 33" in texts
    # Las columnas de conteo de la tabla agregada son cadenas de visualización con separadores en español.
    df = at.table[0].value
    assert list(df["Disponibles"]) == ["77"] * 8
    assert list(df["Solicitadas"]) == ["77"] * 8
    assert list(df["Evaluadas"]) == ["77"] * 8
    # Sin comas-miles en ninguna parte de la copia renderizada.
    assert "1,000" not in texts


def test_no_english_streaming_or_snapshot_in_public_copy() -> None:
    """La copia pública usa español natural: 'filas transmitidas', 'instantánea completa'."""
    at = make_app()
    at.run()
    assert not at.exception, at.exception
    texts = _all_texts(at)
    assert "filas transmitidas" in texts
    assert "instantánea completa" in texts
    assert "snapshot" not in texts
    assert "streaming" not in texts


def test_raw_identifiers_absent_before_details_and_present_in_details() -> None:
    """Los identificadores técnicos crudos nunca aparecen antes del encabezado de detalles.

    ``observed_sales`` y ``bounded_real_evaluation`` pasan a una redacción pública en
    español en la UI principal y cada identificador exacto se conserva solo dentro de
    la sección 'Detalles técnicos y procedencia'.
    """
    at = make_app()
    at.run()
    assert not at.exception, at.exception
    texts = _all_texts(at)
    assert DETAILS_HEADING in texts
    primary = texts.split(DETAILS_HEADING)[0]
    assert "observed_sales" not in primary
    assert "bounded_real_evaluation" not in primary
    details = texts.split(DETAILS_HEADING)[1]
    assert "`observed_sales`" in details
    assert "`inventory_optimizer.bounded_real_evaluation/v1`" in details


# ---------------------------------------------------------------------------
# Capa derivada (app/viewer_helpers.py): datos calculados, verificables sin UI.
# ---------------------------------------------------------------------------


def _load_viewer_helpers():
    spec = importlib.util.spec_from_file_location("viewer_helpers", APP_DIR / "viewer_helpers.py")
    assert spec is not None and spec.loader is not None, "viewer_helpers.py no encontrado"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(at: AppTest, marker: str) -> str:
    """Devuelve el markdown crudo del bloque HTML que contiene el marcador."""
    for m in at.markdown:
        if marker in m.value:
            return m.value
    raise AssertionError(f"bloque de markdown con {marker!r} no encontrado")


def test_viewer_helpers_comparison_data_derives_min_per_partition() -> None:
    helpers = _load_viewer_helpers()
    report = load_report()
    rows = helpers.comparison_data(report, "MAE")
    assert len(rows) == 8
    assert {r["model_label"] for r in rows} == set(MODEL_LABELS)
    for fold_label in FOLD_LABELS:
        part = [r for r in rows if r["fold_label"] == fold_label]
        assert len(part) == 4
        assert sum(r["is_min"] for r in part) == 1
        expected_min = min(r["value"] for r in part)
        min_row = next(r for r in part if r["is_min"])
        assert min_row["value"] == expected_min
        assert min_row["formatted"] == f"{expected_min:.3f}"
    for r in rows:
        table_row = next(
            x for x in report.results_table()
            if x["fold"] == r["fold"] and x["model"] == r["model"]
        )
        assert r["value"] == table_row["mae"]


def test_viewer_helpers_conclusion_summary_derived_from_report() -> None:
    helpers = _load_viewer_helpers()
    report = load_report()
    summary = helpers.conclusion_summary(report)
    assert len(summary["partitions"]) == 2
    assert summary["has_universal_winner"] is False
    for index, part in enumerate(summary["partitions"]):
        assert part["fold_label"] == FOLD_LABELS[index]
        fold_key = report.folds[index].fold
        for metric in ("mae", "rmse", "wmape"):
            model_label, value = part["per_metric"][metric]
            table_rows = report.results_table(fold=fold_key)
            expected = min(
                (r for r in table_rows if r[metric] is not None),
                key=lambda r: r[metric],
            )
            assert value == expected[metric]
            assert model_label == MODEL_LABELS[report.models.index(expected["model"])]


def test_viewer_helpers_protocol_segments_widths_sum_to_100() -> None:
    helpers = _load_viewer_helpers()
    report = load_report()
    folds = helpers.protocol_segments(report)
    assert len(folds) == 2
    for fold in folds:
        assert fold["fold_label"] in FOLD_LABELS
        names = [seg["name"] for seg in fold["segments"]]
        assert names == ["Entrenamiento", "Validación", "Prueba"]
        total = round(sum((seg["width_pct"] for seg in fold["segments"]), 0.0), 1)
        assert total == 100.0
        train_val = round(
            fold["segments"][0]["width_pct"] + fold["segments"][1]["width_pct"], 1
        )
        assert fold["fit"]["width_pct"] == train_val
        assert fold["fit"]["rows"] == fold["segments"][0]["rows"] + fold["segments"][1]["rows"]


def test_viewer_helpers_stockout_facts_match_report() -> None:
    helpers = _load_viewer_helpers()
    report = load_report()
    facts = helpers.stockout_facts(report)
    assert facts["stockout_rows"] == report.coverage.stockout_rows
    assert facts["rows_with_any_observed_stockout_status"] == report.coverage.rows_with_any_observed_stockout_status
    assert facts["rows"] == report.rows
    assert facts["test_rows_by_fold"] == {
        "Partición 1": report.fold_test_stockout_rows("real-fold-1"),
        "Partición 2": report.fold_test_stockout_rows("real-fold-2"),
    }


def test_viewer_helpers_model_explanations_use_runtime_parameters() -> None:
    helpers = _load_viewer_helpers()
    report = load_report()
    blocks = helpers.model_explanations(report)
    assert [b["model_label"] for b in blocks] == MODEL_LABELS
    moving = next(b for b in blocks if b["model_label"] == "Promedio móvil")
    ses = next(b for b in blocks if b["model_label"] == "Suavizamiento exponencial")
    assert str(report.moving_average_window) in moving["explanation"]
    assert f"{report.ses_alpha:g}" in ses["explanation"]


# ---------------------------------------------------------------------------
# Rediseño visual: resumen, gráfico de comparación, protocolo, quiebres,
# metodología, limitaciones y CSS de accesibilidad/responsive (nuevos bloques).
# ---------------------------------------------------------------------------


def test_metric_selector_switches_chart_values() -> None:
    at = make_app()
    at.run()
    chart = _block(at, "benchmark-chart")
    assert "0.574" in chart and "0.538" in chart  # MAE por defecto
    assert "0.918" not in chart
    at.selectbox[1].select("RMSE").run()
    assert not at.exception, at.exception
    chart = _block(at, "benchmark-chart")
    assert "0.918" in chart and "0.645" in chart
    assert "0.574" not in chart
    at.selectbox[1].select("WMAPE").run()
    assert not at.exception, at.exception
    chart = _block(at, "benchmark-chart")
    assert "0.581" in chart and "0.449" in chart


def test_chart_partition_columns_and_min_markers() -> None:
    at = make_app()
    at.run()
    chart = _block(at, "benchmark-chart")
    assert chart.count('data-testid="partition-column"') == 2
    assert "Partición 1" in chart and "Partición 2" in chart
    assert "2024-06-05 → 2024-06-11" in chart  # rango de prueba de la Partición 1
    assert "2024-06-19 → 2024-06-25" in chart  # rango de prueba de la Partición 2
    marker_re = re.compile(
        r'class="bar-value">([0-9.]+)</span><span class="min-tag" '
        r'data-testid="min-marker">menor valor</span>'
    )
    # Un mínimo por partición, pegado al valor exacto del reporte (MAE).
    assert marker_re.findall(chart) == ["0.538", "0.481"]


def test_chart_has_accessible_equivalent_table() -> None:
    at = make_app()
    at.run()
    chart = _block(at, "benchmark-chart")
    assert 'data-testid="chart-equivalent"' in chart
    equivalent = chart.split('data-testid="chart-equivalent"')[1]
    assert equivalent.count("<tr") == 9  # 1 encabezado + 8 filas del reporte
    for value in ("0.574", "0.586", "0.557", "0.538", "0.610", "0.518", "0.481", "0.515"):
        assert value in equivalent, value
    assert equivalent.count("menor valor") == 2


def test_summary_has_facts_finding_and_derived_conclusions() -> None:
    at = make_app()
    at.run()
    facts = _block(at, "summary-facts")
    for expected in ("Filas transmitidas", "1.000", "Productos", "12", "Particiones", "2", "Líneas base", "4"):
        assert expected in facts, expected
    finding = _block(at, "no-winner-finding")
    assert EXACT_NO_WINNER_SENTENCE in finding
    conclusions = _block(at, "summary-conclusions")
    assert "Partición 1" in conclusions and "Partición 2" in conclusions
    assert "menor valor" in conclusions
    assert "0.538" in conclusions and "0.481" in conclusions  # mínimos de MAE derivados
    provenance = _block(at, "provenance-strip")
    for expected in (
        "Conjunto de datos",
        "Revisión",
        "ID de evaluación",
        "Rango observado",
        "90 días",
        "08c1fab7f9257bc73679d415d65d644165d351d4",
    ):
        assert expected in provenance, expected


def test_protocol_diagram_markup_and_aria() -> None:
    at = make_app()
    at.run()
    proto = _block(at, "protocol-diagram")
    assert 'role="group"' in proto
    assert 'aria-label="Protocolo temporal' in proto
    for expected in (
        "Entrenamiento",
        "Validación",
        "Prueba",
        "2024-03-28 → 2024-05-28",
        "2024-05-29 → 2024-06-04",
        "2024-06-05 → 2024-06-11",
        "2024-06-12 → 2024-06-18",
        "2024-06-19 → 2024-06-25",
        "692 filas",
        "846 filas",
        "77 filas",
        "769 filas",
        "923 filas",
        "historial de ajuste",
        "entrenamiento + validación",
        "las filas de prueba nunca entran al historial de ajuste",
    ):
        assert expected in proto, expected
    assert proto.count('data-testid="protocol-fold"') == 2


def test_stockout_section_derived_values_and_wording() -> None:
    at = make_app()
    at.run()
    block = _block(at, "stockout-section")
    for expected in (
        "593",
        "708",
        "34 / 33",
        "sin corregir",
        "demanda latente",
        "se conservan",
        "no se corrigen ni se enmascaran",
    ):
        assert expected in block, expected


def test_methodology_flow_and_model_blocks() -> None:
    at = make_app()
    at.run()
    block = _block(at, "methodology-section")
    for step in (
        "Fuente",
        "Transmisión acotada",
        "Particiones temporales",
        "Líneas base",
        "Métricas",
        "Análisis de limitaciones",
    ):
        assert step in block, step
    for label in MODEL_LABELS:
        assert label in block, label
    assert block.count('data-testid="model-block"') == 4
    assert "7" in block  # ventana del promedio móvil (derivada del reporte)
    assert "0.3" in block  # alpha de suavizamiento (derivado del reporte)


def test_limitations_items_cover_supplied_limits() -> None:
    at = make_app()
    at.run()
    block = _block(at, "limitations-section")
    for expected in (
        "1.000 filas transmitidas, 12 productos y 2 particiones",
        "instantánea completa",
        "ventas observadas",
        "demanda latente",
        "no es una política de inventario",
        "Sin modelos avanzados",
        "Sin simulación ni optimización",
        "Sin recomendación operativa",
        "Ningún modelo está listo para producción",
    ):
        assert expected in block, expected


def test_responsive_reduced_motion_and_focus_css() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    for token in (
        "@media (max-width: 1024px)",
        "@media (max-width: 640px)",
        "@media (prefers-reduced-motion: reduce)",
        "focus-visible",
        "sr-only",
    ):
        assert token in source, token


def test_viewer_sources_make_no_network_imports() -> None:
    for path in (APP_PATH, APP_DIR / "viewer_helpers.py"):
        source = path.read_text(encoding="utf-8")
        for token in (
            "import requests",
            "import httpx",
            "import urllib",
            "from urllib",
            "import datasets",
            "load_dataset",
            "import socket",
            "import aiohttp",
            "import boto3",
        ):
            assert token not in source, (path.name, token)
