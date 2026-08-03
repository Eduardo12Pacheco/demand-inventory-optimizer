#!/usr/bin/env python3
"""Render docs/assets/p3-baseline-comparison.svg from committed evaluation data.

Deterministic, stdlib-only generator. Reads ONLY the committed bounded-real
evaluation report data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json
and writes the SVG byte-identically for identical input. No network, no
wall-clock fields.

The chart shows per-fold baseline metrics (MAE/RMSE/WMAPE) without an
aggregate winner: SES wins real-fold-1 and moving_average wins real-fold-2 on
every metric. No automatic best-model selection was performed.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json"
OUTPUT = REPO_ROOT / "docs" / "assets" / "p3-baseline-comparison.svg"

WIDTH, HEIGHT = 1100, 720

MODELS = ["naive", "seasonal_naive", "moving_average", "ses"]
METRICS = ["mae", "rmse", "wmape"]
METRIC_LABELS = {"mae": "MAE (menor = mejor)", "rmse": "RMSE (menor = mejor)", "wmape": "WMAPE (menor = mejor)"}
FOLD_LABELS = {"real-fold-1": "pliegue 1", "real-fold-2": "pliegue 2"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def fmt3(value: float) -> str:
    """Spanish decimal-comma formatting with 3 decimals (matches the viewer)."""
    return f"{value:.3f}".replace(".", ",")


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    results = report["results"]
    config = report["configuration"]

    by_fold = {}
    for r in results:
        by_fold.setdefault(r["fold"], {})[r["model"]] = r

    # Panel geometry.
    plot_x0, plot_x1 = 150, 1050
    group_w = (plot_x1 - plot_x0) / len(MODELS)
    bar_w = 66
    panels = [
        (92, 222, "mae"),
        (262, 392, "rmse"),
        (432, 562, "wmape"),
    ]

    def winner(fold: str, metric: str) -> str:
        return min(MODELS, key=lambda m: by_fold[fold][m][metric])

    fold1_winner = winner("real-fold-1", "mae")
    fold2_winner = winner("real-fold-2", "mae")
    assert fold1_winner == "ses" and fold2_winner == "moving_average"
    for metric in METRICS:
        assert winner("real-fold-1", metric) == "ses"
        assert winner("real-fold-2", metric) == "moving_average"

    panels_svg = []
    for top, base, metric in panels:
        panel_h = base - top
        values = [by_fold[f][m][metric] for f in ("real-fold-1", "real-fold-2") for m in MODELS]
        vmax = max(values)

        grid = ""
        for frac, label in ((0.0, "0"), (0.5, fmt3(vmax / 2)), (1.0, fmt3(vmax))):
            y = base - frac * (panel_h - 40)
            grid += (
                f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" '
                f'stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,3"/>'
                f'<text x="{plot_x0 - 8}" y="{y + 3:.1f}" font-family="sans-serif" font-size="9" '
                f'fill="#6b7280" text-anchor="end">{label}</text>'
            )

        bars = ""
        for gi, model in enumerate(MODELS):
            gx = plot_x0 + gi * group_w
            for fold, color in (("real-fold-1", "#2563eb"), ("real-fold-2", "#93c5fd")):
                v = by_fold[fold][model][metric]
                h = v / vmax * (panel_h - 40)
                x = gx + (group_w - 2 * bar_w - 18) / 2 + (0 if fold == "real-fold-1" else bar_w + 18)
                y = base - h
                is_winner = winner(fold, metric) == model
                bars += (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}"'
                    + (' stroke="#1e40af" stroke-width="1.5"' if is_winner else "")
                    + f'><title>{model} · {FOLD_LABELS[fold]}: {metric.upper()} {v:.4f}</title></rect>'
                )
                bars += (
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" font-family="sans-serif" '
                    f'font-size="8.5" fill="#374151" text-anchor="middle">{fmt3(v)}</text>'
                )
                if is_winner:
                    bars += (
                        f'<text x="{x + bar_w / 2:.1f}" y="{base + 14}" font-family="sans-serif" '
                        f'font-size="9" font-weight="bold" fill="#1e40af" text-anchor="middle">menor</text>'
                    )
            bars += (
                f'<text x="{gx + group_w / 2:.1f}" y="{base + 30}" font-family="sans-serif" '
                f'font-size="10.5" fill="#111827" text-anchor="middle">{model}</text>'
            )

        legend = (
            f'<rect x="{plot_x0}" y="{top - 33}" width="11" height="11" fill="#2563eb"/>'
            f'<text x="{plot_x0 + 15}" y="{top - 24}" font-family="sans-serif" font-size="10" fill="#374151">pliegue 1 (test 2024-06-05..06-11)</text>'
            f'<rect x="{plot_x0 + 250}" y="{top - 33}" width="11" height="11" fill="#93c5fd"/>'
            f'<text x="{plot_x0 + 265}" y="{top - 24}" font-family="sans-serif" font-size="10" fill="#374151">pliegue 2 (test 2024-06-19..06-25)</text>'
            f'<text x="{plot_x1}" y="{top - 24}" font-family="sans-serif" font-size="11" font-weight="bold" fill="#111827" text-anchor="end">{METRIC_LABELS[metric]}</text>'
        )
        panels_svg.append(grid + bars + legend)

    conclusion_y = 622
    conclusion = (
        f'<rect x="40" y="{conclusion_y - 22}" width="1020" height="44" fill="#eff6ff" stroke="#2563eb" stroke-width="1"/>'
        f'<text x="60" y="{conclusion_y}" font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#111827">Pliegue 1: gana SES (alpha 0,3) · Pliegue 2: gana media movil (ventana 7) — en las tres metricas</text>'
        f'<text x="60" y="{conclusion_y + 18}" font-family="sans-serif" font-size="11" fill="#374151">Sin ganador universal: el benchmark acotado NO selecciona automaticamente un modelo (warning del propio reporte) y no existe modelo de produccion ni politica de inventario.</text>'
    )

    footer = [
        f"evaluation_id: {report['evaluation_id']} | protocol: {report['evaluation_protocol_version']} | partition: test",
        (
            f"dataset: {report['dataset']} (CC BY 4.0, ver docs/source-contract.md) | "
            f"revision: {report['dataset_revision']}"
        ),
        (
            f"bounded prefix: {report['rows_loaded']} filas transmitidas | {report['coverage']['products']} productos | "
            f"2 pliegues | 77 filas de prueba por pliegue | fit: train+validation"
        ),
        (
            f"parametros fijos: moving_average_window {config['moving_average_window']}, "
            f"ses_alpha {config['ses_alpha']} | stockout_rows: {report['coverage']['stockout_rows']}"
        ),
        "warnings: benchmark acotado (primer prefijo, no snapshot completo); observed_sales != demanda latente; censura por quiebre puede afectar la interpretacion",
    ]
    footer_lines = "".join(
        f'<text x="20" y="{HEIGHT - 64 + i * 14}" font-family="monospace" font-size="9.5" fill="#374151">{esc(line)}</text>'
        for i, line in enumerate(footer)
    )

    metadata = {
        "generator": "scripts/render_p3_baseline_comparison.py",
        "input_file": "data/evaluations/freshretailnet-50k-bounded-real-1000-v2.json",
        "evaluation_id": report["evaluation_id"],
        "evaluation_mode": report["evaluation_mode"],
        "winner_fold_1": {"model": "ses", "metric_values": by_fold["real-fold-1"]["ses"]},
        "winner_fold_2": {"model": "moving_average", "metric_values": by_fold["real-fold-2"]["moving_average"]},
        "no_aggregate_winner": True,
        "no_automatic_selection": True,
        "deterministic": True,
    }

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/><title>Benchmark acotado de lineas base - FreshRetailNet-50K</title><text x="20" y="24" font-family="sans-serif" font-size="16" font-weight="bold" fill="#111827">Benchmark acotado de lineas base - FreshRetailNet-50K (1.000 filas · 12 productos · 2 pliegues)</text><text x="20" y="42" font-family="sans-serif" font-size="11" fill="#4b5563">target: observed_sales · lineas base: naive, seasonal_naive, moving_average (ventana 7), ses (alpha 0,3) · 77 filas de prueba por pliegue</text>
{panels_svg[0]}
{panels_svg[1]}
{panels_svg[2]}
{conclusion}
{footer_lines}
<metadata>{esc(json.dumps(metadata, ensure_ascii=True, sort_keys=True))}</metadata></svg>"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
