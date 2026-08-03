"""Captura una captura de pantalla del visor de demo para el README.

Uso (con la app corriendo en PORT, o la iniciará):

    uv run --with playwright python scripts/capture_demo_screenshot.py

Requiere un binario de Chromium del sistema (verificado vía --chromium-path)
porque la descarga del navegador de Playwright es intencionalmente una
dependencia que no es del proyecto.

Opciones:

    --output PATH        PNG de destino (por defecto: docs/assets/demo-viewer.png)
    --width N            ancho del viewport (por defecto: 1280)
    --height N           alto del viewport (por defecto: 800)
    --viewport-only      captura solo el viewport inicial en lugar de la
                         página completa (por defecto: captura de página completa)
    --zoom FLOAT         zoom CSS real aplicado antes de la captura (por defecto: 1.0);
                         p. ej. 0.80 ajusta más contenido al viewport
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "assets" / "demo-viewer.png"


def wait_for_app(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"App did not become healthy at {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8599")
    parser.add_argument("--chromium-path", default="/usr/bin/chromium")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--viewport-only",
        action="store_true",
        help="captura solo el viewport inicial (por defecto: captura de página completa)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="zoom CSS real aplicado antes de la captura (por defecto: 1.0)",
    )
    args = parser.parse_args()
    if args.zoom <= 0 or args.zoom != args.zoom:  # no positivo o NaN
        parser.error(f"--zoom must be a positive number, got {args.zoom!r}")

    wait_for_app(args.url)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chromium_path,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("h1", timeout=30000)
        page.wait_for_timeout(2000)  # dejar que el dataframe se renderice
        if args.zoom != 1.0:
            # Zoom CSS real: vuelve a maquetar la página para que quepa más
            # contenido en el viewport; las grillas de canvas se redibujan solas.
            page.evaluate("(z) => { document.body.style.zoom = String(z); }", args.zoom)
            page.wait_for_timeout(800)
        if args.viewport_only:
            height = args.height
        else:
            # Streamlit 1.60 hace scroll dentro de [data-testid="stMain"]
            # (overflow-y: auto), así que la captura de página completa debe
            # dimensionar el viewport a ese contenedor.
            height = page.evaluate(
                "document.querySelector('[data-testid=stMain]')?.scrollHeight "
                "|| document.querySelector('[data-testid=stAppViewContainer]')?.scrollHeight "
                "|| 1000"
            )
            page.set_viewport_size({"width": args.width, "height": height})
            page.wait_for_timeout(500)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output))
        browser.close()
    mode = "viewport" if args.viewport_only else "full page"
    print(f"Screenshot ({mode}) written to {output} ({args.width}x{height}px)")


if __name__ == "__main__":
    main()
