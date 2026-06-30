#!/usr/bin/env python3
"""Generate gallery thumbnails by screenshotting the built marimo apps.

Run this LOCALLY (not in CI). It builds the site if needed, serves ``_site/``,
loads each app in headless Chromium, waits for the WASM render to settle, and
writes a 4:3 PNG into ``thumbnails/<stem>.png``. Commit the resulting PNGs.

    pip install playwright && python -m playwright install chromium
    python scripts/screenshots.py            # capture all live apps
    python scripts/screenshots.py --only ptychographic_ctf --settle 20

The build references these committed PNGs; missing ones fall back to a CSS
placeholder, so this script is purely a quality upgrade and never required.
"""

import argparse
import functools
import http.server
import socketserver
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "_site"
THUMBNAILS_DIR = ROOT / "thumbnails"


def live_app_stems() -> list[str]:
    """Stems of source files in ``apps/`` that actually define a marimo app."""
    stems = []
    for path in sorted((ROOT / "apps").rglob("*.py")):
        try:
            if "marimo.App(" in path.read_text(encoding="utf-8"):
                stems.append(path.stem)
        except OSError:
            continue
    return stems


def serve(directory: Path, port: int) -> socketserver.TCPServer:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def wait_for_render(page, ready_timeout_ms: int) -> bool:
    """Best-effort wait until the app has rendered real output (an image/plot).

    marimo WASM apps boot pyodide and install packages before producing output,
    so we poll for a rendered <img>/<canvas>/<svg plot> rather than a fixed sleep.
    Returns True if output appeared, False on timeout (caller still screenshots).
    """
    try:
        # Walk shadow roots too — marimo renders plot output inside shadow DOM,
        # so a plain document.querySelectorAll misses the figures.
        page.wait_for_function(
            """() => {
                function* walk(root) {
                    for (const el of root.querySelectorAll('*')) {
                        yield el;
                        if (el.shadowRoot) yield* walk(el.shadowRoot);
                    }
                }
                let plots = 0;
                for (const el of walk(document)) {
                    if (el.tagName === 'IMG' && el.naturalWidth > 80 && el.naturalHeight > 80) plots++;
                    else if (el.tagName === 'CANVAS' && el.width > 80) plots++;
                    else if (el.tagName === 'svg') {
                        const r = el.getBoundingClientRect();
                        if (r.width > 150 && r.height > 150) plots++;
                    }
                }
                return plots >= 1;
            }""",
            timeout=ready_timeout_ms,
        )
        return True
    except Exception:
        return False


def capture(stem: str, base_url: str, args, browser) -> bool:
    url = f"{base_url}/apps/{stem}.html"
    context = browser.new_context(
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=args.scale,
    )
    page = context.new_page()
    print(f"  → {stem}: loading {url}")
    try:
        page.goto(url, wait_until="load", timeout=120_000)
    except Exception as e:
        print(f"     load error: {e}")
        context.close()
        return False

    try:
        page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception:
        pass

    rendered = wait_for_render(page, args.ready_timeout * 1000)
    print(f"     rendered={'yes' if rendered else 'timeout'}; settling {args.settle}s")
    time.sleep(args.settle)

    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAILS_DIR / f"{stem}.png"
    page.screenshot(path=str(out))  # viewport-sized → already args.width:args.height
    print(f"     saved {out} ({out.stat().st_size // 1024} KB)")
    context.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--width", type=int, default=1280, help="viewport width")
    parser.add_argument("--height", type=int, default=960, help="viewport height (4:3 with default width)")
    parser.add_argument("--scale", type=float, default=2.0, help="device scale factor (crispness)")
    parser.add_argument("--settle", type=float, default=12.0, help="seconds to wait after render before capture")
    parser.add_argument("--ready-timeout", type=int, default=90, help="max seconds to wait for output to appear")
    parser.add_argument("--only", nargs="*", default=None, help="capture only these stems")
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding the site")
    args = parser.parse_args()

    if not args.no_build:
        print("Building site (python scripts/build.py)…")
        subprocess.run(["python", str(ROOT / "scripts" / "build.py")], cwd=ROOT, check=True)

    stems = args.only or live_app_stems()
    stems = [s for s in stems if (SITE_DIR / "apps" / f"{s}.html").exists()]
    if not stems:
        print("No built apps found to screenshot.")
        return

    httpd = serve(SITE_DIR, args.port)
    base_url = f"http://127.0.0.1:{args.port}"
    print(f"Serving {SITE_DIR} at {base_url}")
    print(f"Capturing {len(stems)} app(s): {', '.join(stems)}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for stem in stems:
                capture(stem, base_url, args, browser)
            browser.close()
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
