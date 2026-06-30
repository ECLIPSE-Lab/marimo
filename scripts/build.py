#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse
import html
from typing import List
from pathlib import Path


# ---------------------------------------------------------------------------
# Gallery metadata
# ---------------------------------------------------------------------------
# Per-app card copy, keyed by file stem. Apps missing here fall back to a
# titleized stem with no description/tags, so the build never breaks when a new
# notebook is added.
APPS = {
    "ptychographic_ctf": {
        "title": "Ptychographic CTF",
        "description": "Contrast-transfer function for ptychographic phase imaging.",
        "tags": ["ptychography", "CTF"],
        "order": 10,
    },
    "ptychographic_sampling": {
        "title": "Ptychographic Sampling",
        "description": "Real- vs reciprocal-space sampling trade-offs in 4D-STEM.",
        "tags": ["ptychography", "sampling"],
        "order": 20,
    },
    "ptychographic_coherence": {
        "title": "Ptychographic Coherence",
        "description": "Partial-coherence and finite-source effects on reconstruction.",
        "tags": ["ptychography", "coherence"],
        "order": 30,
    },
    "ff_stem_ssnr": {
        "title": "FF-STEM SSNR",
        "description": "Spectral SNR / DQE: ptychography vs ADF vs full-field STEM.",
        "tags": ["STEM", "SSNR", "DQE"],
        "order": 40,
    },
    "probe": {
        "title": "STEM Probe",
        "description": "Interactive probe former — aperture, defocus and spherical aberration.",
        "tags": ["probe", "aberrations"],
        "order": 50,
    },
}

# Repo-root directory holding committed thumbnails (one `<stem>.png` per app).
THUMBNAILS_DIR = Path("thumbnails")

SITE_TITLE = "Pelz Lab"
SITE_TAGLINE = "Interactive ptychography &amp; electron-microscopy tools"
LAB_URL = "https://pelzlab.science"
REPO_URL = "https://github.com/ECLIPSE-Lab/marimo"


def is_marimo_notebook(path: Path) -> bool:
    """True only for files that define a marimo app.

    The build sweeps ``notebooks/`` and ``apps/`` recursively, which also picks up
    helper modules (e.g. ``apps/ctf/utils.py``). Those are imported by notebooks, not
    standalone apps, so anything without a ``marimo.App(...)`` definition is excluded
    from the gallery.
    """
    try:
        return "marimo.App(" in path.read_text(encoding="utf-8")
    except OSError:
        return False


def export_html_wasm(notebook_path: str, output_dir: str, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML format.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    output_path = notebook_path.replace(".py", ".html")

    cmd = ["marimo", "export", "html-wasm"]
    if as_app:
        print(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])
    else:
        print(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])

    try:
        output_file = os.path.join(output_dir, output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        cmd.extend([notebook_path, "-o", output_file])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {notebook_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def copy_thumbnails(output_dir: str) -> None:
    """Copy committed thumbnails into the built site so cards can reference them."""
    if not THUMBNAILS_DIR.is_dir():
        return
    dest = Path(output_dir) / "thumbnails"
    dest.mkdir(parents=True, exist_ok=True)
    for png in THUMBNAILS_DIR.glob("*.png"):
        shutil.copy2(png, dest / png.name)


def _meta_for(stem: str) -> dict:
    """Card metadata for an app stem, with a graceful fallback for unknown apps."""
    meta = APPS.get(stem)
    if meta:
        return meta
    return {
        "title": stem.replace("_", " ").title(),
        "description": "",
        "tags": [],
        "order": 999,
    }


def _card_html(notebook: str) -> str:
    """Render a single gallery card for a notebook path (e.g. ``apps/foo.py``)."""
    stem = Path(notebook).stem
    meta = _meta_for(stem)
    href = html.escape(notebook.replace(".py", ".html"))
    title = html.escape(meta["title"])
    description = html.escape(meta["description"])

    if (THUMBNAILS_DIR / f"{stem}.png").exists():
        media = (
            f'<img src="thumbnails/{stem}.png" alt="{title} preview" loading="lazy" '
            f'class="h-full w-full object-cover transition duration-300 group-hover:scale-105" />'
        )
    else:
        # CSS gradient placeholder with the app's initial — CI never depends on
        # the thumbnail images existing.
        initial = html.escape(title[:1].upper() or "?")
        media = (
            '<div class="flex h-full w-full items-center justify-center '
            'bg-gradient-to-br from-brand/80 to-slate-800">'
            f'<span class="text-6xl font-bold text-white/40">{initial}</span></div>'
        )

    tags = "".join(
        '<span class="rounded-full bg-brand/10 px-2.5 py-0.5 text-xs font-medium '
        f'text-brand">{html.escape(tag)}</span>'
        for tag in meta["tags"]
    )
    tags_block = f'<div class="mt-3 flex flex-wrap gap-1.5">{tags}</div>' if tags else ""
    description_block = (
        f'<p class="mt-1 text-sm leading-relaxed text-slate-600">{description}</p>'
        if description
        else ""
    )

    return f"""        <a href="{href}" class="group flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl hover:border-brand/40">
          <div class="aspect-[4/3] w-full overflow-hidden bg-slate-100">
            {media}
          </div>
          <div class="flex flex-1 flex-col p-5">
            <h3 class="text-lg font-semibold text-slate-900">{title}</h3>
            {description_block}
            {tags_block}
            <span class="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-brand">
              Open app
              <svg class="h-4 w-4 transition group-hover:translate-x-0.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M7.293 14.707a1 1 0 0 1 0-1.414L10.586 10 7.293 6.707a1 1 0 0 1 1.414-1.414l4 4a1 1 0 0 1 0 1.414l-4 4a1 1 0 0 1-1.414 0z" clip-rule="evenodd"/></svg>
            </span>
          </div>
        </a>
"""


def generate_index(all_notebooks: List[str], output_dir: str) -> None:
    """Generate the gallery index.html."""
    print("Generating index.html")

    index_path = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)

    ordered = sorted(
        all_notebooks,
        key=lambda nb: (_meta_for(Path(nb).stem)["order"], Path(nb).stem),
    )
    cards = "\n".join(_card_html(nb) for nb in ordered)

    page = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{SITE_TITLE} · marimo apps</title>
    <meta name="description" content="Interactive ptychography and electron-microscopy tools from the Pelz Lab, running in your browser via WebAssembly." />
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {{ theme: {{ extend: {{ colors: {{ brand: '#1f7a6b' }} }} }} }};
    </script>
    <style>
      html {{ scroll-behavior: smooth; }}
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
      .hero-gradient {{ background:
        radial-gradient(60rem 30rem at 110% -10%, rgba(31,122,107,0.45), transparent 60%),
        radial-gradient(50rem 25rem at -10% 0%, rgba(31,122,107,0.25), transparent 55%),
        linear-gradient(180deg, #0f172a 0%, #111c33 100%); }}
    </style>
  </head>
  <body class="bg-slate-50 text-slate-900">
    <header class="hero-gradient text-white">
      <div class="mx-auto max-w-6xl px-6 py-16 sm:py-20">
        <p class="text-sm font-semibold uppercase tracking-[0.2em] text-brand">{SITE_TITLE}</p>
        <h1 class="mt-3 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">Interactive electron-microscopy apps</h1>
        <p class="mt-4 max-w-2xl text-lg text-slate-300">{SITE_TAGLINE}. Each app runs entirely in your browser via WebAssembly — no install required.</p>
        <div class="mt-8 flex flex-wrap items-center gap-4">
          <a href="#apps" class="rounded-lg bg-brand px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand/90">Browse apps</a>
          <a href="{LAB_URL}" class="text-sm font-semibold text-slate-200 underline-offset-4 hover:underline">pelzlab.science →</a>
        </div>
      </div>
    </header>

    <main id="apps" class="mx-auto max-w-6xl px-6 py-14">
      <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
{cards}
      </div>
    </main>

    <footer class="border-t border-slate-200">
      <div class="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-6 py-8 text-sm text-slate-500 sm:flex-row">
        <p>Built with <a href="https://marimo.io" class="font-medium text-brand hover:underline">marimo</a> · runs in your browser via WebAssembly</p>
        <a href="{REPO_URL}" class="font-medium text-slate-600 hover:text-brand">Source on GitHub</a>
      </div>
    </footer>
  </body>
</html>"""

    try:
        with open(index_path, "w") as f:
            f.write(page)
    except IOError as e:
        print(f"Error generating index.html: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    args = parser.parse_args()

    all_notebooks: List[str] = []
    for directory in ["notebooks", "apps"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        all_notebooks.extend(
            str(path)
            for path in dir_path.rglob("*.py")
            if is_marimo_notebook(path)
        )

    if not all_notebooks:
        print("No notebooks found!")
        return

    # Export notebooks sequentially
    for nb in all_notebooks:
        export_html_wasm(nb, args.output_dir, as_app=nb.startswith("apps/"))

    # Copy committed thumbnails, then generate the gallery index.
    copy_thumbnails(args.output_dir)
    generate_index(all_notebooks, args.output_dir)


if __name__ == "__main__":
    main()
