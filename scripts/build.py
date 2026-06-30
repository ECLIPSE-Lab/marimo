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
# Static assets (logos) copied verbatim into the built site.
PUBLIC_DIR = Path("public")

SITE_TITLE = "ECLIPSE Lab"
SITE_TAGLINE = "Interactive ptychography &amp; electron-microscopy tools"
LAB_URL = "https://pelzlab.science"
REPO_URL = "https://github.com/ECLIPSE-Lab"


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


def copy_public(output_dir: str) -> None:
    """Copy static assets (logos) into the built site for the landing page."""
    if not PUBLIC_DIR.is_dir():
        return
    dest = Path(output_dir) / "public"
    dest.mkdir(parents=True, exist_ok=True)
    for asset in PUBLIC_DIR.iterdir():
        if asset.is_file():
            shutil.copy2(asset, dest / asset.name)


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
        # Apps render on a white canvas — a light mat reads them as figure panels,
        # matching the lab site's treatment of plots on the dark theme.
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
            'bg-gradient-to-br from-brand-blue/40 to-brand-violet/40">'
            f'<span class="font-display text-6xl font-bold text-white/40">{initial}</span></div>'
        )

    tags = "".join(
        '<span class="rounded border border-slate-400/20 bg-slate-400/10 px-2 py-0.5 '
        f'text-[0.72rem] text-slate-300">{html.escape(tag)}</span>'
        for tag in meta["tags"]
    )
    tags_block = f'<div class="mt-3 flex flex-wrap gap-1.5">{tags}</div>' if tags else ""
    description_block = (
        f'<p class="mt-2 text-sm leading-relaxed text-slate-400">{description}</p>'
        if description
        else ""
    )

    return f"""        <a href="{href}" class="card group flex flex-col overflow-hidden rounded-2xl">
          <div class="aspect-[4/3] w-full overflow-hidden bg-slate-50">
            {media}
          </div>
          <div class="flex flex-1 flex-col p-5">
            <h3 class="font-display text-lg font-semibold text-slate-50">{title}</h3>
            {description_block}
            {tags_block}
            <span class="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-sky-300 transition group-hover:text-sky-200">
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
    <meta name="description" content="Interactive ptychography and electron-microscopy tools from the ECLIPSE Lab, running in your browser via WebAssembly." />
    <link rel="icon" type="image/png" href="public/eclipse-mark.png" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet" />
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {{
        theme: {{ extend: {{
          colors: {{ 'brand-blue': '#3b82f6', 'brand-violet': '#8b5cf6' }},
          fontFamily: {{
            sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            display: ['Outfit', 'ui-sans-serif', 'system-ui', 'sans-serif'],
          }},
        }} }},
      }};
    </script>
    <style>
      html {{ scroll-behavior: smooth; }}
      body {{
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
        background-color: #0b0f19;
        background-image:
          radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.08), transparent 25%),
          radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%);
        background-attachment: fixed;
      }}
      .nav-glass {{
        background: rgba(11, 15, 25, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }}
      .card {{
        background: rgba(20, 25, 40, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
      }}
      .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.1);
      }}
      .btn-grad {{
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
        transition: all 0.3s ease;
      }}
      .btn-grad:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }}
      .btn-outline {{ border: 1px solid rgba(255, 255, 255, 0.2); transition: all 0.3s ease; }}
      .btn-outline:hover {{ background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.4); }}
    </style>
  </head>
  <body class="text-slate-400">
    <nav class="nav-glass sticky top-0 z-50">
      <div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <a href="{LAB_URL}" class="flex items-center gap-2.5">
          <img src="public/eclipse-mark.png" alt="" class="h-8 w-8" />
          <span class="font-display text-lg font-semibold text-slate-50">{SITE_TITLE}</span>
        </a>
        <div class="flex items-center gap-5 text-sm font-medium text-slate-300">
          <a href="{LAB_URL}" class="hidden hover:text-sky-300 sm:inline">Lab home</a>
          <a href="{LAB_URL}/research.html" class="hidden hover:text-sky-300 sm:inline">Research</a>
          <a href="{REPO_URL}" class="hover:text-sky-300">GitHub</a>
        </div>
      </div>
    </nav>

    <header class="relative overflow-hidden">
      <div class="mx-auto max-w-6xl px-6 py-16 text-center sm:py-24">
        <h1 class="font-display text-4xl font-bold tracking-tight text-slate-50 sm:text-5xl">Interactive electron-microscopy apps</h1>
        <p class="mx-auto mt-5 max-w-2xl text-lg font-light leading-relaxed text-slate-300">{SITE_TAGLINE}. Each app runs entirely in your browser via WebAssembly — no install required.</p>
        <div class="mt-8 flex flex-wrap items-center justify-center gap-4">
          <a href="#apps" class="btn-grad rounded-lg px-5 py-2.5 text-sm font-semibold text-white">Browse apps</a>
          <a href="{LAB_URL}" class="btn-outline rounded-lg px-5 py-2.5 text-sm font-semibold text-slate-200">Lab home →</a>
        </div>
      </div>
    </header>

    <main id="apps" class="mx-auto max-w-6xl px-6 py-12">
      <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
{cards}
      </div>
    </main>

    <footer style="background:#070a10;border-top:1px solid rgba(255,255,255,0.05)">
      <div class="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-6 py-8 text-sm text-slate-500 sm:flex-row">
        <p>Built with <a href="https://marimo.io" class="font-medium text-sky-300 hover:text-sky-200">marimo</a> · runs in your browser via WebAssembly</p>
        <a href="{REPO_URL}" class="font-medium text-slate-400 hover:text-sky-300">ECLIPSE Lab on GitHub</a>
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

    # Copy static assets + thumbnails, then generate the gallery index.
    copy_public(args.output_dir)
    copy_thumbnails(args.output_dir)
    generate_index(all_notebooks, args.output_dir)


if __name__ == "__main__":
    main()
