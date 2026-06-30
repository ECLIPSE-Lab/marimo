import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_build():
    spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


@pytest.fixture
def thumbs(tmp_path, monkeypatch):
    """Point the build at an isolated thumbnails dir with one known PNG present."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / "ptychographic_ctf.png").write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(build, "THUMBNAILS_DIR", thumb_dir)
    return thumb_dir


def _generate(tmp_path, notebooks):
    out = tmp_path / "site"
    build.generate_index(notebooks, str(out))
    return (out / "index.html").read_text(encoding="utf-8")


def test_hero_and_footer_present(tmp_path, thumbs):
    html = _generate(tmp_path, ["apps/ptychographic_ctf.py"])
    assert build.SITE_TITLE in html
    assert "WebAssembly" in html
    assert "marimo.io" in html


def test_known_app_with_thumbnail_renders_image(tmp_path, thumbs):
    html = _generate(tmp_path, ["apps/ptychographic_ctf.py"])
    assert 'href="apps/ptychographic_ctf.html"' in html
    assert "Ptychographic CTF" in html
    assert "thumbnails/ptychographic_ctf.png" in html
    assert "<img" in html
    # tags from metadata
    assert ">CTF<" in html


def test_app_without_thumbnail_falls_back_to_placeholder(tmp_path, thumbs):
    # probe.png is absent in the isolated thumbnails dir → gradient placeholder.
    html = _generate(tmp_path, ["apps/probe.py"])
    assert "thumbnails/probe.png" not in html
    assert "bg-gradient-to-br" in html  # placeholder block
    assert "STEM Probe" in html


def test_unknown_app_uses_titleized_fallback(tmp_path, thumbs):
    html = _generate(tmp_path, ["apps/some_new_widget.py"])
    assert "Some New Widget" in html
    assert 'href="apps/some_new_widget.html"' in html


def test_cards_ordered_by_metadata(tmp_path, thumbs):
    notebooks = [
        "apps/probe.py",  # order 50
        "apps/ptychographic_ctf.py",  # order 10
        "apps/ff_stem_ssnr.py",  # order 40
    ]
    html = _generate(tmp_path, notebooks)
    assert html.index("Ptychographic CTF") < html.index("FF-STEM SSNR") < html.index("STEM Probe")
