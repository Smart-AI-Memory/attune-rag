#!/usr/bin/env python3
"""Render attune-rag's ``.help/`` corpus into static help pages under ``help/``.

Self-contained: reads ``../.help/templates/`` + ``../.help/features.yaml``
directly (no ``attune.*`` import — this lives in the attune-rag repo).
Same markdown-it-py renderer + shared ``brand.css`` as the landing page.

Build-time only; Vercel serves the committed HTML as-is. Re-runnable:
    python build_help.py            # build into ./help/
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
    from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover - build-time deps
    sys.stderr.write("requires markdown-it-py + pyyaml\n")
    raise SystemExit(1) from None

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / ".help" / "templates"
MANIFEST = HERE.parent / ".help" / "features.yaml"
BRAND_CSS = (HERE / "brand.css").read_text(encoding="utf-8")
SITE = "https://attune-rag.dev"
GITHUB = "https://github.com/Smart-AI-Memory/attune-rag"
PYPI = "https://pypi.org/project/attune-rag/"

STD_KINDS = (
    "comparison",
    "concept",
    "error",
    "faq",
    "note",
    "quickstart",
    "reference",
    "task",
    "tip",
    "troubleshooting",
    "warning",
)
INTENT_GROUPS = {
    "do": ("task", "quickstart"),
    "solve": ("troubleshooting", "error", "faq"),
    "understand": ("concept",),
    "lookup": ("reference",),
}
INTENT_LABELS = {
    "do": ("Do something", "tasks &amp; quickstarts"),
    "solve": ("Solve a problem", "troubleshooting, errors, FAQ"),
    "understand": ("Understand a concept", "how it works"),
    "lookup": ("Look something up", "reference"),
}

_FM_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_ANCHOR_RE = re.compile(r'<a\s+href="([^"]*)"([^>]*)>')
_KEEP_PREFIX_RE = re.compile(r"^(/|https?://|#|mailto:|tel:)")

HELP_CSS = """
    .help-nav { max-width: 60rem; margin: 0 auto; display: flex;
      align-items: center; justify-content: space-between; gap: 1rem;
      padding: 1.25rem 1.5rem; font-size: 0.9rem; }
    .help-nav .crumbs a { color: var(--muted); text-decoration: none; }
    .help-nav .crumbs a:hover { color: var(--primary); }
    .help-nav .crumbs .sep { color: var(--rule); margin: 0 0.4rem; }
    .help-nav .api-link { color: var(--primary); text-decoration: none; font-weight: 600; }
    main.help-home { max-width: 60rem; margin: 0 auto; padding: 1rem 1.5rem 5rem; }
    .help-home h1 { font-family: "Manrope", -apple-system, sans-serif;
      font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 1rem 0 0.5rem; }
    .help-home .lede { color: var(--muted); font-size: 1.15rem; margin: 0 0 2rem; }
    .search-wrap { margin: 0 0 2.5rem; }
    #help-search { width: 100%; font-size: 1.05rem; padding: 0.85rem 1.1rem;
      border: 1px solid var(--rule); border-radius: 10px;
      background: var(--surface-low); color: var(--ink); }
    #help-search:focus { outline: 2px solid var(--primary); border-color: var(--primary); }
    #search-results { list-style: none; margin: 0.75rem 0 0; padding: 0; }
    #search-results li { margin: 0 0 0.4rem; }
    #search-results a { display: block; padding: 0.6rem 0.8rem; border-radius: 8px;
      text-decoration: none; color: var(--ink); border: 1px solid transparent; }
    #search-results a:hover { background: var(--surface-low); border-color: var(--rule); }
    #search-results .r-kind { color: var(--muted); font-size: 0.85rem; }
    #search-results .r-snip { color: var(--muted); font-size: 0.9rem; display: block; margin-top: 0.15rem; }
    .section-h { font-family: "Manrope", -apple-system, sans-serif;
      font-size: 1.3rem; font-weight: 700; margin: 2.5rem 0 1rem;
      padding-top: 1.25rem; border-top: 1px solid var(--rule); }
    .intent-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 1rem; }
    .intent-card { display: block; padding: 1.1rem 1.2rem; border: 1px solid var(--rule);
      border-radius: 12px; text-decoration: none; color: var(--ink); background: var(--surface-low); }
    .intent-card:hover { border-color: var(--primary); }
    .intent-card .it-title { display: block; font-weight: 700; font-size: 1.05rem; }
    .intent-card .it-sub { display: block; color: var(--muted); font-size: 0.9rem; margin-top: 0.2rem; }
    .feature-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 0.85rem; }
    .feature-card { display: block; padding: 0.9rem 1.1rem; border: 1px solid var(--rule);
      border-radius: 10px; text-decoration: none; color: var(--ink); }
    .feature-card:hover { border-color: var(--primary); background: var(--surface-low); }
    .feature-card .fc-name { display: block; font-weight: 700; }
    .feature-card .fc-meta { display: block; color: var(--muted); font-size: 0.82rem; margin-top: 0.2rem; }
    .kind-tabs { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0 0 2rem; }
    .kind-tabs a { padding: 0.35rem 0.75rem; border: 1px solid var(--rule); border-radius: 999px;
      text-decoration: none; color: var(--muted); font-size: 0.88rem; }
    .kind-tabs a:hover { border-color: var(--primary); color: var(--primary); }
    .markdown-body { max-width: 50rem; margin: 0 auto; padding: 1rem 1.5rem 5rem; }
    .markdown-body h1 { font-family: "Manrope", -apple-system, sans-serif;
      font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0.5rem 0 1rem; }
    .markdown-body h2 { font-size: 1.4rem; margin: 2rem 0 0.75rem; }
    .markdown-body h3 { font-size: 1.15rem; margin: 1.5rem 0 0.5rem; }
    .markdown-body p, .markdown-body li { line-height: 1.6; }
    .markdown-body code { background: var(--surface-low); padding: 0.1rem 0.35rem;
      border-radius: 4px; font-size: 0.9em; }
    .markdown-body pre { background: var(--surface-low); border: 1px solid var(--rule);
      border-radius: 8px; padding: 1rem; overflow-x: auto; }
    .markdown-body pre code { background: none; padding: 0; }
    .markdown-body table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
    .markdown-body th, .markdown-body td { border: 1px solid var(--rule); padding: 0.5rem 0.7rem; text-align: left; }
    .markdown-body blockquote { border-left: 3px solid var(--rule); margin: 1rem 0;
      padding: 0.25rem 1rem; color: var(--muted); }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>__TITLE__</title>
  <meta name="description" content="__DESC__" />
  <meta property="og:title" content="__TITLE__" />
  <meta property="og:description" content="__DESC__" />
  <meta property="og:type" content="website" />
  <meta property="og:image" content="__SITE__/og.png" />
  <link rel="canonical" href="__CANONICAL__" />
  <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
  <link rel="icon" href="/favicon.ico" sizes="32x32" />
  <style>__CSS__</style>
  <script defer src="/_vercel/insights/script.js"></script>
</head>
<body>
  <nav class="help-nav">
    <span class="crumbs">__CRUMBS__</span>
    <a class="api-link" href="__GITHUB__">Source &amp; API &rarr;</a>
  </nav>
__BODY__
  <footer>
    <span>Smart AI Memory</span>
    <span><a href="__GITHUB__">GitHub</a>&nbsp;&middot;&nbsp;<a href="__PYPI__">PyPI</a></span>
  </footer>
</body>
</html>
"""

FOOTER_CSS = """
    footer { max-width: 60rem; margin: 4rem auto 0; padding: 1.5rem;
      border-top: 1px solid var(--rule); color: var(--muted); font-size: 0.85rem;
      display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
    footer a { color: var(--muted); text-decoration: none; }
    footer a:hover { color: var(--ink); }
"""


def _md() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")


def _write(path: Path, text: str) -> None:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    path.write_text(cleaned + "\n", encoding="utf-8")


def _neutralize_relative_links(body_html: str) -> str:
    def repl(m: re.Match[str]) -> str:
        href, rest = m.group(1), m.group(2)
        return m.group(0) if _KEEP_PREFIX_RE.match(href) else f"<a{rest}>"

    return _ANCHOR_RE.sub(repl, body_html)


def _strip_fm(text: str) -> str:
    return _FM_RE.sub("", text, count=1)


def _doc_title(body: str, feature: str) -> str:
    m = _H1_RE.search(body)
    return m.group(1).strip() if m else _title_case(feature)


def _title_case(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _first_para(body: str, *, max_chars: int = 200) -> str:
    for block in _strip_fm(body).split("\n\n"):
        line = " ".join(block.split())
        if line and not line.startswith("#") and not line.startswith("|"):
            return line[:max_chars]
    return ""


def _crumbs(*parts: tuple[str, str | None]) -> str:
    out = []
    for label, href in parts:
        e = html.escape(label)
        out.append(f'<a href="{href}">{e}</a>' if href else f"<span>{e}</span>")
    return '<span class="sep">&rsaquo;</span>'.join(out)


def _page(*, title: str, desc: str, crumbs: str, body: str, canonical: str) -> str:
    return (
        PAGE.replace("__TITLE__", html.escape(title, quote=True))
        .replace("__DESC__", html.escape(desc, quote=True))
        .replace("__CANONICAL__", html.escape(canonical, quote=True))
        .replace("__SITE__", SITE)
        .replace("__GITHUB__", GITHUB)
        .replace("__PYPI__", PYPI)
        .replace("__CSS__", BRAND_CSS + HELP_CSS + FOOTER_CSS)
        .replace("__CRUMBS__", crumbs)
        .replace("__BODY__", body)
    )


def load_manifest() -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    return data.get("features", {})


def discover_features(manifest: dict) -> list[dict]:
    feats = []
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir():
            continue
        kinds = sorted(p.stem for p in d.glob("*.md"))
        kinds = [k for k in STD_KINDS if k in kinds] + [k for k in kinds if k not in STD_KINDS]
        feats.append(
            {
                "name": d.name,
                "kinds": kinds,
                "description": (manifest.get(d.name) or {}).get("description", ""),
            }
        )
    return feats


def _build_kind_page(out: Path, md: MarkdownIt, feature: str, kind: str) -> dict:
    raw = (CORPUS / feature / f"{kind}.md").read_text(encoding="utf-8")
    body = _strip_fm(raw)
    title = _doc_title(body, feature)
    body_html = _neutralize_relative_links(md.render(body))
    crumbs = _crumbs(
        ("attune-rag", "/"),
        ("Help", "/help"),
        (_title_case(feature), f"/help/{feature}"),
        (kind, None),
    )
    page = _page(
        title=f"{title} — {kind}",
        desc=_first_para(raw) or f"{title}: {kind}.",
        crumbs=crumbs,
        body=f'  <article class="markdown-body">\n{body_html}\n  </article>',
        canonical=f"{SITE}/help/{feature}/{kind}",
    )
    _write(out / feature / f"{kind}.html", page)
    return {
        "title": title,
        "feature": feature,
        "kind": kind,
        "url": f"/help/{feature}/{kind}",
        "keywords": " ".join(
            sorted({feature, kind, *title.lower().split(), *_first_para(raw).lower().split()})
        ),
        "snippet": _first_para(raw, max_chars=160),
    }


def _build_feature_page(out: Path, feat: dict) -> None:
    feature = feat["name"]
    tabs = "".join(f'<a href="/help/{feature}/{k}">{html.escape(k)}</a>' for k in feat["kinds"])
    cards = "".join(
        f'<a class="feature-card" href="/help/{feature}/{k}">'
        f'<span class="fc-name">{html.escape(k)}</span></a>'
        for k in feat["kinds"]
    )
    lede = (
        html.escape(feat["description"])
        if feat["description"]
        else (f"{len(feat['kinds'])} help kinds available.")
    )
    body = (
        '  <main class="help-home">\n'
        f"    <h1>{html.escape(_title_case(feature))}</h1>\n"
        f'    <p class="lede">{lede}</p>\n'
        f'    <div class="kind-tabs">{tabs}</div>\n'
        f'    <div class="feature-grid">{cards}</div>\n'
        "  </main>"
    )
    crumbs = _crumbs(("attune-rag", "/"), ("Help", "/help"), (_title_case(feature), None))
    page = _page(
        title=f"{_title_case(feature)} — attune-rag help",
        desc=feat["description"] or f"{_title_case(feature)} — help and reference.",
        crumbs=crumbs,
        body=body,
        canonical=f"{SITE}/help/{feature}",
    )
    (out / feature).mkdir(parents=True, exist_ok=True)
    _write(out / feature / "index.html", page)


def _build_landing(out: Path, features: list[dict]) -> None:
    intent_cards = ""
    for intent, (label, sub) in INTENT_LABELS.items():
        kinds = INTENT_GROUPS.get(intent, ())
        target = next(
            (f"/help/{f['name']}/{k}" for f in features for k in kinds if k in f["kinds"]), "/help"
        )
        intent_cards += (
            f'<a class="intent-card" href="{target}">'
            f'<span class="it-title">{label}</span>'
            f'<span class="it-sub">{sub}</span></a>'
        )
    feature_cards = "".join(
        f'<a class="feature-card" href="/help/{f["name"]}">'
        f'<span class="fc-name">{html.escape(_title_case(f["name"]))}</span>'
        f'<span class="fc-meta">{len(f["kinds"])} kinds</span></a>'
        for f in features
    )
    body = (
        '  <main class="help-home">\n'
        "    <h1>attune-rag help</h1>\n"
        '    <p class="lede">Learn, do, solve, and look things up — generated from '
        "the attune-rag corpus.</p>\n"
        '    <div class="search-wrap">\n'
        '      <input id="help-search" type="search" placeholder="Search help…" autocomplete="off" />\n'
        '      <ul id="search-results"></ul>\n'
        "    </div>\n"
        '    <div class="section-h">Browse by what you need</div>\n'
        f'    <div class="intent-row">{intent_cards}</div>\n'
        '    <div class="section-h">All features</div>\n'
        f'    <div class="feature-grid">{feature_cards}</div>\n'
        '    <script src="/help/search.js" defer></script>\n'
        "  </main>"
    )
    crumbs = _crumbs(("attune-rag", "/"), ("Help", None))
    page = _page(
        title="attune-rag help",
        desc="Browse and search the attune-rag help corpus — tasks, concepts, "
        "reference, and troubleshooting for every feature.",
        crumbs=crumbs,
        body=body,
        canonical=f"{SITE}/help",
    )
    _write(out / "index.html", page)


SEARCH_JS = """// Client-side help search over the static index (no server).
(function () {
  var input = document.getElementById("help-search");
  var results = document.getElementById("search-results");
  if (!input || !results) return;
  var index = [];
  fetch("/help/search-index.json").then(function (r) { return r.json(); })
    .then(function (data) { index = data; });
  function tokens(s) { return s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean); }
  function render(hits) {
    results.innerHTML = hits.map(function (h) {
      return '<li><a href="' + h.url + '"><strong>' + h.title +
        '</strong> <span class="r-kind">' + h.feature + " / " + h.kind +
        '</span><span class="r-snip">' + (h.snippet || "") + "</span></a></li>";
    }).join("");
  }
  input.addEventListener("input", function () {
    var q = tokens(input.value);
    if (!q.length) { results.innerHTML = ""; return; }
    var scored = index.map(function (it) {
      var feat = (it.feature || "").toLowerCase();
      var title = (it.title || "").toLowerCase();
      var kw = (it.keywords || "").toLowerCase();
      var score = q.reduce(function (acc, t) {
        return acc + (feat.indexOf(t) >= 0 ? 3 : 0) +
          (title.indexOf(t) >= 0 ? 2 : 0) + (kw.indexOf(t) >= 0 ? 1 : 0);
      }, 0);
      return { it: it, score: score };
    }).filter(function (x) { return x.score > 0; });
    var order = { concept: 0, quickstart: 1, task: 2, reference: 3 };
    scored.sort(function (a, b) {
      if (b.score !== a.score) return b.score - a.score;
      var ak = order[a.it.kind] === undefined ? 9 : order[a.it.kind];
      var bk = order[b.it.kind] === undefined ? 9 : order[b.it.kind];
      return ak - bk;
    });
    render(scored.slice(0, 20).map(function (x) { return x.it; }));
  });
})();
"""


def build(out: Path) -> int:
    manifest = load_manifest()
    features = discover_features(manifest)
    if not features:
        sys.stderr.write(f"no features under {CORPUS}\n")
        return 1
    md = _md()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    search_index, n = [], 0
    for feat in features:
        _build_feature_page(out, feat)
        for kind in feat["kinds"]:
            search_index.append(_build_kind_page(out, md, feat["name"], kind))
            n += 1
    _build_landing(out, features)
    (out / "search-index.json").write_text(
        json.dumps(search_index, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    _write(out / "search.js", SEARCH_JS)
    print(
        f"built {len(features)} features, {n} kind pages, {len(search_index)} search entries -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(build(HERE / "help"))
