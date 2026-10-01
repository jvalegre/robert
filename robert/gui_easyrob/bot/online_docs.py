"""Bounded, cached reads of fixed official documentation pages."""

from functools import lru_cache
from html.parser import HTMLParser
import re
import time

import requests

from .answer_metadata import SourceCitation
from .bot_rag import KnowledgeChunk, normalize_terms
from .evidence_policy import official_documentation_domains


def documentation_urls(question: str) -> tuple[str, ...]:
    domains = official_documentation_domains(question)
    urls = []
    if "aqme.readthedocs.io" in domains:
        page = "Quickstart/defaults.html" if re.search(r"csearch|cmin|qprep|nmr|parameter|parametro|solvent|default", question, re.I) else "Modules/qdescp.html"
        if re.search(r"qcorr", question, re.I):
            page = "Modules/qcorr.html"
        urls.append("https://aqme.readthedocs.io/en/latest/" + page)
    if "robert.readthedocs.io" in domains:
        page = "verify" if re.search(r"verify|y[- ]shuffle|y[- ]mean", question, re.I) else "generate"
        if re.search(r"curate|curation|correlated", question, re.I):
            page = "curate"
        elif re.search(r"shap|pfi|predict|prediction", question, re.I):
            page = "predict" if page != "verify" else page
        path = "Technical/defaults.html" if re.search(r"\b(?:defaults?|parameters?|parametros?)\b", question, re.I) else f"Modules/{page}.html"
        urls.append(f"https://robert.readthedocs.io/en/latest/{path}")
    return tuple(urls)


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"p", "li", "dd", "dt", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")


def extract_document_text(html: str) -> str:
    main = re.search(r'<(?:div|main)\b[^>]*\brole=["\']main["\'][^>]*>', html, re.I)
    if not main:
        return ""
    html = html[main.end():].split("<footer", 1)[0]
    html = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
    parser = _TextParser()
    parser.feed(html)
    return "\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip())


@lru_cache(maxsize=24)
def _read_page(url: str, hour: int) -> str:
    # URLs come only from documentation_urls; never follow links or redirects.
    started = time.monotonic()
    with requests.get(url, timeout=(5, 12), allow_redirects=False, stream=True) as response:
        response.raise_for_status()
        if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            raise ValueError("The documentation response was not an HTML page.")
        content = bytearray()
        for block in response.iter_content(16384):
            content.extend(block)
            if len(content) > 1_000_000 or time.monotonic() - started > 20:
                raise ValueError("The documentation response exceeded its read budget.")
    text = extract_document_text(content.decode("utf-8", errors="replace"))
    if not text:
        raise ValueError("The documentation page contained no main text.")
    return text


def fetch_documentation(question: str):
    """Return relevant excerpts and citations; a failed read never becomes evidence."""
    chunks, sources = [], []
    terms = set(normalize_terms(question))
    urls = documentation_urls(question)
    for url in urls:
        try:
            text = _read_page(url, int(time.time() // 3600))
        except (requests.RequestException, ValueError):
            continue
        if not text:
            continue
        lines = text.splitlines()
        ranked = sorted(range(len(lines)), key=lambda i: len(terms & set(normalize_terms(lines[i]))), reverse=True)
        selected = set()
        for index in ranked[:4]:
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))
        excerpt = "\n".join(lines[index] for index in sorted(selected))[:2400 if len(urls) == 1 else 800]
        title = f"Official documentation: {url.split('/')[2]} / {url.rsplit('/', 1)[-1]}"
        citation = SourceCitation.create(title, url)
        if citation is None:
            continue
        sources.append(citation)
        for offset in range(0, len(excerpt), 800):
            chunks.append(KnowledgeChunk(
                id=f"online-{len(chunks)}", source=url, topic="Fetched official documentation", tab="",
                keywords=(), text=f"Fetched from {url}\n{excerpt[offset:offset + 800]}",
            ))
    return chunks[:3], tuple(sources)
