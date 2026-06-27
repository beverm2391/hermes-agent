"""Crawl4AI content extraction provider.

This provider is extraction-only. Pair it with a search provider such as
SearXNG:

    web:
      search_backend: "searxng"
      extract_backend: "crawl4ai"

Env vars:

    CRAWL4AI_URL=http://localhost:11235
    CRAWL4AI_API_TOKEN=...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterable, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

CRAWL4AI_ENDPOINT = "/crawl"
REQUEST_TIMEOUT_SECONDS = 75
PAGE_TIMEOUT_MS = 45_000
MAX_EXTRACT_CHARS = 60_000
HTML_EXCERPT_CHARS = 8_000


def _config_env(name: str) -> str:
    """Return config-aware env values while keeping provider imports cheap."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


def _crawl4ai_url() -> str:
    return _config_env("CRAWL4AI_URL")


def _crawl4ai_token() -> str:
    return _config_env("CRAWL4AI_API_TOKEN")


def _coerce_text(value: Any, preferred_keys: Iterable[str] | None = None) -> str:
    """Normalize Crawl4AI text-ish values across response variants."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        keys = tuple(preferred_keys or ()) + (
            # Fit/filter variants are the most useful for agent context because
            # Crawl4AI has already discarded obvious boilerplate.
            "fit_markdown",
            "filtered_markdown",
            "markdown_with_citations",
            "raw_markdown",
            "markdown",
            "content",
            "text",
        )
        for key in keys:
            nested = value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return str(value)


def _response_items(payload: Any) -> List[Dict[str, Any]]:
    """Return per-page result dictionaries from common Crawl4AI envelopes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("results", "data", "pages", "documents"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]

    value = payload.get("result")
    if isinstance(value, dict):
        return [value]

    if any(key in payload for key in ("markdown", "html", "content", "text")):
        return [payload]

    return []


def _looks_like_html(text: str) -> bool:
    prefix = text.lstrip()[:200].lower()
    return prefix.startswith("<!doctype html") or prefix.startswith("<html") or "<body" in prefix


def _cap_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _normalize_item(item: Dict[str, Any], fallback_url: str) -> Dict[str, Any]:
    """Map a Crawl4AI page result to Hermes' standard extract document."""
    url = str(
        item.get("url")
        or item.get("input_url")
        or item.get("requested_url")
        or fallback_url
        or ""
    )
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    title = str(item.get("title") or metadata.get("title") or "")
    metadata = {
        **metadata,
        **{
            key: item[key]
            for key in ("status_code", "response_status", "crawl4ai_elapsed_ms")
            if key in item
        },
    }

    if item.get("success") is False:
        return {
            "url": url,
            "title": title,
            "content": "",
            "raw_content": "",
            "metadata": metadata,
            "error": str(item.get("error") or "Crawl4AI extraction failed"),
        }

    markdown = (
        _coerce_text(item.get("fit_markdown"))
        or _coerce_text(item.get("markdown"), ("fit_markdown", "raw_markdown"))
        or _coerce_text(item.get("raw_markdown"))
    )
    html = _coerce_text(item.get("html") or item.get("cleaned_html"), ("html", "cleaned_html"))
    content = markdown or (
        "" if _looks_like_html(_coerce_text(item.get("content"))) else _coerce_text(item.get("content"))
    ) or (
        "" if _looks_like_html(_coerce_text(item.get("text"))) else _coerce_text(item.get("text"))
    )

    if not content and html:
        excerpt, truncated = _cap_text(html, HTML_EXCERPT_CHARS)
        return {
            "url": url,
            "title": title,
            "content": "",
            "raw_content": excerpt,
            "metadata": {
                **metadata,
                "html_excerpt_chars": len(excerpt),
                "html_truncated": truncated,
                "markdown_missing": True,
            },
            "error": "Crawl4AI returned HTML without markdown",
        }

    content, content_truncated = _cap_text(content, MAX_EXTRACT_CHARS)
    raw_content = _coerce_text(item.get("raw_content")) or content
    raw_content, raw_truncated = _cap_text(raw_content, MAX_EXTRACT_CHARS)
    metadata = {
        **metadata,
        "content_chars": len(content),
        "content_truncated": content_truncated,
        "raw_content_truncated": raw_truncated,
    }

    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": raw_content,
        "metadata": metadata,
    }


def _crawl_payload(urls: List[str], *, want_html: bool = False) -> Dict[str, Any]:
    """Build a bounded Crawl4AI request body.

    Crawl4AI's Docker API accepts typed config objects for BrowserConfig and
    CrawlerRunConfig. Keeping the timeout/content settings here makes the red
    Hermes extraction path deterministic and prevents a raw HTML blob from
    being treated as normal markdown downstream.
    """
    crawler_params: Dict[str, Any] = {
        "stream": False,
        "cache_mode": "bypass",
        "page_timeout": PAGE_TIMEOUT_MS,
        "wait_until": "domcontentloaded",
        "word_count_threshold": 10,
        "remove_overlay_elements": True,
        "excluded_tags": ["script", "style", "nav", "footer", "header", "form", "aside"],
    }
    if want_html:
        crawler_params["only_text"] = False

    return {
        "urls": urls,
        "browser_config": {
            "type": "BrowserConfig",
            "params": {"headless": True},
        },
        "crawler_config": {
            "type": "CrawlerRunConfig",
            "params": crawler_params,
        },
    }

def _error_results(urls: Iterable[str], message: str) -> List[Dict[str, Any]]:
    return [
        {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "metadata": {},
            "error": message,
        }
        for url in urls
    ]


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Extract content via a self-hosted Crawl4AI service."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        return bool(_crawl4ai_url() and _crawl4ai_token())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract one or more URLs through Crawl4AI's ``/crawl`` endpoint."""
        import httpx

        base_url = _crawl4ai_url().rstrip("/")
        token = _crawl4ai_token()
        if not base_url:
            return _error_results(urls, "CRAWL4AI_URL is not set")
        if not token:
            return _error_results(urls, "CRAWL4AI_API_TOKEN is not set")

        payload = _crawl_payload(urls, want_html=kwargs.get("format") == "html")

        try:
            started = time.monotonic()
            response = httpx.post(
                f"{base_url}{CRAWL4AI_ENDPOINT}",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            response.raise_for_status()
            body = response.json()
            elapsed_ms = round((time.monotonic() - started) * 1000)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Crawl4AI HTTP error while extracting %d URL(s): %s",
                len(urls),
                exc.response.status_code,
            )
            return _error_results(
                urls,
                f"Crawl4AI returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.warning(
                "Crawl4AI request error while extracting %d URL(s): %s",
                len(urls),
                type(exc).__name__,
            )
            return _error_results(urls, f"Could not reach Crawl4AI at {base_url}: {exc}")
        except ValueError:
            logger.warning(
                "Crawl4AI returned non-JSON response while extracting %d URL(s)",
                len(urls),
            )
            return _error_results(urls, "Could not parse Crawl4AI response as JSON")

        items = _response_items(body)
        if not items:
            return _error_results(urls, "Crawl4AI returned no extract results")

        results: List[Dict[str, Any]] = []
        for index, item in enumerate(items):
            fallback_url = urls[index] if index < len(urls) else ""
            item.setdefault("crawl4ai_elapsed_ms", elapsed_ms)
            results.append(_normalize_item(item, fallback_url))
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "self-hosted",
            "tag": "Private content extraction. Pair with SearXNG for search.",
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4AI service URL (e.g. http://localhost:11235)",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
                {
                    "key": "CRAWL4AI_API_TOKEN",
                    "prompt": "Crawl4AI bearer token",
                    "secret": True,
                },
            ],
        }
