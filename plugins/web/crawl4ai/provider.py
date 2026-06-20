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
from typing import Any, Dict, Iterable, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


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


def _coerce_text(value: Any) -> str:
    """Normalize Crawl4AI markdown/html values across response variants."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in (
            "raw_markdown",
            "markdown",
            "fit_markdown",
            "content",
            "html",
            "text",
        ):
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

    if item.get("success") is False:
        return {
            "url": url,
            "title": title,
            "content": "",
            "raw_content": "",
            "metadata": metadata,
            "error": str(item.get("error") or "Crawl4AI extraction failed"),
        }

    markdown = _coerce_text(item.get("markdown"))
    html = _coerce_text(item.get("html") or item.get("cleaned_html"))
    content = (
        markdown
        or _coerce_text(item.get("content"))
        or _coerce_text(item.get("text"))
        or html
    )
    raw_content = _coerce_text(item.get("raw_content")) or html or content

    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": raw_content,
        "metadata": metadata,
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

        payload: Dict[str, Any] = {
            "urls": urls,
            "cache_mode": "bypass",
        }
        if kwargs.get("format") == "html":
            payload["only_text"] = False

        try:
            response = httpx.post(
                f"{base_url}/crawl",
                json=payload,
                timeout=90,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            response.raise_for_status()
            body = response.json()
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
