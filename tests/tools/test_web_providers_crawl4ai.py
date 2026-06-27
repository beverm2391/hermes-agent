"""Tests for the Crawl4AI web extract provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


class TestCrawl4AIProviderConfig:
    def test_available_when_url_and_token_set(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert Crawl4AIWebSearchProvider().is_available() is True

    def test_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert Crawl4AIWebSearchProvider().is_available() is False

    def test_unavailable_when_token_missing(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert Crawl4AIWebSearchProvider().is_available() is False

    def test_extract_only_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert issubclass(Crawl4AIWebSearchProvider, WebSearchProvider)
        provider = Crawl4AIWebSearchProvider()
        assert provider.name == "crawl4ai"
        assert provider.supports_search() is False
        assert provider.supports_extract() is True


class TestCrawl4AIProviderExtract:
    def _mock_response(self, data, status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = data
        response.raise_for_status = MagicMock()
        return response

    def test_extract_normalizes_markdown_response(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        payload = {
            "results": [
                {
                    "url": "https://example.com",
                    "markdown": "# Example\nHello",
                    "html": "<h1>Example</h1>",
                    "metadata": {"title": "Example Domain"},
                }
            ]
        }

        with patch("httpx.post", return_value=self._mock_response(payload)) as post:
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert result[0]["url"] == "https://example.com"
        assert result[0]["title"] == "Example Domain"
        assert result[0]["content"] == "# Example\nHello"
        assert result[0]["raw_content"] == "# Example\nHello"
        assert result[0]["metadata"]["title"] == "Example Domain"
        assert result[0]["metadata"]["content_chars"] == 15
        assert result[0]["metadata"]["content_truncated"] is False
        assert result[0]["metadata"]["raw_content_truncated"] is False
        assert isinstance(result[0]["metadata"]["crawl4ai_elapsed_ms"], int)
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token"
        crawler_params = post.call_args.kwargs["json"]["crawler_config"]["params"]
        assert crawler_params["cache_mode"] == "bypass"

    def test_extract_handles_markdown_object_response(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235/")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        payload = {
            "results": [
                {
                    "url": "https://example.com",
                    "markdown": {"raw_markdown": "raw md", "fit_markdown": "fit md"},
                    "metadata": {},
                }
            ]
        }

        with patch("httpx.post", return_value=self._mock_response(payload)) as post:
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert post.call_args.args[0] == "http://localhost:11235/crawl"
        assert result[0]["content"] == "fit md"
        assert result[0]["raw_content"] == "fit md"

    def test_extract_prefers_fit_markdown_over_raw_markdown(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        payload = {
            "results": [
                {
                    "url": "https://example.com",
                    "fit_markdown": "filtered md",
                    "markdown": {"raw_markdown": "raw md"},
                    "metadata": {},
                }
            ]
        }

        with patch("httpx.post", return_value=self._mock_response(payload)):
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert result[0]["content"] == "filtered md"

    def test_extract_does_not_silently_fall_back_to_html(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        payload = {
            "results": [
                {
                    "url": "https://example.com",
                    "html": "<!doctype html><html><body>raw page</body></html>",
                    "metadata": {},
                }
            ]
        }

        with patch("httpx.post", return_value=self._mock_response(payload)):
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert result[0]["content"] == ""
        assert result[0]["raw_content"].startswith("<!doctype html>")
        assert result[0]["metadata"]["markdown_missing"] is True
        assert result[0]["error"] == "Crawl4AI returned HTML without markdown"

    def test_extract_caps_huge_markdown_before_summarization(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        huge = "x" * 70_000
        payload = {"results": [{"url": "https://example.com", "markdown": huge}]}

        with patch("httpx.post", return_value=self._mock_response(payload)):
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert len(result[0]["content"]) == 60_000
        assert result[0]["metadata"]["content_truncated"] is True

    def test_missing_token_returns_per_url_error(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert result[0]["url"] == "https://example.com"
        assert "CRAWL4AI_API_TOKEN" in result[0]["error"]

    def test_http_error_returns_per_url_error(self, monkeypatch):
        import httpx

        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        response = MagicMock()
        response.status_code = 401
        error = httpx.HTTPStatusError("unauthorized", request=MagicMock(), response=response)

        with patch("httpx.post", side_effect=error):
            result = Crawl4AIWebSearchProvider().extract(["https://example.com"])

        assert result[0]["error"] == "Crawl4AI returned HTTP 401"

    def test_extract_sends_bounded_crawl_config(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        payload = {"results": [{"url": "https://example.com", "markdown": "ok"}]}

        with patch("httpx.post", return_value=self._mock_response(payload)) as post:
            Crawl4AIWebSearchProvider().extract(["https://example.com"])

        request = post.call_args.kwargs["json"]
        assert request["browser_config"]["type"] == "BrowserConfig"
        assert request["crawler_config"]["type"] == "CrawlerRunConfig"
        crawler_params = request["crawler_config"]["params"]
        assert crawler_params["cache_mode"] == "bypass"
        assert crawler_params["page_timeout"] == 45_000
        assert crawler_params["remove_overlay_elements"] is True


@pytest.mark.usefixtures("web_registry_populated")
class TestCrawl4AIWebExtractDispatch:
    @pytest.mark.asyncio
    async def test_web_extract_dispatches_to_configured_crawl4ai(self, monkeypatch):
        from tools import web_tools

        register_all_web_providers()
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "test-token")
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "crawl4ai"},
        )
        async def safe_url(url):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)

        payload = {
            "results": [
                {
                    "url": "https://example.com",
                    "markdown": "Example text",
                    "metadata": {"title": "Example"},
                }
            ]
        }

        with patch("httpx.post", return_value=MagicMock(
            json=MagicMock(return_value=payload),
            raise_for_status=MagicMock(),
        )):
            result_json = await web_tools.web_extract_tool(
                ["https://example.com"],
                use_llm_processing=False,
            )

        result = json.loads(result_json)
        assert result["results"][0]["content"] == "Example text"
        assert result["results"][0]["title"] == "Example"
