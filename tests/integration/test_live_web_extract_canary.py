"""Live canary tests for configured web extraction providers.

This file is intentionally excluded from normal test runs. It proves the real
provider path against public URLs, so it must only run when explicitly enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.tools.conftest import register_all_web_providers
from tools import web_tools

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_web_extract,
]

DEFAULT_MANIFEST = Path(__file__).with_name("live_web_extract_canary.yaml")
DEFAULT_RESULTS = Path("live_web_extract_canary_results.json")


def _load_manifest() -> dict[str, Any]:
    manifest_path = Path(os.getenv("WEB_EXTRACT_CANARY_MANIFEST", DEFAULT_MANIFEST))
    if not manifest_path.exists():
        pytest.fail(f"Web extract canary manifest does not exist: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        pytest.fail("Web extract canary manifest must be a mapping")
    return data


def _enabled_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        pytest.fail("Web extract canary manifest must define a cases list")

    enabled = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            pytest.fail(f"Canary case {index} must be a mapping")
        if case.get("enabled", True):
            enabled.append(case)
    if not enabled:
        pytest.fail("Web extract canary manifest has no enabled cases")
    return enabled


def _merged_case(defaults: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **case}
    for key in ("expected_contains", "forbidden_contains"):
        default_values = defaults.get(key) or []
        case_values = case.get(key) or []
        if key in defaults and key in case:
            merged[key] = [*default_values, *case_values]
    return merged


def _write_web_config(backend: str) -> None:
    hermes_home = Path(os.environ["HERMES_HOME"])
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"web": {"extract_backend": backend}}, sort_keys=True),
        encoding="utf-8",
    )


def _result_content(result: dict[str, Any]) -> str:
    return str(result.get("content") or result.get("raw_content") or "")


def _check_case(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    content = _result_content(result)
    content_lower = content.lower()
    errors: list[str] = []

    if case.get("require_success", True) and result.get("error"):
        errors.append(f"provider error: {result['error']}")

    min_chars = int(case.get("min_chars") or 0)
    if len(content) < min_chars:
        errors.append(f"content too short: {len(content)} chars < {min_chars}")

    for needle in case.get("expected_contains") or []:
        if str(needle).lower() not in content_lower:
            errors.append(f"missing expected text: {needle!r}")

    for needle in case.get("forbidden_contains") or []:
        if str(needle).lower() in content_lower:
            errors.append(f"found block/challenge text: {needle!r}")

    return errors


async def _extract_case(case: dict[str, Any]) -> dict[str, Any]:
    timeout_s = float(case.get("timeout_s") or 90)
    url = str(case["url"])
    result_json = await asyncio.wait_for(
        web_tools.web_extract_tool(
            [url],
            format=case.get("format"),
            use_llm_processing=False,
        ),
        timeout=timeout_s,
    )
    response = json.loads(result_json)
    if "results" not in response:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "metadata": {},
            "error": response.get("error") or "web_extract_tool returned no results",
        }
    results = response.get("results") or []
    if not results:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "metadata": {},
            "error": "web_extract_tool returned an empty results list",
        }
    return results[0]


@pytest.mark.asyncio
async def test_live_web_extract_canary() -> None:
    if os.getenv("LIVE_WEB_EXTRACT_CANARY") != "1":
        pytest.skip("set LIVE_WEB_EXTRACT_CANARY=1 to run live extract canaries")

    manifest = _load_manifest()
    defaults = manifest.get("defaults") or {}
    if not isinstance(defaults, dict):
        pytest.fail("Web extract canary defaults must be a mapping")

    backend = os.getenv("WEB_EXTRACT_BACKEND", "crawl4ai").strip() or "crawl4ai"
    _write_web_config(backend)
    register_all_web_providers()

    report: dict[str, Any] = {
        "backend": backend,
        "manifest": str(Path(os.getenv("WEB_EXTRACT_CANARY_MANIFEST", DEFAULT_MANIFEST))),
        "cases": [],
    }
    failures: list[str] = []

    try:
        for raw_case in _enabled_cases(manifest):
            case = _merged_case(defaults, raw_case)
            for key in ("id", "tier", "url"):
                if not case.get(key):
                    pytest.fail(f"Canary case missing required key {key!r}: {case}")

            result = await _extract_case(case)
            content = _result_content(result)
            errors = _check_case(case, result)
            report["cases"].append(
                {
                    "id": case["id"],
                    "tier": case["tier"],
                    "url": case["url"],
                    "ok": not errors,
                    "errors": errors,
                    "title": result.get("title", ""),
                    "content_chars": len(content),
                    "metadata": result.get("metadata", {}),
                }
            )
            if errors:
                failures.append(f"{case['id']} ({case['tier']}): " + "; ".join(errors))
    finally:
        result_path = Path(os.getenv("WEB_EXTRACT_CANARY_RESULTS", DEFAULT_RESULTS))
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        pytest.fail("Live web extract canary failures:\n" + "\n".join(failures))
