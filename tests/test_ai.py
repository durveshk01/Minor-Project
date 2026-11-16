"""Provider failure handling, grounded prompts and credential non-disclosure."""

import json
from types import SimpleNamespace

import httpx
import pytest

from src.ai.gemini_client import GeminiClient
from src.ai.insight_generator import build_analytics_context, generate_insights
from src.data.pipeline import process_dataset
from src.data.schema import ColumnMapping


class ProviderFailure(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def fake_client(error=None, text=None):
    def invoke(**kwargs):
        if error:
            raise error
        return SimpleNamespace(text=text)
    return SimpleNamespace(models=SimpleNamespace(get=invoke, generate_content=invoke), close=lambda: None)


def test_missing_key_is_graceful_and_does_not_make_api_call():
    client = GeminiClient(api_key="")
    assert client.check_connectivity().status == "missing_key"
    assert client.generate("analytics", "Explain").status == "missing_key"


@pytest.mark.parametrize("error,status", [
    (ProviderFailure(400, "API key secret-test-value is invalid"), "invalid_key"),
    (ProviderFailure(403, "secret-test-value forbidden"), "invalid_key"),
    (ProviderFailure(429, "quota exhausted"), "quota_exhausted"),
    (ProviderFailure(429, "too many requests"), "rate_limited"),
    (ProviderFailure(404, "not found"), "model_unavailable"),
    (ProviderFailure(503, "unavailable"), "service_unavailable"),
    (httpx.ConnectError("offline"), "network_error"),
])
def test_api_failures_never_leak_raw_provider_message(error, status):
    client = GeminiClient(api_key="secret-test-value", client_factory=lambda **kwargs: fake_client(error), max_retries=0)
    result = client.generate("analytics", "Explain")
    assert result.status == status
    assert "secret-test-value" not in result.message
    assert client.check_connectivity().status == status


@pytest.mark.parametrize("text", [None, "", "not json", '{"narrative": "too short"}', '{"narrative": 12, "cited_sections": []}'])
def test_empty_and_invalid_responses_are_handled(text):
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: fake_client(text=text))
    assert client.generate("context", "Explain").status == "invalid_response"


def test_analytics_context_excludes_customer_rows_and_credentials(daily_frame):
    daily_frame["CustomerID"] = "private-customer"
    daily_frame["Email"] = "private@example.com"
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), metadata={"name": "Internal private-customer account"})
    context = build_analytics_context(dataset)
    serialized = json.dumps(context)
    assert "CustomerID" not in serialized and "private-customer" not in serialized and "private@example.com" not in serialized
    assert context["history"]["total"] == pytest.approx(daily_frame["Sales"].sum())


def test_unavailable_gemini_returns_labeled_actual_local_analytics(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"))
    context = build_analytics_context(dataset)
    result = generate_insights(context, client=GeminiClient(api_key=""))
    assert result.source == "Local analytics"
    assert result.status == "missing_key"
    assert f"{context['history']['total']:,.2f}" in result.text


def test_valid_grounded_gemini_response_and_bad_citations(daily_frame):
    context = build_analytics_context(process_dataset(daily_frame, ColumnMapping("Date", "Sales")))
    text = json.dumps({"narrative": "The observed sales series should be reviewed alongside its uncertainty and data coverage.", "cited_sections": ["history"]})
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: fake_client(text=text))
    result = generate_insights(context, client=client)
    assert result.source == "Gemini" and result.status == "success"
    bad_text = json.dumps({"narrative": "Unsupported forecast claims should not be accepted as genuine evidence.", "cited_sections": ["forecast"]})
    bad_client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: fake_client(text=bad_text))
    assert generate_insights(context, client=bad_client).status == "invalid_response"


def test_transient_rate_limit_is_bounded_and_retried():
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderFailure(429, "too many requests")
        return SimpleNamespace(text=json.dumps({"narrative": "Review the observed data coverage before relying on any trend explanation.", "cited_sections": ["history"]}))
    provider = SimpleNamespace(models=SimpleNamespace(generate_content=generate), close=lambda: None)
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: provider, max_retries=1)
    assert client.generate("computed data", "Explain only evidence").ok
    assert len(calls) == 2


def test_pinned_sdk_transport_is_closed_once_when_public_close_is_unavailable():
    closed = []
    provider = SimpleNamespace(_api_client=SimpleNamespace(_httpx_client=SimpleNamespace(close=lambda: closed.append(True))))
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: provider)
    client.close()
    client.close()
    assert closed == [True]
    assert not client.configured


def test_nontext_signature_and_thought_parts_do_not_break_json_explanations():
    document = {"narrative": "Review the observed data coverage and uncertainty before planning sales.", "cited_sections": ["history"]}
    parts = [
        SimpleNamespace(text="Internal reasoning is not an explanation.", thought=True),
        SimpleNamespace(text=None, thought=False, thought_signature="opaque-provider-signature"),
        SimpleNamespace(text=json.dumps(document), thought=False),
    ]
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])
    provider = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response), close=lambda: None)
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: provider)
    result = client.generate("computed analytics", "Explain only the evidence")
    assert result.ok
    assert json.loads(result.text) == document
    assert "Internal reasoning" not in result.text
