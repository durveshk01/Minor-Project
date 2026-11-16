"""Official google-genai SDK adapter with bounded requests and safe failure states."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx

from config.settings import get_settings


@dataclass(frozen=True)
class AIResponse:
    status: str
    text: str
    message: str
    model: str
    generated_at: str

    @property
    def ok(self) -> bool:
        return self.status == "success"


ERROR_MESSAGES = {
    "missing_key": "Gemini is not configured. Set GEMINI_API_KEY in the server environment or Streamlit secrets.",
    "invalid_key": "Gemini rejected the API credentials. Check the server-side key and its API permissions.",
    "quota_exhausted": "Gemini quota is exhausted. Core forecasting remains available; try again when quota resets.",
    "rate_limited": "Gemini is temporarily rate-limited. Please try again later.",
    "model_unavailable": "The configured Gemini model is unavailable. Select a model supported by your API account.",
    "network_error": "Gemini could not be reached. Check network connectivity and try again.",
    "service_unavailable": "Gemini is temporarily unavailable. Core analytics and forecasts remain available.",
    "invalid_request": "Gemini could not accept this request. Check the configured model and supported API settings.",
    "invalid_response": "Gemini returned an empty or invalid response. Local analytics remain available.",
    "configuration_error": "The Gemini client could not be initialized. Check SDK installation and server configuration.",
    "api_error": "The Gemini request failed. Core analytics and forecasts remain available.",
    "client_closed": "The Gemini client is closed. Create a new client before requesting an explanation.",
}


def classify_api_error(error: Exception) -> str:
    code = getattr(error, "code", getattr(error, "status_code", None))
    description = str(error).lower()
    if code in {401, 403} or (code == 400 and any(token in description for token in ["api key", "api_key", "credential"])):
        return "invalid_key"
    if code == 429:
        return "quota_exhausted" if "quota" in description or "exhausted" in description else "rate_limited"
    if code == 404:
        return "model_unavailable"
    if code in {500, 502, 503, 504}:
        return "service_unavailable"
    if code == 400:
        return "invalid_request"
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError, OSError)):
        return "network_error"
    return "api_error"


class GeminiClient:
    """A server-side client that never propagates provider error text or credentials."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client_factory: Callable | None = None,
        max_retries: int = 1,
    ) -> None:
        settings = get_settings()
        key = settings.gemini_api_key if api_key is None else api_key
        self.model = model or settings.gemini_model
        self.max_retries = min(max(max_retries, 0), 2)
        self._client = None
        self._initial_status = "missing_key" if not key else "success"
        if key:
            try:
                if client_factory:
                    self._client = client_factory(api_key=key)
                else:
                    from google import genai
                    from google.genai import types

                    self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=25_000, retry_options=types.HttpRetryOptions(attempts=1)))
            except Exception:
                self._initial_status = "configuration_error"
        self._logger = logging.getLogger("salesvision")

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _response(self, status: str, text: str = "", message: str = "") -> AIResponse:
        return AIResponse(status, text, message or ERROR_MESSAGES.get(status, ""), self.model, datetime.now(timezone.utc).isoformat())

    def check_connectivity(self) -> AIResponse:
        if not self.configured:
            return self._response(self._initial_status)
        try:
            self._client.models.get(model=self.model)
            return self._response("success", message="The configured Gemini model is reachable with the server-side credentials.")
        except Exception as exc:
            status = classify_api_error(exc)
            self._logger.info("Gemini connectivity status=%s", status)
            return self._response(status)

    def generate(self, prompt: str, system_instruction: str) -> AIResponse:
        if not self.configured:
            return self._response(self._initial_status)
        from google.genai import types

        schema = {
            "type": "object",
            "properties": {
                "narrative": {"type": "string"},
                "cited_sections": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["narrative", "cited_sections"],
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2, max_output_tokens=2500,
                        response_mime_type="application/json", response_schema=schema,
                    ),
                )
                candidates = getattr(response, "candidates", None)
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    parts = getattr(content, "parts", None) or []
                    text = "".join(part.text for part in parts if getattr(part, "text", None) and not getattr(part, "thought", False))
                else:
                    text = getattr(response, "text", None)
                if not isinstance(text, str) or not text.strip():
                    return self._response("invalid_response")
                try:
                    document = json.loads(text)
                except (TypeError, ValueError):
                    return self._response("invalid_response")
                if not isinstance(document, dict) or not isinstance(document.get("narrative"), str) or len(document["narrative"].strip()) < 20 or not isinstance(document.get("cited_sections"), list):
                    return self._response("invalid_response")
                return self._response("success", text=text, message="Gemini explanation generated from aggregated application analytics.")
            except Exception as exc:
                status = classify_api_error(exc)
                self._logger.info("Gemini generation status=%s attempt=%d", status, attempt + 1)
                if status in {"rate_limited", "service_unavailable"} and attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                return self._response(status)
        return self._response("api_error")

    def close(self) -> None:
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
                else:
                    transport = getattr(getattr(self._client, "_api_client", None), "_httpx_client", None)
                    if transport is not None:
                        transport.close()
            except Exception:
                pass
            finally:
                self._client = None
                self._initial_status = "client_closed"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
