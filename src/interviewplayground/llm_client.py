import io
import json
import os
import re
import time
from typing import Any

import litellm


def _balanced_object(text: str, start: int) -> str | None:
    """Return the balanced {...} substring beginning at index `start`, or None."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == '\\' and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _extract_json(text: str) -> Any:
    """
    Extract the first valid JSON object from text that may contain non-JSON
    preamble (e.g. chain-of-thought from a thinking model) or markdown fences.

    Tries a fenced ```json block first, then scans every '{' and returns the
    first balanced object that parses — so a decoy like "{not json}" in the
    reasoning is skipped in favor of the real payload.
    """
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    search_from = 0
    while True:
        start = text.find('{', search_from)
        if start == -1:
            break
        candidate = _balanced_object(text, start)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        search_from = start + 1

    raise ValueError(f"No valid JSON object found in response: {text[:200]!r}")


def _json_response_format(schema: dict | None) -> dict:
    """
    Build the response_format for a JSON call.

    With a schema, request strict constrained decoding
    (response_format type "json_schema"). This is essential for Gemini 3
    thinking models, which in schema-less JSON mode occasionally emit a
    premature end-of-turn mid-object (truncated JSON with finish_reason
    'stop'); constrained decoding forces a complete, schema-valid object.
    Without a schema, fall back to plain JSON-object mode.

    The schema should be a JSON Schema object; use type ["<t>", "null"] for
    nullable fields (portable across OpenAI and Gemini via LiteLLM). Strict
    mode requires additionalProperties:false and every property in "required".
    """
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": schema},
    }


# Model prefixes that batch through the LiteLLM Batch API (OpenAI-compatible).
# Gemini is handled separately (native google-genai path) — see _infer_batch_provider.
_BATCH_PROVIDERS: dict[str, str] = {
    "openai": "openai",
    "azure": "azure",
}


def _to_gemini_schema(schema: dict) -> dict:
    """
    Adapt a JSON Schema (the form call() passes to OpenAI) for the google-genai
    batch path's `response_schema` field.

    The Gemini batch inlined-request path mishandles the newer
    `response_json_schema` field for nested schemas — arrays of objects and
    arrays of arrays collapse to their property names — so batches use
    `response_schema`, which follows the OpenAPI subset. That subset has no
    `additionalProperties`, so strip it recursively; every other keyword we use
    (type, properties, required, items) maps directly.
    """
    if isinstance(schema, dict):
        return {
            k: _to_gemini_schema(v)
            for k, v in schema.items()
            if k != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_to_gemini_schema(v) for v in schema]
    return schema

# Gemini batch job states (google-genai) normalized to the LiteLLM/OpenAI status
# vocabulary so batch_status/batch_call/batch_retrieve treat both providers alike.
# Any state not listed maps to 'in_progress'. PARTIALLY_SUCCEEDED is treated as
# 'completed'; per-request errors then surface in batch_retrieve, as with OpenAI.
_GEMINI_STATUS: dict[str, str] = {
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


def _infer_batch_provider(model: str) -> str:
    """
    Return the batch provider for `model`, or raise ValueError.

    'openai'/'azure' route through the LiteLLM Batch API; 'gemini' (Google AI
    Studio, GEMINI_API_KEY) is handled natively via the google-genai SDK, since
    LiteLLM does not wrap the Gemini batch API.
    """
    if "/" in model:
        prefix = model.split("/")[0].lower()
        if prefix in _BATCH_PROVIDERS:
            return _BATCH_PROVIDERS[prefix]
        if prefix == "gemini":
            return "gemini"
    lower = model.lower()
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if lower.startswith("gemini"):
        return "gemini"
    if lower.startswith("azure/"):
        return "azure"
    raise ValueError(
        f"batch_call() is not supported for model '{model}'. "
        f"Supported providers: openai, azure, gemini. "
        f"Use call() for sequential execution."
    )

_default_model: str | None = None
_default_api_base: str | None = None
_default_api_key: str | None = None
_default_extra_body: dict[str, Any] | None = None

_default_embedding_model: str | None = None
_default_embedding_api_base: str | None = None
_default_embedding_api_key: str | None = None


def set_default_model(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> None:
    """Set the default completion model (and optional VLLM endpoint) for all
    subsequent LLM calls. extra_body is forwarded as-is to litellm.completion()
    on every call — for endpoint-specific parameters an OpenAI-compatible
    server accepts outside the standard schema (e.g. a self-hosted vLLM
    model's chat_template_kwargs)."""
    global _default_model, _default_api_base, _default_api_key, _default_extra_body
    _default_model = model
    _default_api_base = api_base
    _default_api_key = api_key
    _default_extra_body = extra_body


def set_embedding_model(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    """Set the default embedding model (and optional VLLM endpoint) for all subsequent embed calls."""
    global _default_embedding_model, _default_embedding_api_base, _default_embedding_api_key
    _default_embedding_model = model
    _default_embedding_api_base = api_base
    _default_embedding_api_key = api_key


class LLMClient:
    DEFAULT_MODEL = "gpt-5.5"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model or _default_model or os.getenv("SIMSTUDY_MODEL", self.DEFAULT_MODEL)
        self.api_base = api_base or _default_api_base or os.getenv("SIMSTUDY_API_BASE")
        self.api_key = api_key or _default_api_key or os.getenv("SIMSTUDY_API_KEY")
        self.extra_body = extra_body or _default_extra_body

        self.embedding_model = (
            _default_embedding_model
            or os.getenv("SIMSTUDY_EMBEDDING_MODEL", self.DEFAULT_EMBEDDING_MODEL)
        )
        self.embedding_api_base = _default_embedding_api_base or os.getenv("SIMSTUDY_EMBEDDING_API_BASE")
        self.embedding_api_key = _default_embedding_api_key or os.getenv("SIMSTUDY_EMBEDDING_API_KEY")

    def call(
        self,
        prompt: str,
        json_mode: bool = True,
        max_retries: int = 3,
        schema: dict | None = None,
    ) -> Any:
        """
        Call the LLM with the given prompt.

        Returns a parsed dict when json_mode=True, or a raw string when False.
        When json_mode and a `schema` (JSON Schema dict) is given, the model is
        constrained to emit a matching object — the reliable way to get JSON
        from Gemini 3 thinking models (see _json_response_format).

        Retries up to max_retries times: API/network errors back off
        exponentially; a parse failure retries immediately (defensive fallback
        for schema-less json_mode).
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": 600,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if json_mode:
            kwargs["response_format"] = _json_response_format(schema)

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = litellm.completion(**kwargs)
                text = resp.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

            if not json_mode:
                if text is not None:
                    return text
                last_exc = RuntimeError("Model returned null message content")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            try:
                return _extract_json(text)
            except ValueError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    def _provider_kwargs(self) -> dict[str, Any]:
        """Shared kwargs (provider + api_key) for all LiteLLM batch API calls."""
        provider = _infer_batch_provider(self.model)
        kwargs: dict[str, Any] = {"custom_llm_provider": provider}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    # -- Gemini native batch (google-genai) -----------------------------------
    # LiteLLM does not wrap the Gemini batch API, so these methods talk to the
    # google-genai SDK directly. Requests are sent inline (no file upload);
    # results come back in input order in job.dest.inlined_responses.

    def _gemini_client(self) -> Any:
        from google import genai  # lazy: only needed for Gemini batch

        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=api_key)

    def _gemini_batch_submit(
        self,
        prompts: list[str],
        json_mode: bool,
        schema: dict | None,
    ) -> str:
        from google.genai import types

        model_name = self.model.split("/", 1)[-1] if "/" in self.model else self.model

        config_kwargs: dict[str, Any] = {}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            if schema is not None:
                config_kwargs["response_schema"] = _to_gemini_schema(schema)
        config = types.GenerateContentConfig(**config_kwargs)

        requests = [
            types.InlinedRequest(
                contents=[types.Content(role="user", parts=[types.Part(text=p)])],
                config=config,
            )
            for p in prompts
        ]

        client = self._gemini_client()
        job = client.batches.create(
            model=model_name,
            src=requests,
            config=types.CreateBatchJobConfig(display_name="interviewplayground-eval"),
        )
        print(f"[batch_submit] Submitted batch_id={job.name!r} ({len(prompts)} requests)")
        return job.name

    def _gemini_get_job(self, batch_id: str) -> Any:
        """Fetch a Gemini batch job, retrying transient errors up to 3 times."""
        client = self._gemini_client()
        for attempt in range(3):
            try:
                return client.batches.get(name=batch_id)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Failed to retrieve batch {batch_id} after 3 attempts: {exc}"
                    ) from exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Unreachable: _gemini_get_job({batch_id})")  # pragma: no cover

    def _gemini_batch_status(self, batch_id: str) -> str:
        job = self._gemini_get_job(batch_id)
        state = job.state.name if job.state else "JOB_STATE_UNSPECIFIED"
        return _GEMINI_STATUS.get(state, "in_progress")

    def _gemini_batch_retrieve(self, batch_id: str, json_mode: bool) -> list[Any]:
        job = self._gemini_get_job(batch_id)
        state = job.state.name if job.state else "JOB_STATE_UNSPECIFIED"
        status = _GEMINI_STATUS.get(state, "in_progress")
        if status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not ready (status '{status}', state {state}). "
                f"Call batch_status() and only retrieve once it is 'completed'."
            )

        responses = job.dest.inlined_responses if job.dest else None
        if not responses:
            raise RuntimeError(
                f"Batch {batch_id} completed but returned no inlined responses."
            )

        # A batch of thousands of requests occasionally has a handful come back
        # errored or cancelled server-side even though the batch job itself
        # completed successfully. Losing every other result over a few bad
        # requests would be far worse than the requests themselves — so a
        # failure here becomes an empty dict/string placeholder (the caller's
        # assembly logic already treats a missing key as "no judgment for this
        # one" rather than crashing) instead of aborting the whole batch.
        results: list[Any] = []
        failed_indices: list[int] = []
        for i, item in enumerate(responses):
            text = None if item.error else (item.response.text if item.response else None)
            if text is None:
                failed_indices.append(i)
                results.append("" if not json_mode else {})
                continue
            if not json_mode:
                results.append(text)
            else:
                try:
                    results.append(json.loads(text))
                except json.JSONDecodeError:
                    try:
                        results.append(_extract_json(text))
                    except ValueError:
                        failed_indices.append(i)
                        results.append({})
        if failed_indices:
            shown = failed_indices[:20]
            more = f" (+{len(failed_indices) - 20} more)" if len(failed_indices) > 20 else ""
            print(f"[batch_retrieve] WARNING: {len(failed_indices)}/{len(responses)} request(s) "
                  f"in batch {batch_id} failed or were unparseable — indices {shown}{more}. "
                  f"Substituting empty results for those; the rest are unaffected.")
        return results

    def batch_submit(
        self,
        prompts: list[str],
        json_mode: bool = True,
        schema: dict | None = None,
    ) -> str:
        """
        Submit prompts as a single batch job and return the batch_id immediately.

        Does NOT wait for completion. Pair with batch_status() to poll and
        batch_retrieve() to fetch results (possibly in a later process/day).

        openai/azure models use the LiteLLM Batch API; gemini models use the
        native Gemini batch API via google-genai. Raises ValueError for
        unsupported models, or (openai/azure only) when api_base is set.

        A `schema` (JSON Schema dict) constrains every request to a matching
        object; all prompts in one batch must share the same schema.
        """
        if not prompts:
            raise ValueError("batch_submit() requires a non-empty prompts list.")

        if _infer_batch_provider(self.model) == "gemini":
            return self._gemini_batch_submit(prompts, json_mode, schema)

        if self.api_base:
            raise ValueError(
                "batch_submit() is not supported when api_base is set (local/custom endpoints). "
                "Use call() for sequential execution."
            )

        model_name = self.model.split("/", 1)[-1] if "/" in self.model else self.model

        lines = []
        for i, prompt in enumerate(prompts):
            body: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if json_mode:
                body["response_format"] = _json_response_format(schema)
            lines.append(json.dumps({
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))
        jsonl_bytes = "\n".join(lines).encode("utf-8")

        shared = self._provider_kwargs()
        file_obj = litellm.create_file(
            file=("batch_requests.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
            **shared,
        )

        batch = litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            **shared,
        )
        print(f"[batch_submit] Submitted batch_id={batch.id!r} ({len(prompts)} requests)")
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        """
        Return the current status string for a batch (e.g. 'validating',
        'in_progress', 'completed', 'failed', 'expired', 'cancelled').

        Retries transient retrieval errors up to 3 times before raising.
        """
        if _infer_batch_provider(self.model) == "gemini":
            return self._gemini_batch_status(batch_id)

        shared = self._provider_kwargs()
        for attempt in range(3):
            try:
                status_obj = litellm.retrieve_batch(batch_id=batch_id, **shared)
                return status_obj.status
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Failed to retrieve status for batch {batch_id} after 3 attempts: {exc}"
                    ) from exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Unreachable: batch_status({batch_id})")  # pragma: no cover

    def batch_retrieve(
        self,
        batch_id: str,
        json_mode: bool = True,
    ) -> list[Any]:
        """
        Fetch and parse the results of a completed batch, in input order.

        Raises RuntimeError if the batch is not yet completed (call this only
        after batch_status() returns 'completed'), or if it failed/expired.
        Returns a parsed dict per prompt when json_mode=True, raw string when False.
        """
        if _infer_batch_provider(self.model) == "gemini":
            return self._gemini_batch_retrieve(batch_id, json_mode)

        shared = self._provider_kwargs()

        status_obj = None
        for attempt in range(3):
            try:
                status_obj = litellm.retrieve_batch(batch_id=batch_id, **shared)
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Failed to retrieve batch {batch_id} after 3 attempts: {exc}"
                    ) from exc
                time.sleep(5 * (attempt + 1))

        status = status_obj.status
        if status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not ready (status '{status}'). "
                f"Call batch_status() and only retrieve once it is 'completed'."
            )

        file_response = litellm.file_content(
            file_id=status_obj.output_file_id,
            **shared,
        )
        raw_text = file_response.content.decode("utf-8")
        output_by_id: dict[str, Any] = {}
        for line in raw_text.split("\n"):
            if line.strip():
                item = json.loads(line)
                output_by_id[item["custom_id"]] = item

        # Reconstruct input order from the numeric suffix of each custom_id.
        indices = sorted(int(cid.split("-")[1]) for cid in output_by_id)
        results: list[Any] = []
        for i in indices:
            item = output_by_id[f"req-{i}"]
            resp = item["response"]
            if resp["status_code"] != 200:
                raise RuntimeError(
                    f"Batch request req-{i} failed with status {resp['status_code']}: "
                    f"{resp.get('body', {})}"
                )
            text = resp["body"]["choices"][0]["message"]["content"]
            if not json_mode:
                results.append(text)
            else:
                try:
                    results.append(json.loads(text))
                except json.JSONDecodeError:
                    results.append(_extract_json(text))
        return results

    def _concurrent_call(
        self,
        prompts: list[str],
        json_mode: bool,
        schema: dict | None,
        max_concurrency: int | None = None,
    ) -> list[Any]:
        """Run prompts concurrently via call(), preserving input order.

        Fallback for OpenAI-compatible endpoints (e.g. a local vLLM server) that
        don't expose a provider batch API. call() carries its own retries.
        Concurrency defaults to the LLM_BATCH_CONCURRENCY env var (else 16); set
        it to roughly the server's --max-num-seqs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if max_concurrency is None:
            max_concurrency = int(os.getenv("LLM_BATCH_CONCURRENCY", "16"))
        results: list[Any] = [None] * len(prompts)
        workers = max(1, min(max_concurrency, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.call, p, json_mode=json_mode, schema=schema): i
                    for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

    def batch_call(
        self,
        prompts: list[str],
        json_mode: bool = True,
        poll_interval: int = 30,
        max_wait: int = 86400,
        schema: dict | None = None,
    ) -> list[Any]:
        """
        Blocking convenience wrapper: submit a batch, poll until complete, and
        return results in input order.

        Equivalent to batch_submit() followed by polling batch_status() and then
        batch_retrieve(). For submit-now / retrieve-later workflows, use those
        three methods directly instead.

        A `schema` (JSON Schema dict) constrains every request; all prompts in
        one batch must share the same schema.

        Raises RuntimeError if the batch fails, and TimeoutError if max_wait is
        exceeded. Does not fall back to sequential — use call() for that.
        """
        if not prompts:
            return []

        # Custom/local OpenAI-compatible endpoints (e.g. vLLM) do not offer the
        # provider batch APIs; run the prompts concurrently via call() instead.
        if self.api_base:
            return self._concurrent_call(prompts, json_mode, schema)

        batch_id = self.batch_submit(prompts, json_mode=json_mode, schema=schema)
        print(f"[batch_call] Polling batch {batch_id} every {poll_interval}s (max {max_wait}s)")

        start = time.monotonic()
        while True:
            time.sleep(poll_interval)
            status = self.batch_status(batch_id)
            if status == "completed":
                break
            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch {batch_id} ended with status '{status}'")
            if time.monotonic() - start > max_wait:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {max_wait}s"
                )

        return self.batch_retrieve(batch_id, json_mode=json_mode)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return an embedding vector for each text string."""
        kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "input": texts,
            "timeout": 30,
        }
        if self.embedding_api_base:
            kwargs["api_base"] = self.embedding_api_base
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key
        response = litellm.embedding(**kwargs)
        return [
            item["embedding"] if isinstance(item, dict) else item.embedding
            for item in response.data
        ]
