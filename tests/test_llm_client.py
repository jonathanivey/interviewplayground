import pytest

from interviewplayground.llm_client import LLMClient, set_default_model


def _resp(content):
    class _Message:
        pass

    class _Choice:
        pass

    class _Resp:
        pass

    msg = _Message()
    msg.content = content
    choice = _Choice()
    choice.message = msg
    resp = _Resp()
    resp.choices = [choice]
    return resp


def test_call_retries_on_null_content_then_succeeds(mocker):
    mock_completion = mocker.patch("interviewplayground.llm_client.litellm.completion")
    mock_completion.side_effect = [_resp(None), _resp("real answer")]
    mocker.patch("interviewplayground.llm_client.time.sleep")

    client = LLMClient(model="test-model")
    result = client.call("prompt", json_mode=False)

    assert result == "real answer"
    assert mock_completion.call_count == 2


def test_call_raises_after_repeated_null_content(mocker):
    mock_completion = mocker.patch("interviewplayground.llm_client.litellm.completion")
    mock_completion.side_effect = [_resp(None), _resp(None), _resp(None)]
    mocker.patch("interviewplayground.llm_client.time.sleep")

    client = LLMClient(model="test-model")
    with pytest.raises(RuntimeError):
        client.call("prompt", json_mode=False, max_retries=3)

    assert mock_completion.call_count == 3


def test_call_forwards_extra_body_when_set(mocker):
    mock_completion = mocker.patch("interviewplayground.llm_client.litellm.completion")
    mock_completion.return_value = _resp("answer")

    client = LLMClient(model="test-model", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    client.call("prompt", json_mode=False)

    assert mock_completion.call_args.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_call_omits_extra_body_by_default(mocker):
    mock_completion = mocker.patch("interviewplayground.llm_client.litellm.completion")
    mock_completion.return_value = _resp("answer")

    client = LLMClient(model="test-model")
    client.call("prompt", json_mode=False)

    assert "extra_body" not in mock_completion.call_args.kwargs


def test_set_default_model_propagates_extra_body(mocker, monkeypatch):
    mock_completion = mocker.patch("interviewplayground.llm_client.litellm.completion")
    mock_completion.return_value = _resp("answer")
    # set_default_model() mutates module globals directly; monkeypatch restores
    # them after the test regardless of pass/fail, isolating other tests from it.
    monkeypatch.setattr("interviewplayground.llm_client._default_model", None)
    monkeypatch.setattr("interviewplayground.llm_client._default_api_base", None)
    monkeypatch.setattr("interviewplayground.llm_client._default_api_key", None)
    monkeypatch.setattr("interviewplayground.llm_client._default_extra_body", None)

    set_default_model("test-model", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    client = LLMClient()
    client.call("prompt", json_mode=False)

    assert mock_completion.call_args.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_batch_call_with_api_base_uses_concurrent_fallback(mocker):
    """A custom api_base (e.g. a local vLLM server) has no provider batch API,
    so batch_call() must fall back to threaded call()s instead of attempting
    batch_submit()."""
    client = LLMClient(model="openai/my-model", api_base="http://localhost:8000/v1")
    mock_call = mocker.patch.object(client, "call", side_effect=lambda p, **kw: {"i": p})
    mock_submit = mocker.patch.object(client, "batch_submit")

    results = client.batch_call(["a", "b", "c"])

    mock_submit.assert_not_called()
    assert mock_call.call_count == 3
    assert results == [{"i": "a"}, {"i": "b"}, {"i": "c"}]
