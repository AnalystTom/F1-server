# tests/test_gemini_agent.py
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.main  # the file that defines GeminiAgent, _agent and app


# ---------- helpers ----------
def _patch_gemini(monkeypatch, *, sync_text="stub-sync", stream_tokens=("tok-1", "tok-2")):
    """Replace the underlying genai client with a dummy that returns fixed data."""

    def _fake_generate_content(*_, **__):
        return SimpleNamespace(text=sync_text)

    def _fake_generate_content_stream(*_, **__):
        for t in stream_tokens:
            yield SimpleNamespace(text=t)

    dummy_models = SimpleNamespace(
        generate_content=_fake_generate_content,
        generate_content_stream=_fake_generate_content_stream,
    )
    dummy_client = SimpleNamespace(models=dummy_models)

    # patch the *instance* already constructed inside main._agent
    monkeypatch.setattr(src.main._agent, "client", dummy_client, raising=False)
    return stream_tokens, sync_text


# ---------- unit tests for the helper class ----------
def test_generate_response(monkeypatch):
    _, expected_text = _patch_gemini(monkeypatch, sync_text="hello-world")
    assert src.main._agent.generate_response("who?") == expected_text


@pytest.mark.asyncio
async def test_generate_response_stream(monkeypatch):
    tokens, _ = _patch_gemini(monkeypatch)
    got = []
    async for tok in src.main._agent.generate_response_stream("stream?"):
        got.append(tok)
    assert got == list(tokens)


# ---------- fastapi route tests ----------
client = TestClient(src.main.app)


def test_chat_endpoint(monkeypatch):
    _patch_gemini(monkeypatch, sync_text="route-ok")
    resp = client.post("/chat", json={"query": "ping"})
    assert resp.status_code == 200
    assert resp.json() == {"response": "route-ok"}


def test_chat_stream_endpoint(monkeypatch):
    tokens, _ = _patch_gemini(monkeypatch, stream_tokens=("a", "b", "c"))

    # using TestClient.stream to consume the SSE response
    with client.stream("POST", "/chat/stream", json={"query": "ping"}) as s:
        body = b"".join(list(s.iter_bytes()))
    # The EventSourceResponse separator is two newlines ("\n\n") between events
    raw_events = [chunk for chunk in body.split(b"\n\n") if chunk]
    streamed_data = [evt.split(b"data: ")[1].decode() for evt in raw_events]

    assert streamed_data == list(tokens)
