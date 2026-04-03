# / tests for shared llm http client (groq + deepseek)

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import src.data.llm_client as mod


@pytest.fixture(autouse=True)
def _reset_clients():
    # / clean module-level state before and after each test
    mod._clients.clear()
    yield
    mod._clients.clear()


class TestGetLlmClient:
    @pytest.mark.asyncio
    async def test_groq_returns_client(self):
        client = await mod.get_llm_client("groq")
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_deepseek_returns_client(self):
        client = await mod.get_llm_client("deepseek")
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_separate_clients_per_provider(self):
        g = await mod.get_llm_client("groq")
        d = await mod.get_llm_client("deepseek")
        assert g is not d
        await g.aclose()
        await d.aclose()

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown llm provider"):
            await mod.get_llm_client("openai")

    @pytest.mark.asyncio
    async def test_recreates_client_if_closed(self):
        c1 = await mod.get_llm_client("groq")
        await c1.aclose()
        c2 = await mod.get_llm_client("groq")
        assert c2 is not c1
        assert not c2.is_closed
        await c2.aclose()


class TestCloseLlmClients:
    @pytest.mark.asyncio
    async def test_closes_all_and_clears(self):
        g = await mod.get_llm_client("groq")
        d = await mod.get_llm_client("deepseek")
        await mod.close_llm_clients()
        assert g.is_closed
        assert d.is_closed
        assert len(mod._clients) == 0

    @pytest.mark.asyncio
    async def test_noop_when_empty(self):
        # / should not raise when nothing to close
        await mod.close_llm_clients()
        assert len(mod._clients) == 0


class TestLlmCall:
    @pytest.mark.asyncio
    async def test_valid_call_makes_correct_request(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch.object(mod, "get_llm_client", return_value=mock_client):
                result = await mod.llm_call("groq", [{"role": "user", "content": "hi"}])

        assert result == {"choices": [{"message": {"content": "ok"}}]}
        call_kwargs = mock_client.post.call_args
        assert "chat/completions" in call_kwargs[0][0]
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"
        assert call_kwargs[1]["json"]["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="missing GROQ_API_KEY"):
                await mod.llm_call("groq", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown llm provider"):
            await mod.llm_call("openai", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_forwards_model_param(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False

        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            with patch.object(mod, "get_llm_client", return_value=mock_client):
                await mod.llm_call(
                    "groq",
                    [{"role": "user", "content": "x"}],
                    model="llama-3.1-8b-instant",
                )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "llama-3.1-8b-instant"

    @pytest.mark.asyncio
    async def test_forwards_timeout_param(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k"}):
            with patch.object(mod, "get_llm_client", return_value=mock_client):
                await mod.llm_call(
                    "deepseek",
                    [{"role": "user", "content": "x"}],
                    timeout=60.0,
                )

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_uses_default_timeout_when_none(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False

        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            with patch.object(mod, "get_llm_client", return_value=mock_client):
                await mod.llm_call("groq", [{"role": "user", "content": "x"}])

        call_kwargs = mock_client.post.call_args[1]
        # / groq default timeout is 15.0
        assert call_kwargs["timeout"] == 15.0
