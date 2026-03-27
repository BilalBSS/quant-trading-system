# / tests for social sentiment (stocktwits + fear & greed)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.social_sentiment import (
    compute_social_score,
    fetch_fear_greed_index,
    fetch_reddit_sentiment,
    fetch_stocktwits_sentiment,
    run_social_sentiment,
    store_social_sentiment,
)


def _mock_pool():
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_pool, mock_conn


def _stocktwits_response(bullish: int, bearish: int, msg_count: int = 5):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "symbol": {
            "sentiments": {"bullish": bullish, "bearish": bearish},
        },
        "messages": [{"id": i} for i in range(msg_count)],
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _fng_response(value: str):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"value": value, "value_classification": "Neutral"}],
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestFetchStockTwitsSentiment:
    @pytest.mark.asyncio
    async def test_returns_bullish_data(self):
        resp = _stocktwits_response(80, 20, msg_count=10)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("AAPL")
            assert result is not None
            assert result["bullish_pct"] == pytest.approx(0.8)
            assert result["bearish_pct"] == pytest.approx(0.2)
            assert result["volume"] == 10
            assert result["raw_score"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_returns_bearish_data(self):
        resp = _stocktwits_response(10, 90)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("TSLA")
            assert result is not None
            assert result["bullish_pct"] == pytest.approx(0.1)
            assert result["bearish_pct"] == pytest.approx(0.9)
            assert result["raw_score"] == pytest.approx(-0.8)

    @pytest.mark.asyncio
    async def test_even_split(self):
        resp = _stocktwits_response(50, 50)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("MSFT")
            assert result is not None
            assert result["bullish_pct"] == pytest.approx(0.5)
            assert result["bearish_pct"] == pytest.approx(0.5)
            assert result["raw_score"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_returns_none_no_sentiments(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"symbol": {}, "messages": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=mock_resp):
            result = await fetch_stocktwits_sentiment("UNKNOWN")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_zero_total(self):
        resp = _stocktwits_response(0, 0)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("XYZ")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, side_effect=Exception("404")):
            result = await fetch_stocktwits_sentiment("BAD")
            assert result is None

    @pytest.mark.asyncio
    async def test_all_bullish(self):
        resp = _stocktwits_response(100, 0)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("MOON")
            assert result is not None
            assert result["bullish_pct"] == pytest.approx(1.0)
            assert result["bearish_pct"] == pytest.approx(0.0)
            assert result["raw_score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_bearish(self):
        resp = _stocktwits_response(0, 100)
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_stocktwits_sentiment("DUMP")
            assert result is not None
            assert result["raw_score"] == pytest.approx(-1.0)


class TestFetchFearGreedIndex:
    @pytest.mark.asyncio
    async def test_neutral_value(self):
        resp = _fng_response("50")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["raw_value"] == pytest.approx(50.0)
            assert result["normalized"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_extreme_fear(self):
        resp = _fng_response("0")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["normalized"] == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_extreme_greed(self):
        resp = _fng_response("100")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["normalized"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_fear_region(self):
        # / value=25 -> (25-50)/50 = -0.5
        resp = _fng_response("25")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["normalized"] == pytest.approx(-0.5)

    @pytest.mark.asyncio
    async def test_greed_region(self):
        # / value=75 -> (75-50)/50 = 0.5
        resp = _fng_response("75")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["normalized"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_returns_none_no_data(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=mock_resp):
            result = await fetch_fear_greed_index()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=mock_resp):
            result = await fetch_fear_greed_index()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await fetch_fear_greed_index()
            assert result is None

    @pytest.mark.asyncio
    async def test_clamps_above_100(self):
        # / edge case: value > 100 should clamp to 1.0
        resp = _fng_response("110")
        with patch("src.data.social_sentiment.api_get", new_callable=AsyncMock, return_value=resp):
            result = await fetch_fear_greed_index()
            assert result is not None
            assert result["normalized"] <= 1.0


class TestFetchRedditSentiment:
    @pytest.mark.asyncio
    async def test_returns_none_placeholder(self):
        result = await fetch_reddit_sentiment("AAPL")
        assert result is None


class TestStoreSocialSentiment:
    @pytest.mark.asyncio
    async def test_stores_to_db(self):
        pool, conn = _mock_pool()
        await store_social_sentiment(pool, "AAPL", "stocktwits", 0.8, 0.2, 50, 0.6)
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "social_sentiment" in args[0]
        assert args[1] == "AAPL"
        assert args[3] == "stocktwits"

    @pytest.mark.asyncio
    async def test_stores_with_none_values(self):
        # / fear & greed stores with null bullish/bearish/volume
        pool, conn = _mock_pool()
        await store_social_sentiment(pool, "BTC-USD", "fear_greed", None, None, None, 0.3)
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert args[4] is None  # bullish_pct
        assert args[5] is None  # bearish_pct
        assert args[6] is None  # volume

    @pytest.mark.asyncio
    async def test_upsert_query(self):
        pool, conn = _mock_pool()
        await store_social_sentiment(pool, "ETH-USD", "stocktwits", 0.5, 0.5, 10, 0.0)
        sql = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql


class TestComputeSocialScore:
    @pytest.mark.asyncio
    async def test_both_sources(self):
        st_data = {"bullish_pct": 0.7, "bearish_pct": 0.3, "volume": 5, "raw_score": 0.4}
        fng_data = {"raw_value": 75.0, "normalized": 0.5}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                score = await compute_social_score("AAPL")
                # / weighted: (0.4 * 0.6 + 0.5 * 0.4) / 1.0 = 0.44
                assert score == pytest.approx(0.44)

    @pytest.mark.asyncio
    async def test_stocktwits_only(self):
        st_data = {"bullish_pct": 0.9, "bearish_pct": 0.1, "volume": 10, "raw_score": 0.8}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=None):
                score = await compute_social_score("TSLA")
                # / only stocktwits: 0.8 * 0.6 / 0.6 = 0.8
                assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_fng_only(self):
        fng_data = {"raw_value": 25.0, "normalized": -0.5}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=None):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                score = await compute_social_score("SPY")
                # / only fng: -0.5 * 0.4 / 0.4 = -0.5
                assert score == pytest.approx(-0.5)

    @pytest.mark.asyncio
    async def test_no_sources_returns_zero(self):
        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=None):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=None):
                score = await compute_social_score("NOPE")
                assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_clamped(self):
        # / extreme values still bounded
        st_data = {"bullish_pct": 1.0, "bearish_pct": 0.0, "volume": 100, "raw_score": 1.0}
        fng_data = {"raw_value": 100.0, "normalized": 1.0}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                score = await compute_social_score("MOON")
                assert -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_negative_score(self):
        st_data = {"bullish_pct": 0.1, "bearish_pct": 0.9, "volume": 20, "raw_score": -0.8}
        fng_data = {"raw_value": 10.0, "normalized": -0.8}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                score = await compute_social_score("DUMP")
                # / ((-0.8 * 0.6) + (-0.8 * 0.4)) / 1.0 = -0.8
                assert score == pytest.approx(-0.8)


class TestRunSocialSentiment:
    @pytest.mark.asyncio
    async def test_processes_multiple_symbols(self):
        pool, conn = _mock_pool()
        st_data = {"bullish_pct": 0.6, "bearish_pct": 0.4, "volume": 5, "raw_score": 0.2}
        fng_data = {"raw_value": 60.0, "normalized": 0.2}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                results = await run_social_sentiment(pool, ["AAPL", "MSFT"])
                assert len(results) == 2
                assert "AAPL" in results
                assert "MSFT" in results

    @pytest.mark.asyncio
    async def test_stores_stocktwits_and_fng(self):
        pool, conn = _mock_pool()
        st_data = {"bullish_pct": 0.7, "bearish_pct": 0.3, "volume": 10, "raw_score": 0.4}
        fng_data = {"raw_value": 50.0, "normalized": 0.0}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                await run_social_sentiment(pool, ["AAPL"])
                # / should store both stocktwits and fear_greed
                assert conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_fetch_failure_gracefully(self):
        pool, conn = _mock_pool()

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=None):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=None):
                results = await run_social_sentiment(pool, ["AAPL"])
                assert results["AAPL"] == 0.0

    @pytest.mark.asyncio
    async def test_continues_on_symbol_error(self):
        pool, conn = _mock_pool()

        call_count = 0

        async def failing_then_ok(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("api down")
            return {"bullish_pct": 0.6, "bearish_pct": 0.4, "volume": 5, "raw_score": 0.2}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", side_effect=failing_then_ok):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=None):
                results = await run_social_sentiment(pool, ["BAD", "GOOD"])
                assert results["BAD"] == 0.0
                assert "GOOD" in results

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self):
        pool, conn = _mock_pool()
        results = await run_social_sentiment(pool, [])
        assert results == {}

    @pytest.mark.asyncio
    async def test_fng_fetched_once(self):
        pool, conn = _mock_pool()
        st_data = {"bullish_pct": 0.5, "bearish_pct": 0.5, "volume": 1, "raw_score": 0.0}
        fng_data = {"raw_value": 50.0, "normalized": 0.0}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data) as mock_fng:
                await run_social_sentiment(pool, ["AAPL", "MSFT", "GOOG"])
                # / fear & greed is market-wide, fetched once not per-symbol
                mock_fng.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_computation_matches(self):
        pool, conn = _mock_pool()
        st_data = {"bullish_pct": 0.8, "bearish_pct": 0.2, "volume": 20, "raw_score": 0.6}
        fng_data = {"raw_value": 70.0, "normalized": 0.4}

        with patch("src.data.social_sentiment.fetch_stocktwits_sentiment", new_callable=AsyncMock, return_value=st_data):
            with patch("src.data.social_sentiment.fetch_fear_greed_index", new_callable=AsyncMock, return_value=fng_data):
                results = await run_social_sentiment(pool, ["AAPL"])
                # / (0.6 * 0.6 + 0.4 * 0.4) / 1.0 = 0.52
                assert results["AAPL"] == pytest.approx(0.52)
