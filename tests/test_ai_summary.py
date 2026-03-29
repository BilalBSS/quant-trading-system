# / tests for ai summary generation

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.ai_summary import (
    AnalysisSummary,
    _build_fallback_summary,
    _build_prompt,
    generate_summary,
)
from src.analysis.dcf_model import DCFResult
from src.analysis.earnings_signals import EarningsSignal
from src.analysis.insider_activity import InsiderSignal
from src.analysis.ratio_analysis import RatioScore


def _sample_ratio() -> RatioScore:
    return RatioScore(
        symbol="AAPL", date=date(2026, 3, 25),
        pe_score=75.0, ps_score=65.0, peg_score=70.0,
        fcf_margin_score=80.0, debt_equity_score=85.0,
        composite_score=74.0,
    )


def _sample_dcf() -> DCFResult:
    return DCFResult(
        symbol="AAPL", date=date(2026, 3, 25),
        fair_value_median=220.0, fair_value_p10=180.0, fair_value_p90=270.0,
        current_price=190.0, upside_pct=0.1579,
        num_simulations=10000, confidence="medium",
    )


def _sample_earnings() -> EarningsSignal:
    return EarningsSignal(
        symbol="AAPL", date=date(2026, 3, 25),
        signal="bullish", strength=65.0,
        surprise_pct=0.12, consecutive_beats=3,
        avg_surprise_4q=0.10,
    )


def _sample_insider() -> InsiderSignal:
    return InsiderSignal(
        symbol="AAPL", date=date(2026, 3, 25),
        signal="bullish", strength=55.0,
        net_buy_ratio=0.7, total_buys=5, total_sells=1,
        buy_value=500000.0, sell_value=80000.0,
        cluster_detected=True, unique_buyers=4, unique_sellers=1,
    )


class TestBuildPrompt:
    def test_includes_symbol(self):
        prompt = _build_prompt("AAPL", None, None, None, None)
        assert "AAPL" in prompt

    def test_includes_ratio_data(self):
        prompt = _build_prompt("AAPL", _sample_ratio(), None, None, None)
        assert "74.0" in prompt
        assert "Overall ratio score" in prompt

    def test_includes_dcf_data(self):
        prompt = _build_prompt("AAPL", None, _sample_dcf(), None, None)
        assert "220.00" in prompt
        assert "190.00" in prompt

    def test_includes_earnings(self):
        prompt = _build_prompt("AAPL", None, None, _sample_earnings(), None)
        assert "bullish" in prompt
        assert "3" in prompt

    def test_includes_insider(self):
        prompt = _build_prompt("AAPL", None, None, None, _sample_insider())
        assert "0.70" in prompt

    def test_includes_regime(self):
        prompt = _build_prompt("AAPL", None, None, None, None, regime="bull")
        assert "bull" in prompt


class TestBuildPromptDeep:
    def test_all_components_together(self):
        # / all data present should produce valid prompt with all sections
        prompt = _build_prompt(
            "AAPL", _sample_ratio(), _sample_dcf(),
            _sample_earnings(), _sample_insider(), regime="bull",
        )
        assert "AAPL" in prompt
        assert "bull" in prompt
        assert "74.0" in prompt     # ratio composite
        assert "220.00" in prompt   # dcf fair value
        assert "bullish" in prompt  # earnings signal
        assert "0.70" in prompt     # insider net buy ratio
        assert "Cluster" in prompt  # cluster detected

    def test_none_inputs_dont_crash(self):
        # / all None inputs should not raise
        prompt = _build_prompt("NONE", None, None, None, None, regime=None)
        assert "NONE" in prompt
        assert len(prompt) > 0


class TestBuildFallbackSummaryDeep:
    def test_cluster_detected_adds_extra_line(self):
        insider = InsiderSignal(
            symbol="CLU", date=date(2026, 3, 25),
            signal="bullish", strength=60.0,
            net_buy_ratio=0.8, total_buys=5, total_sells=0,
            buy_value=500000.0, sell_value=0.0,
            cluster_detected=True, unique_buyers=5, unique_sellers=0,
        )
        result = _build_fallback_summary("CLU", None, None, None, insider)
        assert "cluster" in result.summary.lower()

    def test_confidence_is_average_of_strengths(self):
        # / ratio composite=70, earnings strength=60 -> avg
        ratio = RatioScore(symbol="AVG", date=date(2026, 3, 25), composite_score=70.0)
        earnings = EarningsSignal(
            symbol="AVG", date=date(2026, 3, 25),
            signal="bullish", strength=60.0,
            surprise_pct=0.10, consecutive_beats=2, avg_surprise_4q=0.08,
        )
        result = _build_fallback_summary("AVG", ratio, None, earnings, None)
        # / total_strength = 70.0 + 60.0 = 130.0, signal_count = 2
        # / avg_strength = 130.0 / 2 = 65.0
        assert result.confidence == 65.0

    def test_all_data_present_calculates_correctly(self):
        result = _build_fallback_summary(
            "FULL", _sample_ratio(), _sample_dcf(),
            _sample_earnings(), _sample_insider(),
        )
        assert result.signal == "bullish"
        assert result.confidence > 0
        # / 4 signals present, all bullish
        # / ratio: 74.0, dcf: min(100, 0.1579*200)=31.58, earnings: 65.0, insider: 55.0
        # / total = 74.0 + 31.58 + 65.0 + 55.0 = 225.58
        # / avg = 225.58 / 4 = 56.395 -> 56.4
        expected = round((74.0 + min(100.0, abs(0.1579) * 200) + 65.0 + 55.0) / 4, 1)
        assert result.confidence == expected


class TestBuildFallbackSummary:
    def test_bullish_summary(self):
        result = _build_fallback_summary(
            "AAPL", _sample_ratio(), _sample_dcf(),
            _sample_earnings(), _sample_insider(),
        )
        assert result.signal == "bullish"
        assert result.symbol == "AAPL"
        assert "AAPL" in result.summary
        assert result.model_used is None

    def test_bearish_summary(self):
        bearish_ratio = RatioScore(
            symbol="BAD", date=date(2026, 3, 25),
            composite_score=25.0,
        )
        bearish_dcf = DCFResult(
            symbol="BAD", date=date(2026, 3, 25),
            fair_value_median=50.0, fair_value_p10=30.0, fair_value_p90=70.0,
            current_price=100.0, upside_pct=-0.50,
            num_simulations=1000, confidence="medium",
        )
        bearish_earnings = EarningsSignal(
            symbol="BAD", date=date(2026, 3, 25),
            signal="bearish", strength=70.0,
            surprise_pct=-0.20, consecutive_beats=-3,
            avg_surprise_4q=-0.15,
        )
        result = _build_fallback_summary("BAD", bearish_ratio, bearish_dcf, bearish_earnings, None)
        assert result.signal == "bearish"

    def test_neutral_when_mixed(self):
        bullish_ratio = RatioScore(symbol="MIX", date=date(2026, 3, 25), composite_score=70.0)
        bearish_earnings = EarningsSignal(
            symbol="MIX", date=date(2026, 3, 25),
            signal="bearish", strength=60.0,
            surprise_pct=-0.10, consecutive_beats=-2, avg_surprise_4q=-0.08,
        )
        result = _build_fallback_summary("MIX", bullish_ratio, None, bearish_earnings, None)
        assert result.signal == "neutral"

    def test_no_data(self):
        result = _build_fallback_summary("EMPTY", None, None, None, None)
        assert result.signal == "neutral"
        assert result.confidence == 0.0


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_uses_fallback_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await generate_summary(
                "AAPL", ratio=_sample_ratio(), dcf=_sample_dcf(),
            )
            assert result.model_used is None
            assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_uses_groq_when_available(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Bullish. AAPL shows strong fundamentals."}}],
            "model": "llama-3.1-8b-instant",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await generate_summary(
                    "AAPL", ratio=_sample_ratio(),
                )
                assert result.model_used == "llama-3.1-8b-instant"
                assert result.signal == "bullish"

    @pytest.mark.asyncio
    async def test_bearish_in_first_50_chars(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Bearish. AAPL faces headwinds from declining margins and revenue slowdown."}}],
            "model": "llama-3.1-8b-instant",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await generate_summary("AAPL", ratio=_sample_ratio())
                assert result.signal == "bearish"

    @pytest.mark.asyncio
    async def test_neutral_when_neither_bullish_nor_bearish(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Mixed signals. AAPL has balanced fundamentals with no clear direction."}}],
            "model": "llama-3.1-8b-instant",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await generate_summary("AAPL", ratio=_sample_ratio())
                assert result.signal == "neutral"

    @pytest.mark.asyncio
    async def test_falls_back_on_api_error(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", side_effect=Exception("connection error")):
                result = await generate_summary(
                    "AAPL", ratio=_sample_ratio(), dcf=_sample_dcf(),
                )
                # / should fall back gracefully
                assert result.model_used is None
                assert result.symbol == "AAPL"


# ---------------------------------------------------------------------------
# / generate_daily_synthesis tests
# ---------------------------------------------------------------------------

class TestGenerateDailySynthesis:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_none(self):
        from src.analysis.ai_summary import generate_daily_synthesis
        pool = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            result = await generate_daily_synthesis(pool, ["AAPL", "MSFT"])
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_deepseek_reasoner(self):
        from src.analysis.ai_summary import generate_daily_synthesis

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"symbol": "AAPL", "composite_score": 75.0, "regime": "bull", "ai_consensus": "bullish"},
        ]
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_conn
        mock_ctx.__aexit__.return_value = False
        pool = MagicMock()
        pool.acquire.return_value = mock_ctx

        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"top_buys": [{"symbol": "AAPL"}], "top_avoids": [], "portfolio_risk": "low", "per_symbol_notes": {}}'}}],
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("src.agents.tools.store_daily_synthesis", new_callable=AsyncMock, return_value=1):
                    result = await generate_daily_synthesis(pool, ["AAPL"])

        assert result is not None
        assert result["top_buys"] == [{"symbol": "AAPL"}]
        # / verify it called the deepseek api
        call_kwargs = mock_client.post.call_args
        assert "deepseek" in call_kwargs[0][0] or "deepseek" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_stores_to_db(self):
        from src.analysis.ai_summary import generate_daily_synthesis

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"symbol": "AAPL", "composite_score": 60.0, "regime": "sideways", "ai_consensus": "neutral"},
        ]
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_conn
        mock_ctx.__aexit__.return_value = False
        pool = MagicMock()
        pool.acquire.return_value = mock_ctx

        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"top_buys": [], "top_avoids": [], "portfolio_risk": "none", "per_symbol_notes": {}}'}}],
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                with patch("src.agents.tools.store_daily_synthesis", new_callable=AsyncMock, return_value=1) as mock_store:
                    await generate_daily_synthesis(pool, ["AAPL"])
                    mock_store.assert_called_once()
