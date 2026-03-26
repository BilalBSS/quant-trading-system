# / tests for analyst agent

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analyst_agent import AnalystAgent
from src.analysis.ratio_analysis import RatioScore
from src.analysis.dcf_model import DCFResult
from src.analysis.earnings_signals import EarningsSignal
from src.analysis.insider_activity import InsiderSignal
from src.analysis.ai_summary import AnalysisSummary


# ---------------------------------------------------------------------------
# / helpers
# ---------------------------------------------------------------------------

def _mock_pool(mock_conn=None):
    if mock_conn is None:
        mock_conn = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


def _make_ratio(composite: float = 70.0) -> RatioScore:
    return RatioScore(
        symbol="AAPL", date=date.today(),
        pe_score=80.0, ps_score=60.0, peg_score=75.0,
        fcf_margin_score=65.0, debt_equity_score=70.0,
        composite_score=composite,
        details={
            "pe_ratio": "15.0", "ps_ratio": "5.0", "peg_ratio": "1.2",
            "fcf_margin": "0.20", "debt_to_equity": "0.5",
        },
    )


def _make_dcf(upside: float = 0.25) -> DCFResult:
    return DCFResult(
        symbol="AAPL", date=date.today(),
        fair_value_median=180.0, fair_value_p10=140.0, fair_value_p90=220.0,
        current_price=150.0, upside_pct=upside,
        num_simulations=10000, confidence="high",
    )


def _make_earnings(strength: float = 60.0) -> EarningsSignal:
    return EarningsSignal(
        symbol="AAPL", date=date.today(),
        signal="bullish", strength=strength,
        surprise_pct=0.08, consecutive_beats=3,
        avg_surprise_4q=0.06,
    )


def _make_insider(strength: float = 40.0) -> InsiderSignal:
    return InsiderSignal(
        symbol="AAPL", date=date.today(),
        signal="bullish", strength=strength,
        net_buy_ratio=0.6, total_buys=5, total_sells=2,
        buy_value=500000, sell_value=100000,
        cluster_detected=True, unique_buyers=4, unique_sellers=1,
    )


def _make_summary() -> AnalysisSummary:
    return AnalysisSummary(
        symbol="AAPL", date=date.today(),
        summary="AAPL looks bullish", model_used=None,
        signal="bullish", confidence=70.0,
    )


# ---------------------------------------------------------------------------
# / _compute_fundamental_score
# ---------------------------------------------------------------------------

class TestComputeFundamentalScore:
    def setup_method(self):
        self.agent = AnalystAgent()

    def test_all_components(self):
        # / hand-computed: ratio 70*0.35 + dcf_score*0.25 + earnings 60*0.20 + insider 40*0.20
        # / dcf upside 0.25 -> score = (0.25+0.5)/1.0*100 = 75
        # / weighted = 70*0.35 + 75*0.25 + 60*0.20 + 40*0.20 = 24.5+18.75+12+8 = 63.25
        # / total_weight = 1.0, result = 63.25 -> 63.2 (round to 1)
        score = self.agent._compute_fundamental_score(
            _make_ratio(70.0), _make_dcf(0.25), _make_earnings(60.0), _make_insider(40.0),
        )
        assert score == pytest.approx(63.2, abs=0.1)

    def test_partial_ratio_and_earnings(self):
        # / only ratio (70) and earnings (60)
        # / reweighted: ratio 70*0.35 + earnings 60*0.20 = 24.5+12 = 36.5
        # / total_weight = 0.35+0.20 = 0.55
        # / result = 36.5 / 0.55 = 66.36..
        score = self.agent._compute_fundamental_score(
            _make_ratio(70.0), None, _make_earnings(60.0), None,
        )
        assert score == pytest.approx(66.4, abs=0.1)

    def test_none_returns_none(self):
        score = self.agent._compute_fundamental_score(None, None, None, None)
        assert score is None

    def test_ratio_only(self):
        score = self.agent._compute_fundamental_score(_make_ratio(80.0), None, None, None)
        assert score == 80.0

    def test_dcf_upside_normalization_negative_50(self):
        # / upside = -0.50 -> score = (-0.50+0.5)/1.0*100 = 0
        score = self.agent._compute_fundamental_score(None, _make_dcf(-0.50), None, None)
        assert score == 0.0

    def test_dcf_upside_normalization_positive_50(self):
        # / upside = 0.50 -> score = (0.50+0.5)/1.0*100 = 100
        score = self.agent._compute_fundamental_score(None, _make_dcf(0.50), None, None)
        assert score == 100.0

    def test_dcf_upside_normalization_zero(self):
        # / upside = 0.0 -> score = (0.0+0.5)/1.0*100 = 50
        score = self.agent._compute_fundamental_score(None, _make_dcf(0.0), None, None)
        assert score == 50.0

    def test_dcf_upside_clamped_above(self):
        # / upside = 1.0 -> score = (1.0+0.5)/1.0*100 = 150 -> clamped to 100
        score = self.agent._compute_fundamental_score(None, _make_dcf(1.0), None, None)
        assert score == 100.0

    def test_dcf_upside_clamped_below(self):
        # / upside = -1.0 -> score = (-1.0+0.5)/1.0*100 = -50 -> clamped to 0
        score = self.agent._compute_fundamental_score(None, _make_dcf(-1.0), None, None)
        assert score == 0.0

    def test_ratio_none_composite_ignored(self):
        r = _make_ratio(70.0)
        r.composite_score = None
        score = self.agent._compute_fundamental_score(r, None, _make_earnings(60.0), None)
        # / only earnings contributes
        assert score == 60.0


# ---------------------------------------------------------------------------
# / _build_details
# ---------------------------------------------------------------------------

class TestBuildDetails:
    def setup_method(self):
        self.agent = AnalystAgent()

    def test_includes_all_fields(self):
        d = self.agent._build_details(
            _make_ratio(), _make_dcf(), _make_earnings(), _make_insider(), _make_summary(),
        )
        # / ratio fields
        assert "pe_ratio" in d
        assert "ps_ratio" in d
        assert "peg_ratio" in d
        assert "fcf_margin" in d
        assert "debt_to_equity" in d
        assert "ratio_composite" in d
        # / dcf fields
        assert "dcf_upside" in d
        assert "dcf_median" in d
        assert "dcf_confidence" in d
        # / earnings fields
        assert "earnings_surprise_pct" in d
        assert "consecutive_beats" in d
        assert "earnings_signal" in d
        # / insider fields
        assert "insider_net_buy_ratio" in d
        assert "insider_signal" in d
        # / summary fields
        assert "summary" in d
        assert "summary_signal" in d

    def test_empty_when_all_none(self):
        d = self.agent._build_details(None, None, None, None, None)
        assert d == {}


# ---------------------------------------------------------------------------
# / run (integration-level with mocks)
# ---------------------------------------------------------------------------

class TestAnalystAgentRun:
    def setup_method(self):
        self.agent = AnalystAgent()

    @pytest.mark.asyncio
    async def test_run_all_succeed(self):
        mock_conn = AsyncMock()
        # / regime query
        mock_conn.fetchrow.return_value = {"regime": "bull", "confidence": 0.8}
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, return_value=_make_ratio()),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=_make_dcf()),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=_make_earnings()),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=_make_insider()),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1) as mock_store,
        ):
            results = await self.agent.run(pool, ["AAPL"])

        assert "AAPL" in results
        assert results["AAPL"] is not None
        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs.kwargs["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_run_partial_failure(self):
        # / ratio fails, others succeed -> score computed from available
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"regime": "bull", "confidence": 0.8}
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, side_effect=Exception("ratio failed")),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=_make_dcf()),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=_make_earnings()),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=_make_insider()),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1),
        ):
            results = await self.agent.run(pool, ["AAPL"])

        assert results["AAPL"] is not None  # / still computed from dcf+earnings+insider

    @pytest.mark.asyncio
    async def test_run_all_fail(self):
        # / all analysis raise, score is None
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None  # / no regime
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, side_effect=Exception("fail")),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1),
        ):
            results = await self.agent.run(pool, ["AAPL"])

        assert results["AAPL"] is None

    @pytest.mark.asyncio
    async def test_run_empty_symbols(self):
        pool = _mock_pool()
        results = await self.agent.run(pool, [])
        assert results == {}

    @pytest.mark.asyncio
    async def test_regime_fetched_from_db(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"regime": "bear", "confidence": 0.9}
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, return_value=_make_ratio()),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=_make_dcf()),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=_make_earnings()),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=_make_insider()),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()) as mock_summary,
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1) as mock_store,
        ):
            await self.agent.run(pool, ["AAPL"])

        # / regime passed to generate_summary
        mock_summary.assert_called_once()
        call_kwargs = mock_summary.call_args
        assert call_kwargs.kwargs["regime"] == "bear"
        # / regime passed to store_analysis_score
        store_kwargs = mock_store.call_args.kwargs
        assert store_kwargs["regime"] == "bear"

    @pytest.mark.asyncio
    async def test_symbol_failure_continues_to_next(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"regime": "bull", "confidence": 0.8}
        pool = _mock_pool(mock_conn)

        call_count = 0

        async def _ratios_side_effect(pool, symbol):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise Exception("bad symbol")
            return _make_ratio()

        with (
            patch("src.agents.analyst_agent.analyze_ratios", side_effect=_ratios_side_effect),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=_make_dcf()),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=_make_earnings()),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=_make_insider()),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1),
        ):
            results = await self.agent.run(pool, ["AAPL", "MSFT"])

        assert len(results) == 2
        assert results["AAPL"] is not None
        assert results["MSFT"] is not None

    @pytest.mark.asyncio
    async def test_store_called_with_correct_composite(self):
        # / verify composite_score matches fundamental_score when no technical
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None  # / no regime
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, return_value=_make_ratio(80.0)),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1) as mock_store,
        ):
            await self.agent.run(pool, ["AAPL"])

        kw = mock_store.call_args.kwargs
        assert kw["fundamental_score"] == 80.0
        assert kw["composite_score"] == 80.0
        assert kw["technical_score"] is None

    @pytest.mark.asyncio
    async def test_used_fundamentals_flag(self):
        # / if ratio_score or dcf_result is present, used_fundamentals=True
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        with (
            patch("src.agents.analyst_agent.analyze_ratios", new_callable=AsyncMock, return_value=_make_ratio()),
            patch("src.agents.analyst_agent.analyze_dcf", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.analyze_earnings", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.analyze_insider_activity", new_callable=AsyncMock, return_value=None),
            patch("src.agents.analyst_agent.generate_summary", new_callable=AsyncMock, return_value=_make_summary()),
            patch("src.agents.analyst_agent.tools.store_analysis_score", new_callable=AsyncMock, return_value=1) as mock_store,
        ):
            await self.agent.run(pool, ["AAPL"])

        assert mock_store.call_args.kwargs["used_fundamentals"] is True
