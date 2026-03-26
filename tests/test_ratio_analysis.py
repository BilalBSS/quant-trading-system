# / tests for ratio analysis scoring

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analysis.ratio_analysis import (
    RatioScore,
    analyze_ratios,
    analyze_ratios_batch,
    compute_ratio_score,
    score_debt_equity,
    score_fcf_margin,
    score_pe,
    score_peg,
    score_ps,
)


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


class TestScorePe:
    def test_undervalued_vs_sector(self):
        # / pe 15 vs sector avg 30 = very undervalued
        score = score_pe(Decimal("15"), Decimal("30"))
        assert score is not None
        assert score > 70

    def test_overvalued_vs_sector(self):
        # / pe 60 vs sector avg 30 = overvalued
        score = score_pe(Decimal("60"), Decimal("30"))
        assert score is not None
        assert score < 30

    def test_fair_value_vs_sector(self):
        # / pe equals sector avg
        score = score_pe(Decimal("30"), Decimal("30"))
        assert score is not None
        assert 50 < score < 80

    def test_negative_pe(self):
        assert score_pe(Decimal("-10"), Decimal("30")) == 0.0

    def test_none_pe(self):
        assert score_pe(None, Decimal("30")) is None

    def test_no_sector_avg_low_pe(self):
        # / absolute scale: pe 10 -> high score
        score = score_pe(Decimal("10"), None)
        assert score is not None
        assert score > 80

    def test_no_sector_avg_high_pe(self):
        # / absolute scale: pe 45 -> low score
        score = score_pe(Decimal("45"), None)
        assert score is not None
        assert score < 20

    def test_clamps_to_100(self):
        score = score_pe(Decimal("5"), Decimal("100"))
        assert score is not None
        assert score == 100.0

    def test_clamps_to_0(self):
        score = score_pe(Decimal("100"), Decimal("10"))
        assert score is not None
        assert score == 0.0


class TestScorePs:
    def test_undervalued(self):
        score = score_ps(Decimal("3"), Decimal("10"))
        assert score is not None
        assert score > 70

    def test_overvalued(self):
        score = score_ps(Decimal("25"), Decimal("10"))
        assert score is not None
        assert score == 0.0

    def test_none(self):
        assert score_ps(None, Decimal("10")) is None

    def test_negative(self):
        assert score_ps(Decimal("-5"), Decimal("10")) == 0.0


class TestScorePeg:
    def test_undervalued(self):
        # / peg < 1 = undervalued
        score = score_peg(Decimal("0.7"))
        assert score is not None
        assert score > 80

    def test_fair(self):
        score = score_peg(Decimal("1.5"))
        assert score is not None
        assert 40 < score < 80

    def test_overvalued(self):
        score = score_peg(Decimal("2.8"))
        assert score is not None
        assert score < 20

    def test_negative_peg(self):
        # / negative peg = negative growth
        assert score_peg(Decimal("-0.5")) == 0.0

    def test_none(self):
        assert score_peg(None) is None


class TestScoreFcfMargin:
    def test_high_margin(self):
        score = score_fcf_margin(Decimal("0.30"))
        assert score is not None
        assert score == 100.0

    def test_low_margin(self):
        score = score_fcf_margin(Decimal("0.05"))
        assert score is not None
        assert 30 < score < 50

    def test_negative_margin(self):
        score = score_fcf_margin(Decimal("-0.05"))
        assert score is not None
        assert score < 20

    def test_none(self):
        assert score_fcf_margin(None) is None


class TestScoreDebtEquity:
    def test_low_debt(self):
        # / d/e 0.3 (as ratio) = healthy
        score = score_debt_equity(Decimal("0.3"))
        assert score is not None
        assert score > 80

    def test_high_debt(self):
        score = score_debt_equity(Decimal("2.5"))
        assert score is not None
        assert score < 30

    def test_yfinance_percentage_format(self):
        # / yfinance returns 50 meaning 0.5x
        score = score_debt_equity(Decimal("50"))
        assert score is not None
        assert score > 80

    def test_none(self):
        assert score_debt_equity(None) is None


class TestComputeRatioScore:
    def test_full_data(self):
        data = {
            "symbol": "AAPL",
            "date": date(2026, 3, 25),
            "pe_ratio": Decimal("15"),
            "ps_ratio": Decimal("5"),
            "peg_ratio": Decimal("1.0"),
            "fcf_margin": Decimal("0.25"),
            "debt_to_equity": Decimal("0.5"),
            "sector_pe_avg": Decimal("25"),
            "sector_ps_avg": Decimal("10"),
        }
        result = compute_ratio_score(data)
        assert result.symbol == "AAPL"
        assert result.composite_score is not None
        assert 0 <= result.composite_score <= 100
        assert result.pe_score is not None
        assert result.ps_score is not None

    def test_partial_data(self):
        data = {
            "symbol": "MSFT",
            "date": date(2026, 3, 25),
            "pe_ratio": Decimal("20"),
            "ps_ratio": None,
            "peg_ratio": None,
            "fcf_margin": None,
            "debt_to_equity": None,
            "sector_pe_avg": Decimal("25"),
            "sector_ps_avg": None,
        }
        result = compute_ratio_score(data)
        assert result.composite_score is not None
        assert result.ps_score is None
        assert result.peg_score is None

    def test_no_data(self):
        data = {
            "symbol": "EMPTY",
            "date": date(2026, 3, 25),
        }
        result = compute_ratio_score(data)
        assert result.composite_score is None


class TestAnalyzeRatios:
    @pytest.mark.asyncio
    async def test_returns_score(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "symbol": "AAPL",
            "date": date(2026, 3, 25),
            "pe_ratio": Decimal("20"),
            "pe_forward": Decimal("18"),
            "ps_ratio": Decimal("7"),
            "peg_ratio": Decimal("1.2"),
            "revenue_growth_1y": Decimal("0.10"),
            "revenue_growth_3y": None,
            "fcf_margin": Decimal("0.22"),
            "debt_to_equity": Decimal("45"),
            "sector": "Technology",
            "sector_pe_avg": Decimal("28"),
            "sector_ps_avg": Decimal("9"),
        }
        pool = _mock_pool(mock_conn)

        result = await analyze_ratios(pool, "AAPL")
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.composite_score is not None

    @pytest.mark.asyncio
    async def test_returns_none_no_data(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        result = await analyze_ratios(pool, "FAKE")
        assert result is None

    @pytest.mark.asyncio
    async def test_batch(self):
        call_count = 0

        async def mock_fetchrow(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "FAKE" in args:
                return None
            return {
                "symbol": args[0] if args else "TEST",
                "date": date(2026, 3, 25),
                "pe_ratio": Decimal("20"),
                "pe_forward": None,
                "ps_ratio": None,
                "peg_ratio": None,
                "revenue_growth_1y": None,
                "revenue_growth_3y": None,
                "fcf_margin": None,
                "debt_to_equity": None,
                "sector": "Tech",
                "sector_pe_avg": None,
                "sector_ps_avg": None,
            }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = mock_fetchrow
        pool = _mock_pool(mock_conn)

        results = await analyze_ratios_batch(pool, ["AAPL", "MSFT"])
        assert len(results) == 2
