# / tests for dcf model + monte carlo simulation

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.analysis.dcf_model import (
    DCFAssumptions,
    DCFResult,
    build_assumptions_from_db,
    compute_dcf,
    run_dcf_simulation,
    store_dcf_result,
)


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


class TestRunDcfSimulation:
    def test_basic_simulation(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=1000.0,
            fcf_margin=0.20,
            revenue_growth=0.10,
            shares_outstanding=100.0,
        )
        values = run_dcf_simulation(assumptions, num_simulations=1000, rng=rng)
        assert len(values) == 1000
        assert all(v >= 0 for v in values)

    def test_higher_growth_higher_value(self):
        rng = np.random.default_rng(42)
        low_growth = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.05,
            shares_outstanding=1.0, growth_std=0.01,
        )
        high_growth = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.20,
            shares_outstanding=1.0, growth_std=0.01,
        )
        low_vals = run_dcf_simulation(low_growth, 500, rng)
        rng2 = np.random.default_rng(42)
        high_vals = run_dcf_simulation(high_growth, 500, rng2)

        assert np.median(high_vals) > np.median(low_vals)

    def test_zero_revenue_returns_zero(self):
        assumptions = DCFAssumptions(
            revenue=0.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0,
        )
        values = run_dcf_simulation(assumptions, 100)
        assert all(v == 0.0 for v in values)

    def test_net_debt_reduces_value(self):
        rng1 = np.random.default_rng(42)
        no_debt = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0, net_debt=0.0, growth_std=0.01,
        )
        rng2 = np.random.default_rng(42)
        with_debt = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0, net_debt=500.0, growth_std=0.01,
        )
        no_debt_vals = run_dcf_simulation(no_debt, 500, rng1)
        debt_vals = run_dcf_simulation(with_debt, 500, rng2)

        assert np.median(no_debt_vals) > np.median(debt_vals)

    def test_simulation_produces_distribution(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0, growth_std=0.10,
        )
        values = run_dcf_simulation(assumptions, 5000, rng)
        # / should have meaningful spread
        p10 = np.percentile(values, 10)
        p90 = np.percentile(values, 90)
        assert p90 > p10


class TestComputeDcf:
    def test_basic_result(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0,
        )
        result = compute_dcf("AAPL", 150.0, assumptions, date(2026, 3, 25), 1000, rng)

        assert result.symbol == "AAPL"
        assert result.current_price == 150.0
        assert result.fair_value_median > 0
        assert result.fair_value_p10 <= result.fair_value_median <= result.fair_value_p90
        assert result.confidence in ("high", "medium", "low")
        assert result.num_simulations == 1000

    def test_upside_calculation(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.15,
            shares_outstanding=1.0, terminal_multiple=20.0,
        )
        # / set current price very low to guarantee upside
        result = compute_dcf("CHEAP", 1.0, assumptions, num_simulations=100, rng=rng)
        assert result.upside_pct > 0

    def test_zero_current_price(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=100.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0,
        )
        result = compute_dcf("TEST", 0.0, assumptions, num_simulations=100, rng=rng)
        assert result.upside_pct == 0.0

    def test_confidence_levels(self):
        # / tight distribution = high confidence
        rng = np.random.default_rng(42)
        tight = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            growth_std=0.001, margin_std=0.001, terminal_multiple_std=0.1,
            shares_outstanding=1.0,
        )
        result = compute_dcf("TIGHT", 100.0, tight, num_simulations=1000, rng=rng)
        assert result.confidence == "high"


class TestBuildAssumptions:
    @pytest.mark.asyncio
    async def test_builds_from_db(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = [
            # / fundamentals row
            {
                "fcf_margin": Decimal("0.20"),
                "revenue_growth_1y": Decimal("0.12"),
                "debt_to_equity": Decimal("0.5"),
                "ps_ratio": Decimal("8.0"),
                "pe_ratio": Decimal("25"),
                "pe_forward": None,
                "peg_ratio": None,
                "revenue_growth_3y": None,
                "sector": "Tech",
                "sector_pe_avg": None,
                "sector_ps_avg": None,
                "symbol": "AAPL",
                "date": date(2026, 3, 25),
            },
            # / price row
            {"close": Decimal("180.00")},
        ]
        pool = _mock_pool(mock_conn)

        result = await build_assumptions_from_db(pool, "AAPL")
        assert result is not None
        assert result.fcf_margin == 0.20
        assert result.revenue_growth == 0.12

    @pytest.mark.asyncio
    async def test_returns_none_no_fundamentals(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        result = await build_assumptions_from_db(pool, "FAKE")
        assert result is None


class TestStoreDcfResult:
    @pytest.mark.asyncio
    async def test_stores_result(self):
        mock_conn = AsyncMock()
        pool = _mock_pool(mock_conn)

        result = DCFResult(
            symbol="AAPL",
            date=date(2026, 3, 25),
            fair_value_median=200.0,
            fair_value_p10=150.0,
            fair_value_p90=260.0,
            current_price=180.0,
            upside_pct=0.1111,
            num_simulations=10000,
            confidence="medium",
        )

        success = await store_dcf_result(pool, result, regime="bull")
        assert success is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_store_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("db error")
        pool = _mock_pool(mock_conn)

        result = DCFResult(
            symbol="FAIL",
            date=date(2026, 3, 25),
            fair_value_median=100.0,
            fair_value_p10=80.0,
            fair_value_p90=120.0,
            current_price=90.0,
            upside_pct=0.11,
            num_simulations=100,
            confidence="low",
        )

        success = await store_dcf_result(pool, result)
        assert success is False
