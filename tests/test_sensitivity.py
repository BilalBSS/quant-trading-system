# / tests for sensitivity matrix analysis

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.analysis.dcf_model import DCFAssumptions
from src.analysis.sensitivity import (
    analyze_sensitivity,
    compute_sensitivity_matrix,
    determine_sensitivity_driver,
)


class TestComputeSensitivityMatrix:
    def test_produces_correct_shape(self):
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0,
        )
        growth_rates = [0.05, 0.10, 0.15]
        multiples = [10.0, 15.0, 20.0]
        rng = np.random.default_rng(42)

        values, upsides = compute_sensitivity_matrix(
            assumptions, 100.0, growth_rates, multiples, 100, rng,
        )

        assert len(values) == 3  # 3 growth rates
        assert len(values[0]) == 3  # 3 multiples
        assert len(upsides) == 3
        assert len(upsides[0]) == 3

    def test_higher_growth_higher_value(self):
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0, growth_std=0.01,
        )
        growth_rates = [0.0, 0.10, 0.20]
        multiples = [15.0]
        rng = np.random.default_rng(42)

        values, _ = compute_sensitivity_matrix(
            assumptions, 100.0, growth_rates, multiples, 500, rng,
        )

        # / each row should have higher value than the previous
        assert values[0][0] < values[1][0] < values[2][0]

    def test_higher_multiple_higher_value(self):
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0, terminal_multiple_std=0.1,
        )
        growth_rates = [0.10]
        multiples = [8.0, 15.0, 25.0]
        rng = np.random.default_rng(42)

        values, _ = compute_sensitivity_matrix(
            assumptions, 100.0, growth_rates, multiples, 500, rng,
        )

        assert values[0][0] < values[0][1] < values[0][2]

    def test_upside_negative_when_overvalued(self):
        assumptions = DCFAssumptions(
            revenue=10.0, fcf_margin=0.05, revenue_growth=0.02,
            shares_outstanding=1.0, terminal_multiple=5.0,
        )
        rng = np.random.default_rng(42)

        _, upsides = compute_sensitivity_matrix(
            assumptions, 10000.0, [0.02], [5.0], 100, rng,
        )
        # / tiny revenue, huge price -> negative upside
        assert upsides[0][0] < 0


class TestDetermineSensitivityDriver:
    def test_growth_dominant(self):
        # / wide range across rows (growth varies), narrow across columns
        matrix = [
            [50.0, 55.0],
            [100.0, 110.0],
            [200.0, 215.0],
        ]
        result = determine_sensitivity_driver(matrix, [0.0, 0.10, 0.20], [10.0, 15.0])
        assert result == "growth"

    def test_terminal_dominant(self):
        # / narrow range across rows, wide across columns
        matrix = [
            [50.0, 200.0],
            [55.0, 210.0],
        ]
        result = determine_sensitivity_driver(matrix, [0.05, 0.06], [5.0, 30.0])
        assert result == "terminal_multiple"


class TestAnalyzeSensitivity:
    def test_full_analysis(self):
        rng = np.random.default_rng(42)
        assumptions = DCFAssumptions(
            revenue=1000.0, fcf_margin=0.20, revenue_growth=0.10,
            shares_outstanding=1.0,
        )

        result = analyze_sensitivity(
            "AAPL", assumptions, 150.0,
            growth_rates=[0.05, 0.10, 0.15],
            terminal_multiples=[10.0, 15.0, 20.0],
            num_simulations=100,
            rng=rng,
        )

        assert result.symbol == "AAPL"
        assert result.current_price == 150.0
        assert result.base_fair_value > 0
        assert len(result.matrix) == 3
        assert len(result.matrix[0]) == 3
        assert result.most_sensitive_to in ("growth", "terminal_multiple")
