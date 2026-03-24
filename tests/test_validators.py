# / tests for data validation bounds and filtering

from decimal import Decimal

import pytest

from src.data.validators import (
    ValidationResult,
    _check_bound,
    filter_valid,
    validate_fundamentals,
    validate_market_data,
    validate_sentiment,
)


class TestCheckBound:
    def test_none_value_is_valid(self):
        r = _check_bound("price", None, 0, 100)
        assert r.valid is True

    def test_within_bounds(self):
        r = _check_bound("price", 50, 0, 100)
        assert r.valid is True

    def test_at_min_bound(self):
        r = _check_bound("price", 0, 0, 100)
        assert r.valid is True

    def test_at_max_bound(self):
        r = _check_bound("price", 100, 0, 100)
        assert r.valid is True

    def test_below_min(self):
        r = _check_bound("price", -1, 0, 100)
        assert r.valid is False
        assert "below min" in r.reason

    def test_above_max(self):
        r = _check_bound("price", 101, 0, 100)
        assert r.valid is False
        assert "above max" in r.reason

    def test_non_numeric_value(self):
        r = _check_bound("price", "not_a_number", 0, 100)
        assert r.valid is False
        assert "not numeric" in r.reason

    def test_string_numeric(self):
        r = _check_bound("price", "50.5", 0, 100)
        assert r.valid is True

    def test_no_min_bound(self):
        r = _check_bound("price", -99999, None, 100)
        assert r.valid is True

    def test_no_max_bound(self):
        r = _check_bound("price", 99999, 0, None)
        assert r.valid is True


class TestValidateMarketData:
    def test_valid_ohlcv(self):
        row = {"open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000000}
        results = validate_market_data(row)
        assert all(r.valid for r in results)

    def test_negative_price(self):
        row = {"open": -5, "high": 110, "low": 95, "close": 105, "volume": 1000000}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert len(failures) >= 1
        assert any(r.field == "open" for r in failures)

    def test_high_below_low(self):
        row = {"open": 100, "high": 90, "low": 95, "close": 105, "volume": 1000000}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert any(r.field == "high_low" for r in failures)

    def test_high_equals_low(self):
        row = {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 0}
        results = validate_market_data(row)
        assert all(r.valid for r in results)

    def test_missing_fields_pass(self):
        # / none values are valid (missing != wrong)
        row = {}
        results = validate_market_data(row)
        assert all(r.valid for r in results)

    def test_volume_too_high(self):
        row = {"open": 100, "high": 110, "low": 95, "close": 105, "volume": 999_999_999_999}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert any(r.field == "volume" for r in failures)

    def test_price_too_high(self):
        row = {"open": 9999999, "high": 110, "low": 95, "close": 105, "volume": 100}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert any(r.field == "open" for r in failures)

    def test_open_above_high(self):
        row = {"open": 200, "high": 110, "low": 95, "close": 105, "volume": 100}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert any(r.field == "open_range" for r in failures)

    def test_close_below_low(self):
        row = {"open": 100, "high": 110, "low": 95, "close": 50, "volume": 100}
        results = validate_market_data(row)
        failures = [r for r in results if not r.valid]
        assert any(r.field == "close_range" for r in failures)


class TestValidateFundamentals:
    def test_valid_fundamentals(self):
        row = {
            "pe_ratio": 25, "pe_forward": 20, "ps_ratio": 5,
            "peg_ratio": 1.2, "revenue_growth_1y": 0.15,
            "revenue_growth_3y": 0.10, "fcf_margin": 0.12,
            "debt_to_equity": 0.5,
        }
        results = validate_fundamentals(row)
        assert all(r.valid for r in results)

    def test_negative_pe_within_bounds(self):
        row = {"pe_ratio": -50}
        results = validate_fundamentals(row)
        pe_result = next(r for r in results if r.field == "pe_ratio")
        assert pe_result.valid is True

    def test_pe_out_of_bounds(self):
        row = {"pe_ratio": 99999}
        results = validate_fundamentals(row)
        pe_result = next(r for r in results if r.field == "pe_ratio")
        assert pe_result.valid is False

    def test_negative_ps_ratio(self):
        row = {"ps_ratio": -1}
        results = validate_fundamentals(row)
        ps_result = next(r for r in results if r.field == "ps_ratio")
        assert ps_result.valid is False

    def test_fcf_margin_bounds(self):
        # / fcf margin can be negative but not below -10
        row = {"fcf_margin": -0.5}
        results = validate_fundamentals(row)
        fcf_result = next(r for r in results if r.field == "fcf_margin")
        assert fcf_result.valid is True

    def test_fcf_margin_too_low(self):
        row = {"fcf_margin": -15}
        results = validate_fundamentals(row)
        fcf_result = next(r for r in results if r.field == "fcf_margin")
        assert fcf_result.valid is False

    def test_empty_row(self):
        results = validate_fundamentals({})
        assert all(r.valid for r in results)


class TestValidateSentiment:
    def test_valid_sentiment(self):
        row = {"sentiment_score": 0.5}
        results = validate_sentiment(row)
        assert all(r.valid for r in results)

    def test_sentiment_at_bounds(self):
        for val in [-1.0, 0, 1.0]:
            results = validate_sentiment({"sentiment_score": val})
            assert all(r.valid for r in results)

    def test_sentiment_out_of_bounds(self):
        results = validate_sentiment({"sentiment_score": 1.5})
        assert any(not r.valid for r in results)

    def test_missing_sentiment(self):
        results = validate_sentiment({})
        assert all(r.valid for r in results)


class TestFilterValid:
    def test_keeps_valid_rows(self):
        rows = [
            {"symbol": "AAPL", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000},
            {"symbol": "MSFT", "open": 200, "high": 210, "low": 195, "close": 205, "volume": 2000},
        ]
        result = filter_valid(rows, validate_market_data, "test")
        assert len(result) == 2

    def test_drops_invalid_rows(self):
        rows = [
            {"symbol": "AAPL", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000},
            {"symbol": "BAD", "open": -999, "high": 110, "low": 95, "close": 105, "volume": 1000},
        ]
        result = filter_valid(rows, validate_market_data, "test")
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_empty_input(self):
        result = filter_valid([], validate_market_data, "test")
        assert result == []

    def test_all_invalid(self):
        rows = [
            {"symbol": "BAD1", "open": -1, "high": 110, "low": 95, "close": 105, "volume": 1000},
            {"symbol": "BAD2", "open": -2, "high": 110, "low": 95, "close": 105, "volume": 1000},
        ]
        result = filter_valid(rows, validate_market_data, "test")
        assert result == []
