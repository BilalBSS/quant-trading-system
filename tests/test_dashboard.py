# / tests for dashboard api endpoints

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date, datetime
from decimal import Decimal

from src.dashboard.app import _serialize, _serialize_one


class TestSerialize:
    def test_serializes_decimal(self):
        row = {"price": Decimal("182.40")}
        result = _serialize_one(row)
        assert result["price"] == "182.40"

    def test_serializes_date(self):
        row = {"date": date(2026, 3, 26)}
        result = _serialize_one(row)
        assert result["date"] == "2026-03-26"

    def test_serializes_datetime(self):
        row = {"created_at": datetime(2026, 3, 26, 14, 30)}
        result = _serialize_one(row)
        assert "2026-03-26" in result["created_at"]

    def test_preserves_primitives(self):
        row = {"count": 5, "name": "test", "active": True, "note": None}
        result = _serialize_one(row)
        assert result["count"] == 5
        assert result["name"] == "test"
        assert result["active"] is True
        assert result["note"] is None

    def test_none_returns_none(self):
        assert _serialize_one(None) is None

    def test_serialize_list(self):
        rows = [{"a": 1}, {"a": 2}]
        result = _serialize(rows)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_serialize_empty_list(self):
        assert _serialize([]) == []

    def test_unknown_type_becomes_string(self):
        row = {"data": {"nested": "dict"}}
        result = _serialize_one(row)
        # / dicts get str() representation
        assert isinstance(result["data"], str)
