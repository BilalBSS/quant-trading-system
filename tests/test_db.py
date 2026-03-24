# / tests for db module — pool init, migrations, url masking

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.db import _mask_url, _run_migrations, close_db, get_pool, init_db
import src.data.db as db_module


@pytest.fixture(autouse=True)
def reset_pool():
    # / reset global pool state between tests
    db_module._pool = None
    yield
    db_module._pool = None


class TestMaskUrl:
    def test_masks_password(self):
        url = "postgresql://user:secret@host:5432/db"
        result = _mask_url(url)
        assert "secret" not in result
        assert "***" in result
        assert "host:5432" in result

    def test_no_password(self):
        url = "postgresql://host:5432/db"
        assert _mask_url(url) == "postgresql://host:5432/db"

    def test_complex_url(self):
        url = "postgresql://neondb_owner:npg_abc123@ep-pooler.neon.tech/neondb?sslmode=require"
        result = _mask_url(url)
        assert "npg_abc123" not in result
        assert "***" in result
        assert "ep-pooler.neon.tech" in result

    def test_encoded_password(self):
        url = "postgresql://user:p%40ss%3Aword@host:5432/db"
        result = _mask_url(url)
        assert "p%40ss" not in result
        assert "***" in result


class TestGetPool:
    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            await get_pool()

    @pytest.mark.asyncio
    async def test_returns_pool_when_initialized(self):
        fake_pool = MagicMock()
        db_module._pool = fake_pool
        result = await get_pool()
        assert result is fake_pool


class TestCloseDb:
    @pytest.mark.asyncio
    async def test_closes_pool(self):
        fake_pool = AsyncMock()
        db_module._pool = fake_pool
        await close_db()
        fake_pool.close.assert_awaited_once()
        assert db_module._pool is None

    @pytest.mark.asyncio
    async def test_noop_when_no_pool(self):
        # / should not raise
        await close_db()
        assert db_module._pool is None


class TestInitDb:
    @pytest.mark.asyncio
    async def test_raises_without_url(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DATABASE_URL"):
                await init_db()

    @pytest.mark.asyncio
    async def test_returns_existing_pool(self):
        fake_pool = MagicMock()
        db_module._pool = fake_pool
        result = await init_db("postgresql://fake")
        assert result is fake_pool

    @pytest.mark.asyncio
    async def test_creates_pool(self):
        fake_pool = AsyncMock()
        with patch("src.data.db.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=fake_pool)
            with patch("src.data.db._run_migrations", new_callable=AsyncMock):
                result = await init_db("postgresql://user:pass@host/db")
                assert result is fake_pool
                mock_asyncpg.create_pool.assert_awaited_once()


class TestRunMigrations:
    @pytest.mark.asyncio
    async def test_creates_migrations_table(self):
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        # / asyncpg pool.acquire() returns an async context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_ctx

        with patch("src.data.db.MIGRATIONS_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            await _run_migrations(mock_pool)

        # / should have called execute to create _migrations table
        calls = [str(c) for c in mock_conn.execute.call_args_list]
        assert any("_migrations" in c for c in calls)
