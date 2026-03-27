# / tests for notification system — discord, slack, telegram

import asyncio
import os
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.notifications.notifier import (
    DISCORD_CHANNELS,
    NotificationEvent,
    Severity,
    SEVERITY_COLORS,
    SEVERITY_EMOJI,
    _dispatch,
    _is_throttled,
    _last_sent,
    _send_discord,
    _send_slack,
    _send_telegram,
    _error_buffer,
    buffer_error,
    notify,
    notify_async,
    notify_analysis_highlight,
    notify_daily_digest,
    notify_daily_synthesis,
    notify_evolution_summary,
    notify_sentiment_shift,
    notify_strategy_promoted,
    notify_system_error,
    notify_trade_error,
    notify_trade_executed,
)


@pytest.fixture(autouse=True)
def clear_state():
    _last_sent.clear()
    _error_buffer.clear()
    yield
    _last_sent.clear()
    _error_buffer.clear()


def _event(severity=Severity.HIGH, title="test", message="msg", fields=None):
    return NotificationEvent(
        severity=severity,
        title=title,
        message=message,
        fields=fields or {},
    )


class TestSeverity:
    def test_all_severities_have_colors(self):
        for s in Severity:
            assert s in SEVERITY_COLORS

    def test_all_severities_have_emoji(self):
        for s in Severity:
            assert s in SEVERITY_EMOJI


class TestThrottle:
    def test_first_event_not_throttled(self):
        event = _event(title="unique1")
        assert _is_throttled(event) is False

    def test_duplicate_event_throttled(self):
        event = _event(title="dup")
        _is_throttled(event)
        assert _is_throttled(event) is True

    def test_different_titles_not_throttled(self):
        e1 = _event(title="a")
        e2 = _event(title="b")
        _is_throttled(e1)
        assert _is_throttled(e2) is False

    def test_different_severities_not_throttled(self):
        e1 = _event(severity=Severity.HIGH, title="same")
        e2 = _event(severity=Severity.LOW, title="same")
        _is_throttled(e1)
        assert _is_throttled(e2) is False


class TestSendDiscord:
    @pytest.mark.asyncio
    async def test_skips_when_no_url(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await _send_discord(_event())
            assert result is False

    @pytest.mark.asyncio
    async def test_sends_embed(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/hook"}):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock:
                result = await _send_discord(_event(title="trade", message="AAPL +2%"))
                assert result is True
                _, kwargs = mock.call_args
                embed = kwargs["json"]["embeds"][0]
                assert "trade" in embed["title"]
                assert embed["color"] == SEVERITY_COLORS[Severity.HIGH]

    @pytest.mark.asyncio
    async def test_includes_fields(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/hook"}):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock:
                event = _event(fields={"strategy": "momentum_v3"})
                await _send_discord(event)
                embed = mock.call_args[1]["json"]["embeds"][0]
                assert len(embed["fields"]) == 1
                assert embed["fields"][0]["name"] == "strategy"

    @pytest.mark.asyncio
    async def test_handles_send_failure(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/hook"}):
            with patch("src.notifications.notifier.api_post", side_effect=Exception("timeout")):
                result = await _send_discord(_event())
                assert result is False


class TestSendSlack:
    @pytest.mark.asyncio
    async def test_skips_when_no_url(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await _send_slack(_event())
            assert result is False

    @pytest.mark.asyncio
    async def test_sends_text(self):
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/x"}):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock:
                result = await _send_slack(_event(title="alert", message="system down"))
                assert result is True
                text = mock.call_args[1]["json"]["text"]
                assert "*alert*" in text
                assert "system down" in text

    @pytest.mark.asyncio
    async def test_includes_fields(self):
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/x"}):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock:
                event = _event(fields={"sharpe": "1.42"})
                await _send_slack(event)
                text = mock.call_args[1]["json"]["text"]
                assert "*sharpe:*" in text


class TestSendTelegram:
    @pytest.mark.asyncio
    async def test_skips_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await _send_telegram(_event())
            assert result is False

    @pytest.mark.asyncio
    async def test_skips_when_no_chat_id(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}, clear=True):
            result = await _send_telegram(_event())
            assert result is False

    @pytest.mark.asyncio
    async def test_sends_html_message(self):
        env = {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "456"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock:
                result = await _send_telegram(_event(title="alert"))
                assert result is True
                call_kwargs = mock.call_args[1]
                assert "bot123" not in call_kwargs  # / url is positional
                body = call_kwargs["json"]
                assert body["chat_id"] == "456"
                assert body["parse_mode"] == "HTML"
                assert "<b>alert</b>" in body["text"]


class TestDispatch:
    @pytest.mark.asyncio
    async def test_sends_to_all_channels(self):
        env = {
            "DISCORD_WEBHOOK_URL": "https://discord.com/hook",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/x",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "123",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock):
                results = await _dispatch(_event())
                assert results["discord"] is True
                assert results["slack"] is True
                assert results["telegram"] is True


class TestNotify:
    @pytest.mark.asyncio
    async def test_sends_event(self):
        with patch("src.notifications.notifier._dispatch", new_callable=AsyncMock) as mock:
            mock.return_value = {"discord": True}
            await notify(_event(title="unique_notify"))
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_throttled_event_skipped(self):
        with patch("src.notifications.notifier._dispatch", new_callable=AsyncMock) as mock:
            mock.return_value = {"discord": True}
            await notify(_event(title="throttle_test"))
            await notify(_event(title="throttle_test"))
            assert mock.call_count == 1


class TestErrorBuffer:
    def test_buffer_error_adds_to_buffer(self):
        buffer_error("trade_error", "AAPL failed")
        assert "trade_error" in _error_buffer
        assert len(_error_buffer["trade_error"]) == 1

    def test_buffer_error_accumulates(self):
        buffer_error("api_error", "timeout 1")
        buffer_error("api_error", "timeout 2")
        buffer_error("api_error", "timeout 3")
        assert len(_error_buffer["api_error"]) == 3


class TestConvenienceHelpers:
    def test_notify_trade_executed(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_trade_executed("AAPL", "buy", 10.0, 182.40, "momentum_v3")
            event = mock.call_args[0][0]
            assert event.severity == Severity.HIGH
            assert "AAPL" in event.message
            assert "182.40" in event.message

    def test_notify_trade_error_buffers(self):
        notify_trade_error("MSFT", "sell", "broker timeout")
        assert "trade_error" in _error_buffer

    def test_notify_system_error_buffers(self):
        notify_system_error("db connection lost", "orchestrator")
        assert "system_error" in _error_buffer
        assert "orchestrator" in _error_buffer["system_error"][0]

    def test_notify_evolution_summary(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            summary = {"generation": 5, "killed": [1, 2], "mutated": [3], "promoted": [], "errors": []}
            notify_evolution_summary(summary)
            event = mock.call_args[0][0]
            assert event.severity == Severity.MEDIUM
            assert "gen 5" in event.title

    def test_notify_strategy_promoted(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_strategy_promoted("momentum_v3", 1.42, 14)
            event = mock.call_args[0][0]
            assert event.severity == Severity.HIGH
            assert "momentum_v3" in event.message

    def test_notify_daily_digest(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_daily_digest(102450.0, 2450.0, 10, 5)
            event = mock.call_args[0][0]
            assert event.severity == Severity.LOW
            assert "$102,450.00" in event.message

    def test_notify_daily_digest_negative_pnl(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_daily_digest(97500.0, -2500.0, 8, 3)
            event = mock.call_args[0][0]
            assert "-" in event.fields["daily P&L"]


# ---------------------------------------------------------------------------
# / discord channel routing tests
# ---------------------------------------------------------------------------

class TestChannelRouting:
    @pytest.mark.asyncio
    async def test_trades_channel_uses_webhook_trades(self):
        event = NotificationEvent(
            severity=Severity.HIGH, title="test", message="trade", channel="trades",
        )
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_TRADES": "https://hook/trades"}, clear=False):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock_post:
                await _send_discord(event)
                url = mock_post.call_args[0][0]
                assert url == "https://hook/trades"

    @pytest.mark.asyncio
    async def test_analysis_channel_routing(self):
        event = NotificationEvent(
            severity=Severity.MEDIUM, title="test", message="analysis", channel="analysis",
        )
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_ANALYSIS": "https://hook/analysis"}, clear=False):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock_post:
                await _send_discord(event)
                url = mock_post.call_args[0][0]
                assert url == "https://hook/analysis"

    @pytest.mark.asyncio
    async def test_fallback_to_default_webhook(self):
        event = NotificationEvent(
            severity=Severity.LOW, title="test", message="fallback", channel="trades",
        )
        # / no DISCORD_WEBHOOK_TRADES set, falls back to DISCORD_WEBHOOK_URL
        env = {"DISCORD_WEBHOOK_URL": "https://hook/default"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.notifications.notifier.api_post", new_callable=AsyncMock) as mock_post:
                await _send_discord(event)
                url = mock_post.call_args[0][0]
                assert url == "https://hook/default"

    @pytest.mark.asyncio
    async def test_no_webhook_skips_silently(self):
        event = NotificationEvent(
            severity=Severity.LOW, title="test", message="skip", channel="trades",
        )
        with patch.dict(os.environ, {}, clear=True):
            result = await _send_discord(event)
            assert result is False


# ---------------------------------------------------------------------------
# / new notification helper tests
# ---------------------------------------------------------------------------

class TestNewNotifyHelpers:
    def test_notify_analysis_highlight(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_analysis_highlight("NVDA", "bullish", 85.0)
            event = mock.call_args[0][0]
            assert event.channel == "analysis"
            assert "NVDA" in event.title

    def test_notify_sentiment_shift(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            notify_sentiment_shift("AAPL", -0.2, 0.5)
            event = mock.call_args[0][0]
            assert event.channel == "sentiment"
            assert "AAPL" in event.title

    def test_notify_daily_synthesis(self):
        with patch("src.notifications.notifier.notify_async") as mock:
            synthesis = {
                "top_buys": [{"symbol": "NVDA"}, {"symbol": "CRM"}],
                "top_avoids": [{"symbol": "MRNA"}],
                "portfolio_risk": "moderate",
            }
            notify_daily_synthesis(synthesis)
            event = mock.call_args[0][0]
            assert event.channel == "daily"
            assert "NVDA" in event.message
