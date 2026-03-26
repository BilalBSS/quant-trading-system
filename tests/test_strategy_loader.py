# / tests for strategy loader — config validation, loading, saving

from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.strategies.strategy_loader import (
    StrategyConfig,
    SignalConfig,
    EntryConditionsConfig,
    ExitConditionsConfig,
    StopLossConfig,
    PositionSizingConfig,
    StrategyMetadata,
    FundamentalFiltersConfig,
    validate_config,
    load_config_file,
    load_all_configs,
    save_config,
    CONFIGS_DIR,
)


# / --- fixtures ---

def _base_signal():
    return {"indicator": "rsi", "condition": "below", "threshold": 30, "period": 14}


def _fundamental_gated_config():
    # / mirrors strategy_001 — has fundamental_filters + 3 signals
    return {
        "id": "test_fund_001",
        "name": "Test_Fundamental_Gated",
        "version": 1,
        "created_by": "test",
        "asset_class": "stocks",
        "universe": ["AAPL", "MSFT"],
        "fundamental_filters": {
            "pe_ratio_max": 40,
            "pe_vs_sector": "below_average",
            "revenue_growth_min": 0.01,
            "fcf_margin_min": 0.10,
        },
        "entry_conditions": {
            "operator": "AND",
            "signals": [
                {"indicator": "bollinger_bands", "condition": "price_below_lower", "lookback": 20, "std_dev": 2.0},
                {"indicator": "rsi", "condition": "below", "threshold": 35, "period": 14},
                {"indicator": "volume", "condition": "above_average", "multiplier": 1.3, "period": 20},
            ],
        },
        "exit_conditions": {
            "stop_loss": {"type": "atr_trailing", "multiplier": 2.0, "period": 14},
            "take_profit": {"indicator": "bollinger_bands", "condition": "price_above_middle"},
            "time_exit": {"max_holding_days": 30},
        },
        "position_sizing": {
            "method": "kelly_fraction",
            "max_position_pct": 0.08,
            "kelly_fraction": 0.25,
        },
        "metadata": {"generation": 1, "status": "backtest_pending"},
    }


def _momentum_only_config():
    # / mirrors strategy_002 — no fundamental_filters
    return {
        "id": "test_mom_001",
        "name": "Test_Momentum_Only",
        "version": 1,
        "created_by": "test",
        "asset_class": "stocks",
        "universe": ["AAPL", "MSFT", "GOOG"],
        "entry_conditions": {
            "operator": "AND",
            "signals": [
                {"indicator": "macd", "condition": "crossover_bullish"},
                {"indicator": "volume", "condition": "above_average", "multiplier": 1.2, "period": 20},
                {"indicator": "adx", "condition": "above", "threshold": 25, "period": 14},
            ],
        },
        "exit_conditions": {
            "stop_loss": {"type": "atr_trailing", "multiplier": 1.5, "period": 14},
            "take_profit": {"indicator": "bollinger_bands", "condition": "price_above_upper", "period": 20, "std_dev": 2.0},
            "time_exit": {"max_holding_days": 20},
        },
        "position_sizing": {
            "method": "strength_scaled",
            "max_position_pct": 0.04,
        },
        "metadata": {"generation": 1, "status": "backtest_pending"},
    }


def _minimal_config():
    # / smallest valid momentum-only config
    return {
        "id": "test_min_001",
        "name": "Minimal",
        "universe": ["AAPL"],
        "entry_conditions": {
            "signals": [_base_signal()],
        },
        "exit_conditions": {
            "stop_loss": {"type": "fixed_pct", "pct": 0.05},
        },
        "position_sizing": {"max_position_pct": 0.04},
    }


# / --- pydantic model validation ---

class TestValidFundamentalGatedConfig:
    # / fundamental-gated configs (strategy_001 pattern)

    def test_valid_full_config(self):
        config = validate_config(_fundamental_gated_config())
        assert config.id == "test_fund_001"
        assert config.fundamental_filters is not None
        assert len(config.entry_conditions.signals) == 3

    def test_fundamental_filters_parsed(self):
        config = validate_config(_fundamental_gated_config())
        assert config.fundamental_filters.pe_ratio_max == 40
        assert config.fundamental_filters.pe_vs_sector == "below_average"
        assert config.fundamental_filters.revenue_growth_min == 0.01

    def test_entry_conditions_parsed(self):
        config = validate_config(_fundamental_gated_config())
        assert config.entry_conditions.operator == "AND"
        signals = config.entry_conditions.signals
        assert signals[0].indicator == "bollinger_bands"
        assert signals[1].threshold == 35

    def test_exit_conditions_parsed(self):
        config = validate_config(_fundamental_gated_config())
        assert config.exit_conditions.stop_loss.type == "atr_trailing"
        assert config.exit_conditions.take_profit.condition == "price_above_middle"
        assert config.exit_conditions.time_exit.max_holding_days == 30

    def test_position_sizing_kelly(self):
        config = validate_config(_fundamental_gated_config())
        assert config.position_sizing.method == "kelly_fraction"
        assert config.position_sizing.max_position_pct == 0.08
        assert config.position_sizing.kelly_fraction == 0.25

    def test_metadata_defaults(self):
        config = validate_config(_fundamental_gated_config())
        assert config.metadata.generation == 1
        assert config.metadata.status == "backtest_pending"
        assert config.metadata.backtest_sharpe is None

    def test_max_position_at_8pct(self):
        # / fundamental-gated allows up to 8%
        raw = _fundamental_gated_config()
        raw["position_sizing"]["max_position_pct"] = 0.08
        config = validate_config(raw)
        assert config.position_sizing.max_position_pct == 0.08

    def test_or_operator(self):
        raw = _fundamental_gated_config()
        raw["entry_conditions"]["operator"] = "OR"
        config = validate_config(raw)
        assert config.entry_conditions.operator == "OR"

    def test_two_signals_minimum_with_fundamentals(self):
        # / fundamental-gated needs >= 2 signals
        raw = _fundamental_gated_config()
        raw["entry_conditions"]["signals"] = [_base_signal(), _base_signal()]
        config = validate_config(raw)
        assert len(config.entry_conditions.signals) == 2


class TestValidMomentumOnlyConfig:
    # / momentum-only configs (strategy_002 pattern)

    def test_valid_full_config(self):
        config = validate_config(_momentum_only_config())
        assert config.id == "test_mom_001"
        assert config.fundamental_filters is None

    def test_no_fundamental_filters(self):
        config = validate_config(_momentum_only_config())
        assert config.fundamental_filters is None

    def test_position_sizing_capped_at_4pct(self):
        config = validate_config(_momentum_only_config())
        assert config.position_sizing.max_position_pct == 0.04

    def test_minimal_config_defaults(self):
        config = validate_config(_minimal_config())
        assert config.version == 1
        assert config.created_by == "human"
        assert config.description == ""
        assert config.asset_class == "stocks"
        assert config.parent_id is None

    def test_single_signal_allowed(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = [_base_signal()]
        config = validate_config(raw)
        assert len(config.entry_conditions.signals) == 1

    def test_crypto_asset_class(self):
        raw = _minimal_config()
        raw["asset_class"] = "crypto"
        config = validate_config(raw)
        assert config.asset_class == "crypto"

    def test_mixed_asset_class(self):
        raw = _minimal_config()
        raw["asset_class"] = "mixed"
        config = validate_config(raw)
        assert config.asset_class == "mixed"


class TestInvalidOperator:
    def test_xor_operator_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["operator"] = "XOR"
        with pytest.raises(ValidationError, match="operator must be AND or OR"):
            validate_config(raw)

    def test_lowercase_and_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["operator"] = "and"
        with pytest.raises(ValidationError, match="operator must be AND or OR"):
            validate_config(raw)

    def test_empty_operator_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["operator"] = ""
        with pytest.raises(ValidationError, match="operator must be AND or OR"):
            validate_config(raw)


class TestEmptySignals:
    def test_empty_signals_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = []
        with pytest.raises(ValidationError, match="at least one entry signal"):
            validate_config(raw)


class TestPositionSizingAbove10Pct:
    def test_11pct_rejected(self):
        raw = _minimal_config()
        raw["position_sizing"]["max_position_pct"] = 0.11
        with pytest.raises(ValidationError, match="max_position_pct must be"):
            validate_config(raw)

    def test_exactly_10pct_allowed(self):
        # / 10% is the absolute max for the field validator, but track
        # / constraints will still reject it for both tracks
        raw = _minimal_config()
        raw["position_sizing"]["max_position_pct"] = 0.10
        # / field validator passes, but model validator rejects (momentum max is 4%)
        with pytest.raises(ValidationError, match="momentum-only max position is 4%"):
            validate_config(raw)

    def test_zero_rejected(self):
        raw = _minimal_config()
        raw["position_sizing"]["max_position_pct"] = 0.0
        with pytest.raises(ValidationError, match="max_position_pct must be"):
            validate_config(raw)

    def test_negative_rejected(self):
        raw = _minimal_config()
        raw["position_sizing"]["max_position_pct"] = -0.05
        with pytest.raises(ValidationError, match="max_position_pct must be"):
            validate_config(raw)


class TestFundamentalGatedPositionLimit:
    def test_9pct_rejected_for_fundamental(self):
        raw = _fundamental_gated_config()
        raw["position_sizing"]["max_position_pct"] = 0.09
        with pytest.raises(ValidationError, match="fundamental-gated max position is 8%"):
            validate_config(raw)

    def test_8pct_allowed_for_fundamental(self):
        raw = _fundamental_gated_config()
        raw["position_sizing"]["max_position_pct"] = 0.08
        config = validate_config(raw)
        assert config.position_sizing.max_position_pct == 0.08

    def test_5pct_allowed_for_fundamental(self):
        raw = _fundamental_gated_config()
        raw["position_sizing"]["max_position_pct"] = 0.05
        config = validate_config(raw)
        assert config.position_sizing.max_position_pct == 0.05


class TestMomentumOnlyPositionLimit:
    def test_5pct_rejected_for_momentum(self):
        raw = _momentum_only_config()
        raw["position_sizing"]["max_position_pct"] = 0.05
        with pytest.raises(ValidationError, match="momentum-only max position is 4%"):
            validate_config(raw)

    def test_4pct_allowed_for_momentum(self):
        raw = _momentum_only_config()
        raw["position_sizing"]["max_position_pct"] = 0.04
        config = validate_config(raw)
        assert config.position_sizing.max_position_pct == 0.04

    def test_2pct_allowed_for_momentum(self):
        raw = _momentum_only_config()
        raw["position_sizing"]["max_position_pct"] = 0.02
        config = validate_config(raw)
        assert config.position_sizing.max_position_pct == 0.02


class TestTooManyEntryConditions:
    def test_9_signals_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = [_base_signal() for _ in range(9)]
        with pytest.raises(ValidationError, match="max 8 entry conditions"):
            validate_config(raw)

    def test_8_signals_allowed(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = [_base_signal() for _ in range(8)]
        config = validate_config(raw)
        assert len(config.entry_conditions.signals) == 8

    def test_15_signals_rejected(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = [_base_signal() for _ in range(15)]
        with pytest.raises(ValidationError, match="max 8 entry conditions"):
            validate_config(raw)


class TestFundamentalGatedMinSignals:
    def test_1_signal_rejected_with_fundamentals(self):
        raw = _fundamental_gated_config()
        raw["entry_conditions"]["signals"] = [_base_signal()]
        with pytest.raises(ValidationError, match="at least 2 technical signals"):
            validate_config(raw)

    def test_2_signals_allowed_with_fundamentals(self):
        raw = _fundamental_gated_config()
        raw["entry_conditions"]["signals"] = [_base_signal(), _base_signal()]
        config = validate_config(raw)
        assert len(config.entry_conditions.signals) == 2


class TestInvalidStatus:
    def test_unknown_status_rejected(self):
        raw = _minimal_config()
        raw["metadata"] = {"status": "deployed"}
        with pytest.raises(ValidationError, match="status must be one of"):
            validate_config(raw)

    def test_empty_status_rejected(self):
        raw = _minimal_config()
        raw["metadata"] = {"status": ""}
        with pytest.raises(ValidationError, match="status must be one of"):
            validate_config(raw)

    @pytest.mark.parametrize("status", [
        "backtest_pending", "backtesting", "paper_trading", "live", "killed",
    ])
    def test_valid_statuses(self, status):
        raw = _minimal_config()
        raw["metadata"] = {"status": status}
        config = validate_config(raw)
        assert config.metadata.status == status


class TestInvalidAssetClass:
    def test_forex_rejected(self):
        raw = _minimal_config()
        raw["asset_class"] = "forex"
        with pytest.raises(ValidationError, match="asset_class must be stocks/crypto/mixed"):
            validate_config(raw)

    def test_empty_asset_class_rejected(self):
        raw = _minimal_config()
        raw["asset_class"] = ""
        with pytest.raises(ValidationError, match="asset_class must be stocks/crypto/mixed"):
            validate_config(raw)


class TestEmptyUniverse:
    def test_empty_universe_rejected(self):
        raw = _minimal_config()
        raw["universe"] = []
        with pytest.raises(ValidationError, match="universe must not be empty"):
            validate_config(raw)

    def test_single_symbol_allowed(self):
        raw = _minimal_config()
        raw["universe"] = ["BTC-USD"]
        config = validate_config(raw)
        # / list gets joined to comma-separated string
        assert config.universe == "BTC-USD"

    def test_string_universe_reference(self):
        raw = _minimal_config()
        raw["universe"] = "all_stocks"
        config = validate_config(raw)
        assert config.universe == "all_stocks"

    def test_empty_string_universe_rejected(self):
        raw = _minimal_config()
        raw["universe"] = ""
        with pytest.raises(ValidationError, match="universe must not be empty"):
            validate_config(raw)


# / --- load_config_file ---

class TestLoadConfigFile:
    def test_load_strategy_001(self):
        path = CONFIGS_DIR / "strategy_001.json"
        if not path.exists():
            pytest.skip("strategy_001.json not found on disk")
        strategy = load_config_file(path)
        assert strategy.strategy_id == "strategy_001"
        assert strategy.name == "Bollinger_PE_Oversold"
        assert strategy.requires_fundamentals is True

    def test_load_strategy_002(self):
        path = CONFIGS_DIR / "strategy_002.json"
        if not path.exists():
            pytest.skip("strategy_002.json not found on disk")
        strategy = load_config_file(path)
        assert strategy.strategy_id == "strategy_002"
        assert strategy.name == "MACD_Momentum_Breakout"
        assert strategy.requires_fundamentals is False

    def test_load_from_tmp(self, tmp_path):
        raw = _momentum_only_config()
        path = tmp_path / "test_strat.json"
        path.write_text(json.dumps(raw))
        strategy = load_config_file(path)
        assert strategy.strategy_id == "test_mom_001"
        # / config is pydantic-normalized (includes defaults, universe coerced)
        assert strategy.config["id"] == raw["id"]
        assert strategy.config["name"] == raw["name"]

    def test_load_fundamental_gated_from_tmp(self, tmp_path):
        raw = _fundamental_gated_config()
        path = tmp_path / "test_fund.json"
        path.write_text(json.dumps(raw))
        strategy = load_config_file(path)
        assert strategy.strategy_id == "test_fund_001"
        assert strategy.requires_fundamentals is True


# / --- load_all_configs ---

class TestLoadAllConfigs:
    def test_load_from_tmp_dir(self, tmp_path):
        # / write two configs to tmp dir
        for raw in [_fundamental_gated_config(), _momentum_only_config()]:
            path = tmp_path / f"{raw['id']}.json"
            path.write_text(json.dumps(raw))

        strategies = load_all_configs(directory=tmp_path)
        assert len(strategies) == 2
        ids = {s.strategy_id for s in strategies}
        assert ids == {"test_fund_001", "test_mom_001"}

    def test_load_sorted_by_filename(self, tmp_path):
        # / load_all_configs uses sorted() on glob results
        configs = [_fundamental_gated_config(), _momentum_only_config()]
        for raw in configs:
            path = tmp_path / f"{raw['id']}.json"
            path.write_text(json.dumps(raw))

        strategies = load_all_configs(directory=tmp_path)
        # / test_fund_001 < test_mom_001 alphabetically
        assert strategies[0].strategy_id == "test_fund_001"
        assert strategies[1].strategy_id == "test_mom_001"

    def test_load_empty_dir(self, tmp_path):
        strategies = load_all_configs(directory=tmp_path)
        assert strategies == []

    def test_skips_invalid_configs(self, tmp_path):
        # / one valid, one invalid
        valid = _momentum_only_config()
        (tmp_path / "valid.json").write_text(json.dumps(valid))

        invalid = _momentum_only_config()
        invalid["entry_conditions"]["signals"] = []
        (tmp_path / "invalid.json").write_text(json.dumps(invalid))

        strategies = load_all_configs(directory=tmp_path)
        assert len(strategies) == 1
        assert strategies[0].strategy_id == "test_mom_001"

    def test_skips_non_json_files(self, tmp_path):
        raw = _momentum_only_config()
        (tmp_path / "strat.json").write_text(json.dumps(raw))
        (tmp_path / "readme.txt").write_text("not a strategy")

        strategies = load_all_configs(directory=tmp_path)
        assert len(strategies) == 1


class TestLoadAllConfigsWithStatusFilter:
    def test_filter_by_status(self, tmp_path):
        # / one backtest_pending, one paper_trading
        pending = _momentum_only_config()
        pending["id"] = "pending_001"
        pending["metadata"] = {"status": "backtest_pending"}
        (tmp_path / "pending_001.json").write_text(json.dumps(pending))

        paper = _momentum_only_config()
        paper["id"] = "paper_001"
        paper["metadata"] = {"status": "paper_trading"}
        (tmp_path / "paper_001.json").write_text(json.dumps(paper))

        strategies = load_all_configs(
            directory=tmp_path, status_filter={"paper_trading"},
        )
        assert len(strategies) == 1
        assert strategies[0].strategy_id == "paper_001"

    def test_filter_multiple_statuses(self, tmp_path):
        for i, status in enumerate(["backtest_pending", "paper_trading", "live"]):
            raw = _momentum_only_config()
            raw["id"] = f"strat_{i:03d}"
            raw["metadata"] = {"status": status}
            (tmp_path / f"strat_{i:03d}.json").write_text(json.dumps(raw))

        strategies = load_all_configs(
            directory=tmp_path, status_filter={"paper_trading", "live"},
        )
        assert len(strategies) == 2
        ids = {s.strategy_id for s in strategies}
        assert ids == {"strat_001", "strat_002"}

    def test_filter_no_match(self, tmp_path):
        raw = _momentum_only_config()
        raw["metadata"] = {"status": "backtest_pending"}
        (tmp_path / f"{raw['id']}.json").write_text(json.dumps(raw))

        strategies = load_all_configs(
            directory=tmp_path, status_filter={"live"},
        )
        assert strategies == []

    def test_no_filter_returns_all(self, tmp_path):
        for i in range(3):
            raw = _momentum_only_config()
            raw["id"] = f"strat_{i:03d}"
            (tmp_path / f"strat_{i:03d}.json").write_text(json.dumps(raw))

        strategies = load_all_configs(directory=tmp_path, status_filter=None)
        assert len(strategies) == 3


# / --- save_config ---

class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        raw = _momentum_only_config()
        path = save_config(raw, directory=tmp_path)
        assert path.exists()
        assert path.name == "test_mom_001.json"

        # / reload and verify roundtrip
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["id"] == "test_mom_001"
        assert loaded["name"] == "Test_Momentum_Only"

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        raw = _fundamental_gated_config()
        path = save_config(raw, directory=nested)
        assert path.exists()
        assert nested.exists()

    def test_save_overwrites_existing(self, tmp_path):
        raw = _momentum_only_config()
        save_config(raw, directory=tmp_path)

        # / modify and save again
        raw["name"] = "Updated_Name"
        path = save_config(raw, directory=tmp_path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "Updated_Name"

    def test_save_uses_id_as_filename(self, tmp_path):
        raw = _minimal_config()
        path = save_config(raw, directory=tmp_path)
        assert path.name == "test_min_001.json"

    def test_save_invalid_config_rejected(self, tmp_path):
        # / save_config now validates before writing
        raw = {"name": "no_id"}
        with pytest.raises((ValueError, Exception)):
            save_config(raw, directory=tmp_path)

    def test_save_rejects_path_traversal_id(self, tmp_path):
        raw = _minimal_config()
        raw["id"] = "../../evil"
        with pytest.raises(ValueError, match="alphanumeric"):
            save_config(raw, directory=tmp_path)

    def test_save_then_load_config_file(self, tmp_path):
        # / full roundtrip: save raw dict -> load as ConfigDrivenStrategy
        raw = _momentum_only_config()
        path = save_config(raw, directory=tmp_path)
        strategy = load_config_file(path)
        assert strategy.strategy_id == "test_mom_001"
        # / config is pydantic-normalized (e.g. universe list -> string)
        assert strategy.config["id"] == raw["id"]
        assert strategy.config["name"] == raw["name"]

    def test_save_preserves_json_formatting(self, tmp_path):
        raw = _minimal_config()
        path = save_config(raw, directory=tmp_path)
        content = path.read_text()
        # / save_config uses indent=2
        assert "\n" in content
        assert "  " in content


# / --- validate_config ---

class TestValidateConfig:
    def test_returns_strategy_config(self):
        raw = _minimal_config()
        config = validate_config(raw)
        assert isinstance(config, StrategyConfig)

    def test_validates_fundamental_gated(self):
        config = validate_config(_fundamental_gated_config())
        assert config.fundamental_filters is not None

    def test_validates_momentum_only(self):
        config = validate_config(_momentum_only_config())
        assert config.fundamental_filters is None

    def test_rejects_invalid(self):
        raw = _minimal_config()
        raw["universe"] = []
        with pytest.raises(ValidationError):
            validate_config(raw)


# / --- edge cases ---

class TestEdgeCases:
    def test_missing_directory_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent"
        strategies = load_all_configs(directory=missing)
        assert strategies == []

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")
        valid = _momentum_only_config()
        (tmp_path / f"{valid['id']}.json").write_text(json.dumps(valid))

        strategies = load_all_configs(directory=tmp_path)
        assert len(strategies) == 1

    def test_partial_config_missing_id(self):
        raw = {
            "name": "NoId",
            "universe": ["AAPL"],
            "entry_conditions": {"signals": [_base_signal()]},
            "exit_conditions": {"stop_loss": {"type": "fixed_pct"}},
            "position_sizing": {"max_position_pct": 0.04},
        }
        with pytest.raises(ValidationError):
            validate_config(raw)

    def test_partial_config_missing_entry_conditions(self):
        raw = {
            "id": "missing_entry",
            "name": "NoEntry",
            "universe": ["AAPL"],
            "exit_conditions": {"stop_loss": {"type": "fixed_pct"}},
        }
        with pytest.raises(ValidationError):
            validate_config(raw)

    def test_partial_config_missing_exit_conditions(self):
        raw = {
            "id": "missing_exit",
            "name": "NoExit",
            "universe": ["AAPL"],
            "entry_conditions": {"signals": [_base_signal()]},
        }
        with pytest.raises(ValidationError):
            validate_config(raw)

    def test_partial_config_missing_universe(self):
        raw = {
            "id": "missing_universe",
            "name": "NoUniverse",
            "entry_conditions": {"signals": [_base_signal()]},
            "exit_conditions": {"stop_loss": {"type": "fixed_pct"}},
        }
        with pytest.raises(ValidationError):
            validate_config(raw)

    def test_extra_fields_ignored(self):
        raw = _minimal_config()
        raw["extra_field"] = "should be ignored"
        raw["another_extra"] = 42
        # / pydantic ignores extra fields by default
        config = validate_config(raw)
        assert config.id == "test_min_001"

    def test_signal_with_all_optional_fields(self):
        raw = _minimal_config()
        raw["entry_conditions"]["signals"] = [{
            "indicator": "bollinger_bands",
            "condition": "price_below_lower",
            "period": 20,
            "lookback": 20,
            "threshold": 2.0,
            "std_dev": 2.0,
            "multiplier": 1.5,
        }]
        config = validate_config(raw)
        sig = config.entry_conditions.signals[0]
        assert sig.period == 20
        assert sig.lookback == 20
        assert sig.std_dev == 2.0

    def test_load_config_file_invalid_json(self, tmp_path):
        bad_path = tmp_path / "broken.json"
        bad_path.write_text("not json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_config_file(bad_path)

    def test_load_config_file_invalid_schema(self, tmp_path):
        invalid = {"id": "bad", "name": "Bad"}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValidationError):
            load_config_file(path)

    def test_load_config_file_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            load_config_file(Path("/nonexistent/strategy.json"))

    def test_deeply_nested_fundamental_filters(self):
        raw = _fundamental_gated_config()
        raw["fundamental_filters"] = {
            "pe_ratio_max": 50,
            "pe_vs_sector": "below_average",
            "revenue_growth_min": 0.05,
            "fcf_margin_min": 0.20,
            "debt_to_equity_max": 1.0,
            "dcf_upside_min": 0.15,
            "insider_buying_recent": True,
        }
        config = validate_config(raw)
        assert config.fundamental_filters.insider_buying_recent is True
        assert config.fundamental_filters.dcf_upside_min == 0.15

    def test_metadata_with_backtest_results(self):
        raw = _minimal_config()
        raw["metadata"] = {
            "generation": 5,
            "status": "paper_trading",
            "backtest_sharpe": 1.42,
            "backtest_max_drawdown": -0.12,
            "backtest_win_rate": 0.58,
            "brier_score": 0.18,
            "paper_trade_days": 14,
            "paper_trade_pnl": 0.034,
        }
        config = validate_config(raw)
        assert config.metadata.backtest_sharpe == 1.42
        assert config.metadata.paper_trade_days == 14
