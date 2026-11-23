import inspect

import pandas as pd

from waddle.data import storage

START = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _bars(n, start=START, base=100.0):
    """Deterministic 1h OHLCV frame indexed by UTC timestamp."""
    idx = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [base + i for i in range(n)],
            "high": [base + i + 1.0 for i in range(n)],
            "low": [base + i - 1.0 for i in range(n)],
            "close": [base + i + 0.5 for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
        },
        index=idx,
    )


def test_count_is_zero_for_unknown_symbol_on_fresh_db(tmp_path):
    db = tmp_path / "fresh.db"

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 0


def test_count_is_zero_for_unknown_symbol_when_other_symbol_is_stored(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(5), "AVAX/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 0


def test_count_equals_number_of_saved_bars(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(24), "SOL/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 24


def test_rewriting_same_bars_does_not_change_count(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(10), "SOL/USDT", db_path=db)

    # Same (symbol, timestamp) keys with different values: upsert, not insert.
    storage.save_ohlcv(_bars(10, base=500.0), "SOL/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 10


def test_overlapping_save_counts_union_of_timestamps(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(10), "SOL/USDT", db_path=db)

    # Bars 5..14 overlap bars 5..9 already stored: 15 distinct timestamps.
    storage.save_ohlcv(_bars(10, start=START + pd.Timedelta(hours=5)), "SOL/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 15


def test_each_symbol_is_counted_separately_in_same_db(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(7), "SOL/USDT", db_path=db)
    storage.save_ohlcv(_bars(3), "AVAX/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 7
    assert storage.count_ohlcv("AVAX/USDT", db_path=db) == 3


def test_bybit_bars_are_not_counted(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv_bybit(_bars(12), "SOL/USDT:USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT:USDT", db_path=db) == 0


def test_only_binance_spot_bars_are_counted_when_both_sources_stored(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv_bybit(_bars(12), "SOL/USDT", db_path=db)
    storage.save_ohlcv(_bars(4), "SOL/USDT", db_path=db)

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 4


def test_creates_missing_parent_directory(tmp_path):
    db = tmp_path / "nested" / "data" / "waddle.db"
    assert not db.parent.exists()

    assert storage.count_ohlcv("SOL/USDT", db_path=db) == 0
    assert db.parent.is_dir()


def test_default_db_path_is_storage_default():
    # Signature only: never open data/waddle.db from a test.
    default = inspect.signature(storage.count_ohlcv).parameters["db_path"].default

    assert default == storage.DEFAULT_DB
