"""Shared column constants for feature pipelines."""

METADATA_COLS = {
    "ticker",
    "datetime_et",
    "datetime_utc",
    "date_et",
    "t_ms",
}

PRICE_COLS = {"open", "high", "low", "close", "volume", "vwap", "num_trades"}

TARGET_COLS = {"target_returns", "target_direction", "target_price"}


def feature_columns(columns) -> list:
    exclude = METADATA_COLS | PRICE_COLS | TARGET_COLS
    return [c for c in columns if c not in exclude]
