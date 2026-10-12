"""Environment configuration: no trading keys are used by P0."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    data_root: Path
    cors_origins: tuple[str, ...]
    binance_fapi_base_url: str
    binance_data_base_url: str


def get_settings() -> Settings:
    origins = tuple(
        item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if item.strip()
    )
    return Settings(
        data_root=Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve(),
        cors_origins=origins,
        binance_fapi_base_url=os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com").rstrip("/"),
        binance_data_base_url=os.getenv("BINANCE_DATA_BASE_URL", "https://data.binance.vision").rstrip("/"),
    )
