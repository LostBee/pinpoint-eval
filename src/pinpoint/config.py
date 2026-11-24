"""Configuration for Pinpoint (pinpoint-eval)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

SCORER_VERSION = "v2"

TIER_WEIGHTS: dict[str, int] = {
    "continent": 1,
    "country": 2,
    "100km": 3,
    "10km": 4,
    "1km": 5,
}
MAX_TIER_POINTS = sum(TIER_WEIGHTS.values())  # 15

# OpenRouter unified reasoning.effort nomenclature (matches Inspect AI GenerateConfig.reasoning_effort)
VALID_REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def normalize_reasoning_effort(value: str | None) -> str | None:
    """Validate OpenRouter reasoning.effort values; blank/'default' returns None (model default)."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in ("", "default"):
        return None
    if cleaned not in VALID_REASONING_EFFORTS:
        valid = ", ".join(VALID_REASONING_EFFORTS)
        raise ValueError(f"Unsupported reasoning_effort '{value}'. Expected one of: {valid}, or blank/default.")
    return cleaned


def format_model_label(openrouter_id: str, reasoning_effort: str | None = None) -> str:
    """Format human-readable model label including reasoning effort when configured."""
    base = openrouter_id.removeprefix("openrouter/")
    return f"{base} [{reasoning_effort}]" if reasoning_effort else base


def _parse_optional_cost(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if cleaned in ("", "none", "null", "0", "0.0"):
        return None
    val = float(cleaned)
    return val if val > 0 else None


@dataclass(frozen=True)
class PinpointConfig:
    repo_root: Path = REPO_ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PINPOINT_DATA_DIR") or (REPO_ROOT / "data")
        ).resolve()
    )
    max_long_edge: int = 1536
    jpeg_quality: int = 85
    max_search_calls: int = 5
    tavily_max_results: int = 5
    default_temperature: float = 0.0
    default_max_tokens: int = 16384
    default_timeout_s: int = 180
    default_epochs: int = 1
    max_run_cost_usd: float | None = field(
        default_factory=lambda: _parse_optional_cost(os.getenv("PINPOINT_MAX_RUN_COST_USD"))
    )

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def vault_dir(self) -> Path:
        return self.data_dir / "vault"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pinpoint.sqlite"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.inbox_dir,
            self.vault_dir,
            self.images_dir,
            self.logs_dir,
            self.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> PinpointConfig:
    cfg = PinpointConfig()
    cfg.ensure_dirs()
    return cfg
