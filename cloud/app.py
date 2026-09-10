import os
import sys
from pathlib import Path

import torch
from fastmcp import FastMCP
from huggingface_hub import login
from timesfm3 import ModelConfig, TimesFM3Evaluator

ROOT = Path(__file__).resolve().parents[1]
for path in (Path(__file__).resolve().parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting import run_forecast, run_forecast_batch, run_widths  # noqa: E402

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    sys.stderr.write("Authenticating with Hugging Face...\n")
    login(token=hf_token)
else:
    sys.stderr.write("WARNING: HF_TOKEN is not set; gated weight download may fail.\n")

mcp = FastMCP("TimesFM-3")

sys.stderr.write("Loading TimesFM-3 into memory...\n")
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.stderr.write(f"Device: {device}\n")
config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device=device)
forecaster = TimesFM3Evaluator(config)
sys.stderr.write("Model ready.\n")


@mcp.tool()
def forecast(
    series: list[list[float]] | None = None,
    history: list[float] | None = None,
    horizon: int = 5,
    series_ids: list[str] | None = None,
    past_covariates: list[list[float]] | None = None,
    future_covariates: list[list[float]] | None = None,
    start: str | None = None,
    freq: str | None = None,
    timestamps: list[str] | None = None,
) -> dict:
    """Zero-shot forecast with Google TimesFM-3.

    Pass one series (univariate) or several related series (joint multivariate).
    Optional past_covariates must be length T. Optional future_covariates
    (known ahead, e.g. promo flags) must be length T + horizon.

    Optional start + freq (or a history timestamp list) labels the forecast
    with ISO dates. Irregular timestamp lists are rejected; gaps are not filled.

    Returns a median point forecast plus nine quantile bands (q10 through q90)
    per series.

    TimesFM-3 weights are licensed for non-commercial, non-production use only.

    Args:
        series: Target series, each a chronological list of the same length T.
            One row is univariate. Two or more rows are forecast jointly.
        history: Back-compat shortcut for a single series. Do not pass with series.
        horizon: Number of future steps. Must be >= 1.
        series_ids: Optional names, one per series row.
        past_covariates: Channels known only in the past. Each length T.
        future_covariates: Channels known in the past and future. Each length T+horizon.
        start: ISO date/time of the first history point. Requires freq.
        freq: Spacing of the series: H, D, W, or M.
        timestamps: History timestamps, length T, strictly regular. Do not pass with start/freq.
    """
    return run_forecast(
        forecaster,
        series=series,
        history=history,
        horizon=horizon,
        series_ids=series_ids,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=start,
        freq=freq,
        timestamps=timestamps,
    )


@mcp.tool()
def forecast_demand(history: list[float], horizon: int = 5) -> dict:
    """Deprecated alias of `forecast` for a single series. Prefer `forecast`.

    Same TimesFM-3 non-commercial license limit.
    """
    return run_forecast(forecaster, history=history, horizon=horizon)


@mcp.tool()
def forecast_batch(
    windows: list[list[float]],
    horizon: int = 96,
    window_ids: list[str] | None = None,
    asof_timestamps: list[str] | None = None,
) -> dict:
    """Batch zero-shot forecast: N independent univariate contexts in ONE decode.

    Equivalent to calling `forecast` per window, but a single batched model
    pass — much faster for backfills and studies. All windows must have the
    same context length T.

    Optional asof_timestamps (length N, strictly regular ISO dates) labels each
    window with the bar it forecasts from plus the full forecast-step grid, so
    results are never pinned by position alone.

    TimesFM-3 weights are licensed for non-commercial, non-production use only.

    Args:
        windows: N univariate contexts, each a chronological list of length T.
        horizon: Number of future steps per window. Must be >= 1.
        window_ids: Optional names, one per window.
        asof_timestamps: Optional ISO timestamps, one per window, strictly regular.
    """
    return run_forecast_batch(
        forecaster,
        windows=windows,
        horizon=horizon,
        window_ids=window_ids,
        asof_timestamps=asof_timestamps,
    )


@mcp.tool()
def widths(
    windows: list[list[float]],
    horizon: int = 96,
    lower: float = 0.10,
    upper: float = 0.90,
    with_median: bool = True,
    asof_timestamps: list[str] | None = None,
) -> dict:
    """Batch quantile widths: per-window (upper - lower) quantile range at the
    final forward step, plus the median. A compact payload for volatility-band
    consumers that would otherwise ship and reduce nine full quantile curves
    per window. The range is base-invariant (a shared level offset cancels).

    lower / upper must be quantile levels the model returns (0.10-0.90).
    Optional asof_timestamps (length N, strictly regular) labels each window.

    TimesFM-3 weights are licensed for non-commercial, non-production use only.

    Args:
        windows: N univariate contexts, each a chronological list of length T.
        horizon: Number of future steps per window. Must be >= 1.
        lower: Lower quantile level for the width (default 0.10).
        upper: Upper quantile level for the width (default 0.90).
        with_median: Also return the q50 at the final step (default true).
        asof_timestamps: Optional ISO timestamps, one per window, strictly regular.
    """
    return run_widths(
        forecaster,
        windows=windows,
        horizon=horizon,
        lower=lower,
        upper=upper,
        with_median=with_median,
        asof_timestamps=asof_timestamps,
    )


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=7860)
