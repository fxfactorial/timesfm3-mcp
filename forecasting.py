"""TimesFM-3 forecast helpers shared by the local and cloud MCP servers.

Shapes match Google's official TimesFM 3.0 API:

- Univariate context: 1D array of length T
- Multivariate context: 2D array of shape (V, T)
- past_only_covariates: (C_past, T)
- past_future_covariates: (C_future, T + H)
- Univariate output: forecast (H,), quantiles (H, 9)
- Multivariate output: forecast (V, H), quantiles (V, H, 9)

Optional calendar labels (start + freq, or a history timestamp list) are
applied after inference. They are never passed to TimesFM-3.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timedelta
from typing import Any

import numpy as np

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHECKPOINT = "google/timesfm-3.0-pytorch"
LICENSE_NOTE = (
    "TimesFM-3 pretrained weights (google/timesfm-3.0-pytorch) are released "
    "under Google's timesfm-non-commercial-license-v1.0. They may be used for "
    "research, evaluation, and non-production experiments only — not for "
    "commercial or production systems. This MCP server's own code is Apache-2.0."
)

FREQ_ALIASES = {
    "h": "H",
    "hour": "H",
    "hours": "H",
    "d": "D",
    "day": "D",
    "days": "D",
    "w": "W",
    "week": "W",
    "weeks": "W",
    "m": "M",
    "month": "M",
    "months": "M",
}


def serialize_quantiles(quantiles: Any) -> dict[str, list[float]] | None:
    if quantiles is None:
        return None
    q = np.asarray(quantiles)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.ndim != 2:
        raise ValueError(f"quantiles must be 1D or 2D (H, Q), got shape {q.shape}")
    packed: dict[str, list[float]] = {}
    n_q = int(q.shape[-1])
    for i in range(n_q):
        if i < len(QUANTILE_LEVELS):
            key = f"q{int(round(QUANTILE_LEVELS[i] * 100))}"
        else:
            key = f"q_index_{i}"
        packed[key] = q[:, i].astype(float).tolist()
    return packed


def resolve_series(
    series: list[list[float]] | None,
    history: list[float] | None,
) -> np.ndarray:
    if series is not None and history is not None:
        raise ValueError("pass series or history, not both")
    if history is not None:
        if not history:
            raise ValueError("history must be a non-empty list of numbers")
        return np.asarray(history, dtype=np.float32).reshape(1, -1)
    if series is None or not series:
        raise ValueError("series must be a non-empty list of series")
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for i, row in enumerate(series):
        if not row:
            raise ValueError(f"series[{i}] must be a non-empty list of numbers")
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        rows.append(arr)
        lengths.append(int(arr.size))
    if len(set(lengths)) != 1:
        raise ValueError("all series rows must have the same length (context T)")
    return np.stack(rows, axis=0)


def _as_2d_channels(name: str, rows: list[list[float]], expected_len: int) -> np.ndarray:
    if not rows:
        raise ValueError(f"{name} must contain at least one channel")
    arrays: list[np.ndarray] = []
    for i, row in enumerate(rows):
        if not row:
            raise ValueError(f"{name}[{i}] must be a non-empty list of numbers")
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if int(arr.size) != expected_len:
            raise ValueError(
                f"{name}[{i}] length is {int(arr.size)}, expected {expected_len}"
            )
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def _is_date_only(value: str) -> bool:
    s = str(value).strip()
    return len(s) == 10 and s[4] == "-" and s[7] == "-"


def parse_timestamp(value: str) -> datetime:
    s = str(value).strip()
    if not s:
        raise ValueError("timestamp is empty")
    if _is_date_only(s):
        return datetime.fromisoformat(s)
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _format_timestamp(dt: datetime, date_only: bool) -> str:
    if date_only:
        return dt.date().isoformat()
    return dt.replace(microsecond=0).isoformat()


def _add_months(dt: datetime, n: int) -> datetime:
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def add_steps(dt: datetime, freq: str, n: int) -> datetime:
    if n == 0:
        return dt
    if freq == "H":
        return dt + timedelta(hours=n)
    if freq == "D":
        return dt + timedelta(days=n)
    if freq == "W":
        return dt + timedelta(weeks=n)
    if freq == "M":
        return _add_months(dt, n)
    raise ValueError(f"freq must be one of H, D, W, M (got {freq!r})")


def _normalize_freq(freq: str) -> str:
    key = FREQ_ALIASES.get(str(freq).strip().lower())
    if key is None:
        raise ValueError(f"freq must be one of H, D, W, M (got {freq!r})")
    return key


def _freq_from_delta(delta: timedelta) -> str | None:
    seconds = int(delta.total_seconds())
    if seconds == 3600:
        return "H"
    if seconds == 86400:
        return "D"
    if seconds == 86400 * 7:
        return "W"
    return None


def resolve_calendar(
    *,
    context_length: int,
    horizon: int,
    start: str | None = None,
    freq: str | None = None,
    timestamps: list[str] | None = None,
) -> dict[str, Any] | None:
    has_start = start is not None and str(start).strip() != ""
    has_freq = freq is not None and str(freq).strip() != ""
    has_ts = timestamps is not None

    if not has_start and not has_freq and not has_ts:
        return None
    if has_ts and (has_start or has_freq):
        raise ValueError("pass timestamps or start+freq, not both")
    if has_ts:
        return _calendar_from_timestamps(timestamps, context_length, horizon)
    if has_start ^ has_freq:
        raise ValueError("start and freq must be provided together")
    return _calendar_from_start_freq(str(start), str(freq), context_length, horizon)


def _calendar_from_start_freq(
    start: str, freq: str, context_length: int, horizon: int
) -> dict[str, Any]:
    freq_key = _normalize_freq(freq)
    origin = parse_timestamp(start)
    date_only = _is_date_only(start) and freq_key != "H"
    history_end = add_steps(origin, freq_key, context_length - 1)
    forecast_timestamps = [
        _format_timestamp(add_steps(origin, freq_key, context_length + i), date_only)
        for i in range(horizon)
    ]
    return {
        "freq": freq_key,
        "history_end": _format_timestamp(history_end, date_only),
        "forecast_timestamps": forecast_timestamps,
    }


def _calendar_from_timestamps(
    timestamps: list[str], context_length: int, horizon: int
) -> dict[str, Any]:
    if len(timestamps) != context_length:
        raise ValueError(
            f"timestamps length is {len(timestamps)}, expected {context_length}"
        )
    parsed = [parse_timestamp(value) for value in timestamps]
    if any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if context_length < 2:
        raise ValueError("need at least two timestamps to check spacing, or pass start and freq")
    deltas = [later - earlier for earlier, later in zip(parsed, parsed[1:])]
    if any(delta != deltas[0] for delta in deltas[1:]):
        raise ValueError(
            "timestamps are not strictly regular; missing observations are not filled"
        )
    step = deltas[0]
    date_only = all(_is_date_only(value) for value in timestamps)
    last = parsed[-1]
    forecast_timestamps = [
        _format_timestamp(last + step * (i + 1), date_only) for i in range(horizon)
    ]
    inferred = _freq_from_delta(step)
    payload = {
        "history_end": _format_timestamp(last, date_only),
        "forecast_timestamps": forecast_timestamps,
    }
    if inferred is not None:
        payload["freq"] = inferred
    return payload


def _quantile_index(level: float) -> int:
    for i, lv in enumerate(QUANTILE_LEVELS):
        if abs(lv - float(level)) < 1e-9:
            return i
    raise ValueError(f"level {level} is not one of the quantile levels {QUANTILE_LEVELS}")


def resolve_windows(windows: list[list[float]] | None) -> list[np.ndarray]:
    """Validate N univariate contexts of a single shared length T."""
    if windows is None or not windows:
        raise ValueError("windows must be a non-empty list of contexts")
    rows: list[np.ndarray] = []
    for i, w in enumerate(windows):
        if not w:
            raise ValueError(f"windows[{i}] must be a non-empty list of numbers")
        arr = np.asarray(w, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError(f"windows[{i}] contains a non-finite value")
        rows.append(arr)
    t = int(rows[0].size)
    if any(int(r.size) != t for r in rows):
        raise ValueError("all windows must have the same context length T")
    return rows


def resolve_asof(
    asof_timestamps: list[str] | None, n_windows: int
) -> tuple[list[datetime] | None, timedelta | None]:
    """Resolve per-window 'as-of' labels (the bar each window forecasts from).

    Must be length n_windows, strictly increasing, and strictly regular so a
    single step can label every window's forecast grid. Returns (parsed, step)
    where step is None when n_windows == 1 (no spacing to infer).
    """
    if asof_timestamps is None:
        return None, None
    if len(asof_timestamps) != n_windows:
        raise ValueError(
            f"asof_timestamps length is {len(asof_timestamps)}, expected {n_windows}"
        )
    parsed = [parse_timestamp(v) for v in asof_timestamps]
    if any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
        raise ValueError("asof_timestamps must be strictly increasing")
    if len(parsed) < 2:
        return parsed, None
    deltas = [later - earlier for earlier, later in zip(parsed, parsed[1:])]
    if any(delta != deltas[0] for delta in deltas[1:]):
        raise ValueError("asof_timestamps are not strictly regular")
    return parsed, deltas[0]


def _asof_format(
    raw: list[str], parsed: list[datetime], step: timedelta | None, horizon: int
) -> list[dict[str, Any]]:
    # date_only is decided from the RAW strings: a datetime parsed from
    # "2026-09-01" stringifies to "2026-09-01 00:00:00" and would never be
    # date-only-detected, so date-only inputs must echo as dates, not T00:00:00.
    date_only = all(_is_date_only(v) for v in raw)
    out: list[dict[str, Any]] = []
    for i, ts in enumerate(parsed):
        item: dict[str, Any] = {"asof": _format_timestamp(ts, date_only)}
        if step is not None:
            item["timestamps"] = [
                _format_timestamp(ts + step * (j + 1), date_only) for j in range(horizon)
            ]
        out.append(item)
    return out


def _predict_batch(forecaster: Any, contexts: list[np.ndarray], horizon: int) -> list[Any]:
    return list(
        forecaster.predict_batch(
            contexts,
            horizon=int(horizon),
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )


def run_forecast_batch(
    forecaster: Any,
    *,
    windows: list[list[float]],
    horizon: int = 96,
    window_ids: list[str] | None = None,
    asof_timestamps: list[str] | None = None,
) -> dict:
    """Batch zero-shot forecast: N independent univariate contexts in ONE decode.

    Equivalent to calling run_forecast(history=w, horizon=horizon) for each
    window, but a single batched predict_batch call. All windows must share the
    same context length T. Very large batches can be chunked by the caller.

    asof_timestamps (length N, strictly regular) labels each window with the
    bar it forecasts from plus the full forecast-step grid, so consumers never
    pin results by position alone.
    """
    try:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        contexts = resolve_windows(windows)
        n = len(contexts)
        if window_ids is not None and len(window_ids) != n:
            raise ValueError(
                f"window_ids length is {len(window_ids)}, expected {n}"
            )
        parsed, step = resolve_asof(asof_timestamps, n)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    try:
        outputs = _predict_batch(forecaster, contexts, horizon)
    except Exception as exc:  # GPU-side failure fails the whole batch
        return {"status": "error", "error": f"batch decode failed: {exc}"}
    if len(outputs) != n:
        return {
            "status": "error",
            "error": f"model returned {len(outputs)} outputs, expected {n}",
        }

    stamped = (
        _asof_format(asof_timestamps, parsed, step, int(horizon))
        if parsed is not None
        else None
    )
    items: list[dict[str, Any]] = []
    for i, out in enumerate(outputs):
        forecast = np.asarray(out.forecast)
        if forecast.ndim != 1:
            return {
                "status": "error",
                "error": f"window {i}: unexpected forecast shape {forecast.shape}",
            }
        quantiles = None if getattr(out, "quantiles", None) is None else np.asarray(out.quantiles)
        if quantiles is not None:
            if quantiles.ndim == 1:
                quantiles = quantiles.reshape(-1, 1)
            if quantiles.ndim != 2:
                return {
                    "status": "error",
                    "error": f"window {i}: unexpected quantiles shape {quantiles.shape}",
                }
        item: dict[str, Any] = {
            "id": window_ids[i] if window_ids is not None else f"window_{i}",
            "forecast": forecast.astype(float).tolist(),
            "quantiles": serialize_quantiles(quantiles),
        }
        if stamped is not None:
            item.update(stamped[i])
        items.append(item)

    return {
        "status": "success",
        "model": CHECKPOINT,
        "mode": "batch",
        "n_windows": n,
        "context_length": int(contexts[0].size),
        "horizon": int(horizon),
        "windows": items,
        "quantile_levels": QUANTILE_LEVELS,
        "license": LICENSE_NOTE,
    }


def run_widths(
    forecaster: Any,
    *,
    windows: list[list[float]],
    horizon: int = 96,
    lower: float = 0.10,
    upper: float = 0.90,
    with_median: bool = True,
    asof_timestamps: list[str] | None = None,
) -> dict:
    """Batch quantile-width forecast: per-window upper-lower quantile range at
    the final forward step, plus the median. A compact payload for volatility-
    band consumers that would otherwise ship (and reduce) nine full quantile
    curves per window. The range is base-invariant (a shared offset cancels).
    """
    try:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        contexts = resolve_windows(windows)
        n = len(contexts)
        li = _quantile_index(lower)
        ui = _quantile_index(upper)
        if li >= ui:
            raise ValueError("lower must be strictly less than upper")
        mi = _quantile_index(0.5)  # 0.5 is a fixed level, always present
        parsed, step = resolve_asof(asof_timestamps, n)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    try:
        outputs = _predict_batch(forecaster, contexts, horizon)
    except Exception as exc:
        return {"status": "error", "error": f"batch decode failed: {exc}"}
    if len(outputs) != n:
        return {
            "status": "error",
            "error": f"model returned {len(outputs)} outputs, expected {n}",
        }

    stamped = (
        _asof_format(asof_timestamps, parsed, step, int(horizon))
        if parsed is not None
        else None
    )
    items: list[dict[str, Any]] = []
    for i, out in enumerate(outputs):
        q = getattr(out, "quantiles", None)
        if q is None:
            return {"status": "error", "error": f"window {i}: model returned no quantiles"}
        q = np.asarray(q)
        if q.ndim == 1:
            q = q.reshape(-1, 1)
        if q.ndim != 2 or int(q.shape[0]) != int(horizon):
            return {
                "status": "error",
                "error": f"window {i}: unexpected quantiles shape {q.shape}",
            }
        row = q[int(horizon) - 1]
        item: dict[str, Any] = {"id": f"window_{i}"}
        if stamped is not None:
            item.update(stamped[i])
        item["width"] = float(row[ui] - row[li])
        if with_median:
            item["median"] = float(row[mi])
        items.append(item)

    payload: dict[str, Any] = {
        "status": "success",
        "model": CHECKPOINT,
        "mode": "widths",
        "n_windows": n,
        "context_length": int(contexts[0].size),
        "horizon": int(horizon),
        "lower": float(lower),
        "upper": float(upper),
        "windows": items,
        "widths": [it["width"] for it in items],
        "quantile_levels": QUANTILE_LEVELS,
        "license": LICENSE_NOTE,
    }
    if with_median:
        payload["medians"] = [it["median"] for it in items]
    return payload


def run_forecast(
    forecaster: Any,
    *,
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
    try:
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        target = resolve_series(series, history)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    n_series = int(target.shape[0])
    context_length = int(target.shape[1])

    if series_ids is not None and len(series_ids) != n_series:
        return {
            "status": "error",
            "error": f"series_ids length is {len(series_ids)}, expected {n_series}",
        }

    po = None
    pf = None
    try:
        if past_covariates:
            po = [_as_2d_channels("past_covariates", past_covariates, context_length)]
        if future_covariates:
            pf = [
                _as_2d_channels(
                    "future_covariates",
                    future_covariates,
                    context_length + int(horizon),
                )
            ]
        calendar = resolve_calendar(
            context_length=context_length,
            horizon=int(horizon),
            start=start,
            freq=freq,
            timestamps=timestamps,
        )
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    # Official univariate path is a 1D context. Covariates and V>1 need (V, T).
    if n_series == 1 and po is None and pf is None:
        contexts: list[np.ndarray] = [target.reshape(-1)]
    else:
        contexts = [target]

    outputs = list(
        forecaster.predict_batch(
            contexts,
            horizon=int(horizon),
            past_only_covariates=po,
            past_future_covariates=pf,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    if not outputs:
        return {"status": "error", "error": "model returned no forecast"}

    first = outputs[0]
    forecast = np.asarray(first.forecast)
    quantiles = None if getattr(first, "quantiles", None) is None else np.asarray(first.quantiles)

    if forecast.ndim == 1:
        forecast = forecast.reshape(1, -1)
    elif forecast.ndim != 2:
        return {
            "status": "error",
            "error": f"unexpected forecast shape {forecast.shape}",
        }

    if quantiles is not None:
        if quantiles.ndim == 2:
            quantiles = quantiles.reshape(1, quantiles.shape[0], quantiles.shape[1])
        elif quantiles.ndim != 3:
            return {
                "status": "error",
                "error": f"unexpected quantiles shape {quantiles.shape}",
            }

    if forecast.shape[0] != n_series:
        return {
            "status": "error",
            "error": (
                f"model returned {forecast.shape[0]} series, expected {n_series}"
            ),
        }

    stamp = None if calendar is None else calendar["forecast_timestamps"]
    items = []
    for i in range(n_series):
        q_i = None if quantiles is None else quantiles[i]
        sid = series_ids[i] if series_ids is not None else f"series_{i}"
        item: dict[str, Any] = {
            "id": sid,
            "forecast": forecast[i].astype(float).tolist(),
            "quantiles": serialize_quantiles(q_i),
        }
        if stamp is not None:
            item["timestamps"] = stamp
        items.append(item)

    payload: dict[str, Any] = {
        "status": "success",
        "model": CHECKPOINT,
        "mode": "univariate" if n_series == 1 else "multivariate",
        "n_series": n_series,
        "context_length": context_length,
        "horizon": int(horizon),
        "series": items,
        "quantile_levels": QUANTILE_LEVELS,
        "license": LICENSE_NOTE,
    }
    if calendar is not None:
        if calendar.get("freq") is not None:
            payload["freq"] = calendar["freq"]
        payload["history_end"] = calendar["history_end"]
        if n_series == 1:
            payload["timestamps"] = stamp
    if n_series == 1:
        payload["forecast"] = items[0]["forecast"]
        payload["quantiles"] = items[0]["quantiles"]
    return payload
