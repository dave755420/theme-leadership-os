"""Cross-sectional validation metrics (rank IC, spreads, and hits)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _wide(value: Any, *, value_name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        # Accept a long-form frame without making callers pivot it first.
        lower = {str(c).lower(): c for c in frame.columns}
        date_col = next(
            (lower[c] for c in ("date", "decision_at", "timestamp") if c in lower), None
        )
        asset_col = next(
            (lower[c] for c in ("asset", "symbol", "ticker", "theme") if c in lower), None
        )
        value_col = next(
            (
                lower[c]
                for c in (value_name, "value", "score", "return", "forward_return")
                if c in lower
            ),
            None,
        )
        if (
            date_col is not None
            and asset_col is not None
            and value_col is not None
            and frame.index.nlevels == 1
        ):
            frame = frame.pivot(index=date_col, columns=asset_col, values=value_col)
        return frame
    if isinstance(value, pd.Series):
        if isinstance(value.index, pd.MultiIndex) and value.index.nlevels >= 2:
            frame = value.rename(value_name).reset_index()
            frame.columns = ["date", "asset", value_name] + list(frame.columns[3:])
            return frame.pivot(index="date", columns="asset", values=value_name)
        return value.to_frame(value.name or value_name)
    if isinstance(value, Mapping):
        return pd.DataFrame(value)
    return pd.DataFrame(value)


def _align(
    scores: Any, forward_returns: Any, benchmark_returns: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series | None]:
    s = _wide(scores, value_name="score")
    r = _wide(forward_returns, value_name="forward_return")
    s, r = s.align(r, join="inner", axis=0)
    s, r = s.align(r, join="inner", axis=1)
    benchmark = None
    if benchmark_returns is not None:
        benchmark = (
            benchmark_returns.copy()
            if isinstance(benchmark_returns, (pd.Series, pd.DataFrame))
            else pd.Series(benchmark_returns)
        )
        if isinstance(benchmark, pd.DataFrame):
            if benchmark.shape[1] == 1:
                benchmark = benchmark.iloc[:, 0]
            else:
                benchmark = _wide(benchmark, value_name="benchmark_return").mean(axis=1)
        benchmark = pd.to_numeric(benchmark, errors="coerce").reindex(r.index)
    return (
        s.apply(pd.to_numeric, errors="coerce"),
        r.apply(pd.to_numeric, errors="coerce"),
        benchmark,
    )


def rank_ic_by_date(
    scores: Any,
    forward_returns: Any,
    *,
    min_assets: int = 3,
    method: str = "spearman",
) -> pd.Series:
    """Compute one cross-sectional rank IC per decision date."""

    s, r, _ = _align(scores, forward_returns)
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    values: list[float] = []
    dates: list[Any] = []
    for date in s.index:
        pair = pd.concat(
            [s.loc[date].rename("score"), r.loc[date].rename("return")], axis=1
        ).dropna()
        if len(pair) < min_assets:
            values.append(np.nan)
        elif method == "spearman":
            values.append(
                float(
                    pair["score"].rank(method="average").corr(pair["return"].rank(method="average"))
                )
            )
        else:
            values.append(float(pair["score"].corr(pair["return"])))
        dates.append(date)
    return pd.Series(values, index=pd.Index(dates, name=s.index.name), name="rank_ic")


def rank_ic(scores: Any, forward_returns: Any, **kwargs: Any) -> pd.Series:
    return rank_ic_by_date(scores, forward_returns, **kwargs)


def _portfolio_row(
    score: pd.Series,
    outcome: pd.Series,
    top_fraction: float,
    bottom_fraction: float,
    min_assets: int,
) -> dict[str, Any]:
    pair = pd.concat([score.rename("score"), outcome.rename("outcome")], axis=1).dropna()
    n = len(pair)
    if n < min_assets:
        return {
            "top_mean": np.nan,
            "bottom_mean": np.nan,
            "universe_mean": np.nan,
            "spread": np.nan,
            "top_universe_spread": np.nan,
            "top_hit": np.nan,
            "bottom_hit": np.nan,
            "n": n,
            "top_n": 0,
            "bottom_n": 0,
        }
    top_n = max(1, int(np.ceil(n * top_fraction)))
    bottom_n = max(1, int(np.ceil(n * bottom_fraction)))
    # Stable sort makes ties deterministic and preserves input column order.
    ranked = pair.sort_values("score", ascending=False, kind="mergesort")
    top = ranked.iloc[:top_n]
    bottom = ranked.sort_values("score", ascending=True, kind="mergesort").iloc[:bottom_n]
    top_mean = float(top["outcome"].mean())
    bottom_mean = float(bottom["outcome"].mean())
    universe_mean = float(pair["outcome"].mean())
    return {
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "universe_mean": universe_mean,
        "spread": top_mean - bottom_mean,
        "top_universe_spread": top_mean - universe_mean,
        "top_hit": float((top["outcome"] > 0).mean()),
        "bottom_hit": float((bottom["outcome"] > 0).mean()),
        "n": n,
        "top_n": len(top),
        "bottom_n": len(bottom),
    }


def top_bottom_metrics(
    scores: Any,
    forward_returns: Any,
    *,
    benchmark_returns: Any = None,
    top_fraction: float = 0.20,
    bottom_fraction: float = 0.20,
    min_assets: int = 3,
) -> dict[str, Any]:
    """Measure top-minus-bottom outcome and positive-hit rates by date."""

    if not 0 < top_fraction <= 1 or not 0 < bottom_fraction <= 1:
        raise ValueError("top_fraction and bottom_fraction must be in (0, 1]")
    s, r, benchmark = _align(scores, forward_returns, benchmark_returns)
    if benchmark is not None:
        outcome = r.sub(benchmark, axis=0)
    else:
        outcome = r
    rows = []
    for date in s.index:
        row = _portfolio_row(
            s.loc[date], outcome.loc[date], top_fraction, bottom_fraction, min_assets
        )
        row["date"] = date
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    spread = frame["spread"].dropna() if not frame.empty else pd.Series(dtype=float)
    top_universe = (
        frame["top_universe_spread"].dropna() if not frame.empty else pd.Series(dtype=float)
    )
    top_hit = frame["top_hit"].dropna() if not frame.empty else pd.Series(dtype=float)
    bottom_hit = frame["bottom_hit"].dropna() if not frame.empty else pd.Series(dtype=float)
    result = {
        "top_bottom_spread": float(spread.mean()) if len(spread) else None,
        "top_bottom_spread_mean": float(spread.mean()) if len(spread) else None,
        "top_bottom_spread_median": float(spread.median()) if len(spread) else None,
        "top_minus_universe_excess": float(top_universe.mean()) if len(top_universe) else None,
        "top_minus_universe_excess_median": float(top_universe.median())
        if len(top_universe)
        else None,
        "top_mean_return": float(frame["top_mean"].dropna().mean())
        if not frame.empty and frame["top_mean"].notna().any()
        else None,
        "bottom_mean_return": float(frame["bottom_mean"].dropna().mean())
        if not frame.empty and frame["bottom_mean"].notna().any()
        else None,
        "top_hit_rate": float(top_hit.mean()) if len(top_hit) else None,
        "bottom_hit_rate": float(bottom_hit.mean()) if len(bottom_hit) else None,
        "observations": int(len(spread)),
        "per_date": frame,
    }
    # ``hit_rate`` is intentionally top-bucket hit rate: it answers whether a
    # high score led to a positive benchmark-relative outcome.
    result["hit_rate"] = result["top_hit_rate"]
    return result


@dataclass
class RankMetrics:
    rank_ic_mean: float | None
    rank_ic_median: float | None
    rank_ic_std: float | None
    rank_ic_count: int
    top_bottom_spread: float | None
    top_bottom_spread_median: float | None
    hit_rate: float | None
    top_hit_rate: float | None
    bottom_hit_rate: float | None
    per_date: pd.DataFrame
    rank_ic_ir: float | None = None
    top_minus_universe_excess: float | None = None
    top_minus_universe_excess_median: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank_ic": self.rank_ic_mean,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_median": self.rank_ic_median,
            "rank_ic_std": self.rank_ic_std,
            "rank_ic_ir": self.rank_ic_ir,
            "icir": self.rank_ic_ir,
            "top_minus_universe_excess": self.top_minus_universe_excess,
            "top_minus_universe_excess_median": self.top_minus_universe_excess_median,
            "rank_ic_count": self.rank_ic_count,
            "top_bottom_spread": self.top_bottom_spread,
            "top_bottom_spread_median": self.top_bottom_spread_median,
            "hit_rate": self.hit_rate,
            "top_hit_rate": self.top_hit_rate,
            "bottom_hit_rate": self.bottom_hit_rate,
            "per_date": self.per_date,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def evaluate_rank_metrics(
    scores: Any,
    forward_returns: Any,
    *,
    benchmark_returns: Any = None,
    min_assets: int = 3,
    top_fraction: float = 0.20,
    bottom_fraction: float = 0.20,
    method: str = "spearman",
) -> RankMetrics:
    """Return rank IC, top-bottom, and hit statistics in one report."""

    ic = rank_ic_by_date(scores, forward_returns, min_assets=min_assets, method=method)
    portfolio = top_bottom_metrics(
        scores,
        forward_returns,
        benchmark_returns=benchmark_returns,
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
        min_assets=min_assets,
    )
    valid_ic = ic.dropna()
    per_date = portfolio["per_date"].copy()
    per_date["rank_ic"] = ic.reindex(per_date.index)
    return RankMetrics(
        rank_ic_mean=float(valid_ic.mean()) if len(valid_ic) else None,
        rank_ic_median=float(valid_ic.median()) if len(valid_ic) else None,
        rank_ic_std=float(valid_ic.std(ddof=1))
        if len(valid_ic) > 1
        else (0.0 if len(valid_ic) == 1 else None),
        rank_ic_ir=(
            float(valid_ic.mean() / valid_ic.std(ddof=1))
            if len(valid_ic) > 1 and float(valid_ic.std(ddof=1)) > 0
            else None
        ),
        rank_ic_count=int(len(valid_ic)),
        top_bottom_spread=portfolio["top_bottom_spread"],
        top_bottom_spread_median=portfolio["top_bottom_spread_median"],
        hit_rate=portfolio["hit_rate"],
        top_hit_rate=portfolio["top_hit_rate"],
        bottom_hit_rate=portfolio["bottom_hit_rate"],
        per_date=per_date,
        top_minus_universe_excess=portfolio["top_minus_universe_excess"],
        top_minus_universe_excess_median=portfolio["top_minus_universe_excess_median"],
    )


cross_sectional_metrics = evaluate_rank_metrics
performance_metrics = evaluate_rank_metrics
compute_rank_ic = rank_ic_by_date
compute_top_bottom_metrics = top_bottom_metrics
compute_validation_metrics = evaluate_rank_metrics


__all__ = [
    "RankMetrics",
    "rank_ic",
    "rank_ic_by_date",
    "top_bottom_metrics",
    "evaluate_rank_metrics",
    "cross_sectional_metrics",
    "performance_metrics",
    "compute_rank_ic",
    "compute_top_bottom_metrics",
    "compute_validation_metrics",
]
