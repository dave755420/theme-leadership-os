"""Transparent, weekly theme leadership signals.

All public functions accept a tidy :class:`pandas.DataFrame` with at least
``date``, ``theme``, ``security`` and ``value`` columns.  ``value`` is
treated as an end-of-session price (or index level).  A row with
``security == "SPY"`` supplies the market baseline unless a different
security is configured.
"""

from .core import (
    CurrentLeader,
    CurrentLeaderConfig,
    EmergingRadar,
    EmergingRadarConfig,
    HoldCandidate12M,
    HoldCandidate12MConfig,
    SignalConfig,
    cross_sectional_percentile_rank,
    current_leader,
    emerging_radar,
    hold_candidate_12m,
    percentile_rank,
)

__all__ = [
    "SignalConfig",
    "CurrentLeaderConfig",
    "EmergingRadarConfig",
    "HoldCandidate12MConfig",
    "current_leader",
    "emerging_radar",
    "hold_candidate_12m",
    "percentile_rank",
    "cross_sectional_percentile_rank",
    "CurrentLeader",
    "EmergingRadar",
    "HoldCandidate12M",
]
