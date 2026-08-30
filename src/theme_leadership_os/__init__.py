"""Theme Leadership OS signal library.

The public signal functions live in :mod:`theme_leadership_os.signals`.
"""

from .pipeline import (
    RESEARCH_DISCLAIMER,
    PipelineIntegrityError,
    SignalRun,
    run_signal_pipeline,
)
from .signals import (
    CurrentLeader,
    CurrentLeaderConfig,
    EmergingRadar,
    EmergingRadarConfig,
    HoldCandidate12M,
    HoldCandidate12MConfig,
    current_leader,
    emerging_radar,
    hold_candidate_12m,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CurrentLeader",
    "CurrentLeaderConfig",
    "EmergingRadar",
    "EmergingRadarConfig",
    "HoldCandidate12M",
    "HoldCandidate12MConfig",
    "PipelineIntegrityError",
    "RESEARCH_DISCLAIMER",
    "SignalRun",
    "current_leader",
    "emerging_radar",
    "hold_candidate_12m",
    "run_signal_pipeline",
]
