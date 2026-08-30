"""Bias and data-integrity gates for validation inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    rows: int = 0
    examples: tuple[Any, ...] = ()
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "rows": self.rows,
            "examples": list(self.examples),
            "severity": self.severity,
        }


@dataclass
class IntegrityAuditResult:
    passed: bool
    checked_rows: int
    issues: list[AuditIssue] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.passed

    @property
    def failed(self) -> bool:
        return not self.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "ok": self.ok,
            "checked_rows": self.checked_rows,
            "issues": [issue.as_dict() for issue in self.issues],
            "checks": dict(self.checks),
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise IntegrityError(self)

    def __bool__(self) -> bool:
        return self.passed


class IntegrityError(ValueError):
    """Raised by :func:`assert_integrity` when an audit gate fails."""

    def __init__(self, report: IntegrityAuditResult):
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues)
        super().__init__(f"validation integrity audit failed: {codes}")


def _frame(features: Any) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()
    if isinstance(features, pd.Series):
        return features.to_frame()
    if isinstance(features, Mapping):
        return pd.DataFrame(features)
    return pd.DataFrame(features)


def _timestamps(values: Any, index: pd.Index, *, name: str) -> pd.Series:
    if values is None:
        result = pd.Series(index, index=index, dtype="object")
    elif isinstance(values, pd.Series):
        result = values.reindex(index)
    elif isinstance(values, Mapping):
        result = pd.Series(values).reindex(index)
    elif np.isscalar(values):
        result = pd.Series(values, index=index)
    else:
        values_list = list(values)
        if len(values_list) != len(index):
            raise ValueError(f"{name} must have one value per feature row")
        result = pd.Series(values_list, index=index)
    return pd.to_datetime(result, errors="coerce", utc=True)


def _mapping_value(mapping: Any, key: Any, row: pd.Series, names: Iterable[str]) -> Any:
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        if key in mapping:
            value = mapping[key]
        else:
            value = None
        if isinstance(value, Mapping):
            for name in names:
                if name in value:
                    return value[name]
            return None
        if isinstance(value, (tuple, list)) and len(value) >= 1:
            # Caller-specific extraction happens at the call site; returning
            # the pair lets membership/inception helpers handle it.
            return value
        return value
    return None


def _row_meta(
    frame: pd.DataFrame,
    row: pd.Series,
    key: Any,
    mapping: Any,
    columns: tuple[str, ...],
) -> Any:
    for column in columns:
        if column in frame.columns:
            return row[column]
    value = _mapping_value(mapping, key, row, columns)
    return value


def _symbol(row: pd.Series, index_value: Any) -> Any:
    for column in ("symbol", "ticker", "asset", "instrument", "name"):
        if column in row.index and pd.notna(row[column]):
            return row[column]
    if isinstance(index_value, tuple) and index_value:
        return index_value[0]
    return index_value


def _pair_dates(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("membership_start", value.get("inception")))
        end = value.get("end", value.get("membership_end", value.get("delisted_at")))
        return start, end
    if isinstance(value, (tuple, list)):
        return (value[0] if value else None, value[1] if len(value) > 1 else None)
    return value, None


def _bad_examples(mask: pd.Series, index: pd.Index, limit: int = 5) -> tuple[Any, ...]:
    return tuple(index[mask.to_numpy()][:limit].tolist())


def audit_feature_availability(
    features: pd.DataFrame,
    *,
    decision_at: Any = None,
    feature_available_at: Any = None,
    require_column: bool = True,
) -> list[AuditIssue]:
    """Check that every feature is available no later than its decision."""

    frame = _frame(features)
    decision_values = decision_at
    if decision_values is None and "decision_at" in frame.columns:
        decision_values = frame["decision_at"]
    if decision_values is None:
        decision_values = frame.index
    available_values = feature_available_at
    if available_values is None and "feature_available_at" in frame.columns:
        available_values = frame["feature_available_at"]
    if available_values is None:
        if require_column:
            return [
                AuditIssue(
                    "missing_feature_available_at",
                    "feature availability timestamps are required",
                    len(frame),
                )
            ]
        return []
    decisions = _timestamps(decision_values, frame.index, name="decision_at")
    available = _timestamps(available_values, frame.index, name="feature_available_at")
    missing = available.isna() | decisions.isna()
    ordering_violation = available > decisions
    issues: list[AuditIssue] = []
    if missing.any():
        issues.append(
            AuditIssue(
                "invalid_feature_timestamp",
                "decision/availability timestamp is missing or invalid",
                int(missing.sum()),
                _bad_examples(missing, frame.index),
            )
        )
    if ordering_violation.any():
        issues.append(
            AuditIssue(
                "feature_after_decision",
                "feature_available_at must be <= decision_at",
                int(ordering_violation.sum()),
                _bad_examples(ordering_violation, frame.index),
            )
        )
    return issues


def audit_membership_dates(
    features: pd.DataFrame,
    *,
    membership_dates: Mapping[Any, Any] | None = None,
    decision_at: Any = None,
) -> list[AuditIssue]:
    """Ensure a member existed on each decision date.

    End dates are inclusive.  A missing end means the instrument remains in
    the universe.  Supplying no membership metadata skips this optional gate;
    once metadata is supplied, missing per-row dates are errors rather than
    silently assumed membership.
    """

    frame = _frame(features)
    has_columns = any(
        c in frame.columns for c in ("membership_start", "member_since", "inception_date")
    )
    if membership_dates is None and not has_columns:
        return []
    decisions = _timestamps(
        frame["decision_at"]
        if decision_at is None and "decision_at" in frame
        else decision_at
        if decision_at is not None
        else frame.index,
        frame.index,
        name="decision_at",
    )
    invalid = pd.Series(False, index=frame.index)
    missing = pd.Series(False, index=frame.index)
    for idx, row in frame.iterrows():
        key = _symbol(row, idx)
        value = None
        if membership_dates is not None:
            value = membership_dates.get(key)
        if value is None:
            starts = row.get("membership_start", row.get("member_since", row.get("inception_date")))
            ends = row.get("membership_end", row.get("member_until", row.get("delisted_at")))
        else:
            starts, ends = _pair_dates(value)
        if starts is None or pd.isna(starts):
            missing.loc[idx] = True
            continue
        decision = decisions.loc[idx]
        start = pd.to_datetime(starts, errors="coerce", utc=True)
        end = (
            pd.to_datetime(ends, errors="coerce", utc=True)
            if ends is not None and not pd.isna(ends)
            else pd.NaT
        )
        invalid.loc[idx] = bool(
            pd.isna(decision)
            or pd.isna(start)
            or decision < start
            or (pd.notna(end) and decision > end)
        )
    issues: list[AuditIssue] = []
    if missing.any():
        issues.append(
            AuditIssue(
                "missing_membership_date",
                "membership start is required when membership metadata is supplied",
                int(missing.sum()),
                _bad_examples(missing, frame.index),
            )
        )
    if invalid.any():
        issues.append(
            AuditIssue(
                "membership_out_of_range",
                "decision_at falls outside instrument membership dates",
                int(invalid.sum()),
                _bad_examples(invalid, frame.index),
            )
        )
    return issues


def audit_etf_inception(
    features: pd.DataFrame,
    *,
    etf_inception: Mapping[Any, Any] | None = None,
    decision_at: Any = None,
) -> list[AuditIssue]:
    """Reject ETF observations dated before the ETF's inception."""

    frame = _frame(features)
    has_column = any(
        c in frame.columns for c in ("etf_inception", "inception_date", "fund_inception")
    )
    if etf_inception is None and not has_column:
        return []
    decisions = _timestamps(
        frame["decision_at"]
        if decision_at is None and "decision_at" in frame
        else decision_at
        if decision_at is not None
        else frame.index,
        frame.index,
        name="decision_at",
    )
    invalid = pd.Series(False, index=frame.index)
    missing = pd.Series(False, index=frame.index)
    for idx, row in frame.iterrows():
        key = _symbol(row, idx)
        value = etf_inception.get(key) if etf_inception is not None else None
        if value is None:
            value = row.get("etf_inception", row.get("inception_date", row.get("fund_inception")))
        if value is None or pd.isna(value):
            missing.loc[idx] = True
            continue
        inception = pd.to_datetime(value, errors="coerce", utc=True)
        invalid.loc[idx] = bool(
            pd.isna(inception) or pd.isna(decisions.loc[idx]) or decisions.loc[idx] < inception
        )
    issues: list[AuditIssue] = []
    if missing.any():
        issues.append(
            AuditIssue(
                "missing_etf_inception",
                "ETF inception date is required when inception metadata is supplied",
                int(missing.sum()),
                _bad_examples(missing, frame.index),
            )
        )
    if invalid.any():
        issues.append(
            AuditIssue(
                "before_etf_inception",
                "decision_at precedes ETF inception",
                int(invalid.sum()),
                _bad_examples(invalid, frame.index),
            )
        )
    return issues


def audit_etf_history(
    features: pd.DataFrame,
    *,
    etf_inception: Mapping[Any, Any] | None = None,
    decision_at: Any = None,
    min_history_weeks: int = 26,
    history_column: str = "history_weeks",
) -> list[AuditIssue]:
    """Require at least 26 weeks of ETF history after inception.

    A supplied ``history_weeks`` column is preferred.  Otherwise the gate
    derives elapsed calendar weeks from inception metadata.  Non-ETF rows are
    ignored when an asset-type column is present.
    """

    frame = _frame(features)
    if history_column in frame.columns:
        history = pd.to_numeric(frame[history_column], errors="coerce")
        asset_mask = pd.Series(True, index=frame.index)
        for column in ("asset_type", "instrument_type", "security_type"):
            if column in frame.columns:
                asset_mask = frame[column].astype("string").str.lower().eq("etf")
                break
        invalid = asset_mask & (history.isna() | (history < min_history_weeks))
        if invalid.any():
            return [
                AuditIssue(
                    "insufficient_etf_history",
                    f"ETF history must be >= {min_history_weeks} weeks",
                    int(invalid.sum()),
                    _bad_examples(invalid, frame.index),
                )
            ]
        return []
    if etf_inception is None and not any(
        c in frame.columns for c in ("etf_inception", "inception_date", "fund_inception")
    ):
        return []
    decisions = _timestamps(
        frame["decision_at"]
        if decision_at is None and "decision_at" in frame
        else decision_at
        if decision_at is not None
        else frame.index,
        frame.index,
        name="decision_at",
    )
    invalid = pd.Series(False, index=frame.index)
    for idx, row in frame.iterrows():
        key = _symbol(row, idx)
        value = etf_inception.get(key) if etf_inception is not None else None
        if value is None:
            value = row.get("etf_inception", row.get("inception_date", row.get("fund_inception")))
        inception = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(inception) or pd.isna(decisions.loc[idx]):
            continue
        history_delta = np.timedelta64(7 * min_history_weeks, "D")
        invalid.loc[idx] = decisions.loc[idx] < inception + history_delta
    if not invalid.any():
        return []
    return [
        AuditIssue(
            "insufficient_etf_history",
            f"ETF history must be >= {min_history_weeks} weeks after inception",
            int(invalid.sum()),
            _bad_examples(invalid, frame.index),
        )
    ]


def audit_coverage(
    features: pd.DataFrame,
    *,
    min_coverage: float = 0.95,
    coverage_column: str = "coverage",
    coverage_count_column: str = "coverage_count",
    expected_count_column: str = "expected_count",
    require_column: bool = True,
) -> list[AuditIssue]:
    """Apply a minimum non-missing-universe coverage gate."""

    frame = _frame(features)
    if coverage_column in frame.columns:
        coverage = pd.to_numeric(frame[coverage_column], errors="coerce")
    elif coverage_count_column in frame.columns and expected_count_column in frame.columns:
        expected = pd.to_numeric(frame[expected_count_column], errors="coerce")
        count = pd.to_numeric(frame[coverage_count_column], errors="coerce")
        coverage = count.div(expected.replace(0, np.nan))
    else:
        if require_column:
            return [
                AuditIssue(
                    "missing_coverage",
                    "coverage or coverage_count/expected_count is required",
                    len(frame),
                )
            ]
        return []
    # Coverage is a decimal ratio by convention.  Do not guess whether 95
    # means 95% or a 95x ratio; callers must pass 0.95 explicitly.
    threshold = float(min_coverage)
    if not 0 <= threshold <= 1:
        raise ValueError("min_coverage must be a decimal fraction in [0, 1]")
    invalid = coverage.isna() | (coverage < threshold)
    if not invalid.any():
        return []
    return [
        AuditIssue(
            "insufficient_coverage",
            f"coverage must be >= {threshold:.3f}",
            int(invalid.sum()),
            _bad_examples(invalid, frame.index),
        )
    ]


def audit_duplicates(
    features: pd.DataFrame, *, key_columns: Iterable[str] = ("decision_at", "symbol")
) -> list[AuditIssue]:
    frame = _frame(features)
    columns = [column for column in key_columns if column in frame.columns]
    if not columns:
        return []
    duplicate = frame.duplicated(columns, keep=False)
    if not duplicate.any():
        return []
    return [
        AuditIssue(
            "duplicate_feature_key",
            f"duplicate rows for key {columns}",
            int(duplicate.sum()),
            _bad_examples(duplicate, frame.index),
        )
    ]


def audit_integrity(
    features: pd.DataFrame,
    *,
    decision_at: Any = None,
    feature_available_at: Any = None,
    membership_dates: Mapping[Any, Any] | None = None,
    etf_inception: Mapping[Any, Any] | None = None,
    min_coverage: float = 0.95,
    require_feature_available_at: bool = True,
    key_columns: Iterable[str] = ("decision_at", "symbol"),
) -> IntegrityAuditResult:
    """Run all integrity gates and return a machine-readable audit report."""

    frame = _frame(features)
    issues: list[AuditIssue] = []
    issues.extend(
        audit_feature_availability(
            frame,
            decision_at=decision_at,
            feature_available_at=feature_available_at,
            require_column=require_feature_available_at,
        )
    )
    issues.extend(
        audit_membership_dates(frame, membership_dates=membership_dates, decision_at=decision_at)
    )
    issues.extend(audit_etf_inception(frame, etf_inception=etf_inception, decision_at=decision_at))
    issues.extend(audit_etf_history(frame, etf_inception=etf_inception, decision_at=decision_at))
    issues.extend(audit_coverage(frame, min_coverage=min_coverage))
    issues.extend(audit_duplicates(frame, key_columns=key_columns))
    checks = {
        "feature_availability": not any(
            i.code
            in {
                "missing_feature_available_at",
                "invalid_feature_timestamp",
                "feature_after_decision",
            }
            for i in issues
        ),
        "membership_dates": not any(
            i.code in {"missing_membership_date", "membership_out_of_range"} for i in issues
        ),
        "etf_inception": not any(
            i.code in {"missing_etf_inception", "before_etf_inception", "insufficient_etf_history"}
            for i in issues
        ),
        "coverage": not any(
            i.code in {"missing_coverage", "insufficient_coverage"} for i in issues
        ),
        "duplicates": not any(i.code == "duplicate_feature_key" for i in issues),
    }
    return IntegrityAuditResult(
        passed=not issues, checked_rows=len(frame), issues=issues, checks=checks
    )


def assert_integrity(*args: Any, **kwargs: Any) -> IntegrityAuditResult:
    report = audit_integrity(*args, **kwargs)
    report.raise_if_failed()
    return report


run_integrity_audit = audit_integrity
check_integrity = audit_integrity
audit_backtest_inputs = audit_integrity
validate_features = audit_integrity


__all__ = [
    "AuditIssue",
    "IntegrityAuditResult",
    "IntegrityError",
    "audit_feature_availability",
    "audit_membership_dates",
    "audit_etf_inception",
    "audit_etf_history",
    "audit_coverage",
    "audit_duplicates",
    "audit_integrity",
    "run_integrity_audit",
    "check_integrity",
    "audit_backtest_inputs",
    "validate_features",
    "assert_integrity",
]
