"""Point-in-time membership selection and no-lookahead guards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import Temporal, ThemeMembership, parse_temporal, temporal_leq, temporal_lt


class LookaheadError(ValueError):
    """Raised when a caller explicitly asks to inspect unavailable evidence."""


class MembershipConflictError(ValueError):
    """Raised by strict validation when intervals overlap for one member."""


def membership_is_available_as_of(
    membership: ThemeMembership,
    as_of: Temporal | str,
) -> bool:
    """Return true only when validity and evidence timing both pass.

    This tiny predicate is intentionally the single source of truth used by
    the filtering helpers.  A valid interval by itself is insufficient: a
    membership announced later must not leak into an earlier backtest.
    """

    if not isinstance(membership, ThemeMembership):
        raise TypeError("membership must be a ThemeMembership")
    return membership.is_available_as_of(parse_temporal(as_of))


def assert_no_lookahead(
    records: Iterable[ThemeMembership],
    as_of: Temporal | str,
) -> None:
    """Raise if any record has evidence that was not available at ``as_of``.

    Filtering generally omits future records.  This explicit assertion is for
    callers that want a hard failure when a pipeline accidentally hands an
    unfiltered set to a downstream calculation.
    """

    point = parse_temporal(as_of)
    future = [
        record
        for record in records
        if not temporal_leq(record.evidence_available_at, point)
    ]
    if future:
        first = future[0]
        raise LookaheadError(
            "membership evidence is not available as of "
            f"{point.isoformat()}: {first.theme_id}/{first.permanent_id} "
            f"became available at {first.evidence_available_at.isoformat()}"
        )


def filter_memberships_as_of(
    memberships: Iterable[ThemeMembership],
    as_of: Temporal | str,
    *,
    theme_id: str | None = None,
    exposure_tier: str | None = None,
    strict: bool = False,
) -> list[ThemeMembership]:
    """Select memberships that were valid *and knowable* at a point in time.

    ``valid_to`` is exclusive.  Future-evidence rows are excluded by default,
    which makes this safe for ordinary backtests.  Set ``strict=True`` when a
    pipeline should fail loudly if its input contains any future evidence.
    """

    point = parse_temporal(as_of)
    rows = list(memberships)
    if any(not isinstance(row, ThemeMembership) for row in rows):
        raise TypeError("memberships must contain ThemeMembership values")
    if strict:
        assert_no_lookahead(rows, point)
    normalized_theme = theme_id.strip().lower() if theme_id is not None else None
    normalized_tier = exposure_tier.strip().lower().replace("-", "_") if exposure_tier else None
    selected = [
        row
        for row in rows
        if membership_is_available_as_of(row, point)
        and (normalized_theme is None or row.theme_id == normalized_theme)
        and (
            normalized_tier is None
            or (
                row.exposure_tier.value
                if hasattr(row.exposure_tier, "value")
                else str(row.exposure_tier)
            )
            == normalized_tier
        )
    ]
    # Stable ordering makes catalog snapshots and tests reproducible regardless
    # of the source file's row order.
    return sorted(
        selected,
        key=lambda row: (
            row.theme_id,
            row.exposure_tier.value
            if hasattr(row.exposure_tier, "value")
            else str(row.exposure_tier),
            row.permanent_id,
        ),
    )


def memberships_as_of(
    memberships: Iterable[ThemeMembership],
    as_of: Temporal | str,
    **kwargs: object,
) -> list[ThemeMembership]:
    """Alias with the natural noun-first spelling."""

    return filter_memberships_as_of(memberships, as_of, **kwargs)


def as_of_memberships(
    memberships: Iterable[ThemeMembership],
    as_of: Temporal | str,
    **kwargs: object,
) -> list[ThemeMembership]:
    """Backward-compatible alias used by a few consumers."""

    return filter_memberships_as_of(memberships, as_of, **kwargs)


def membership_as_of(
    membership: ThemeMembership,
    as_of: Temporal | str,
) -> bool:
    """Singular predicate alias for interactive use."""

    return membership_is_available_as_of(membership, as_of)


def validate_memberships(
    memberships: Sequence[ThemeMembership],
    *,
    reject_overlaps: bool = True,
) -> None:
    """Validate intervals and optionally reject overlapping versions.

    Overlap checks are kept separate from filtering because a catalog can
    intentionally contain superseded evidence rows.  A strict catalog load
    can opt into this check while exploratory fixtures can still be queried.
    """

    for row in memberships:
        if not isinstance(row, ThemeMembership):
            raise TypeError("memberships must contain ThemeMembership values")
    if not reject_overlaps:
        return
    groups: dict[tuple[str, str], list[ThemeMembership]] = {}
    for row in memberships:
        groups.setdefault((row.theme_id, row.permanent_id), []).append(row)
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: row.valid_from)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.valid_to is None or temporal_lt(current.valid_from, previous.valid_to):
                raise MembershipConflictError(
                    "overlapping membership intervals for "
                    f"{key[0]}/{key[1]}: {previous.valid_from!r} and {current.valid_from!r}"
                )


__all__ = [
    "LookaheadError",
    "MembershipConflictError",
    "as_of_memberships",
    "assert_no_lookahead",
    "filter_memberships_as_of",
    "membership_as_of",
    "membership_is_available_as_of",
    "memberships_as_of",
    "validate_memberships",
]
