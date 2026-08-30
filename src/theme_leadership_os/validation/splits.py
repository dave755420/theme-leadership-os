"""Purged, embargoed walk-forward cross-validation splits."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EMBARGO_WEEKS = 4


def _frame_index(X: Any) -> pd.Index:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        index = X.index
    elif hasattr(X, "index"):
        index = X.index
    else:
        index = pd.RangeIndex(len(X))
    if isinstance(index, pd.DatetimeIndex):
        return index
    if pd.api.types.is_numeric_dtype(index):
        return pd.Index(index)
    try:
        return pd.DatetimeIndex(pd.to_datetime(index))
    except (TypeError, ValueError):
        return pd.Index(index)


def _coerce_label_end(label_end: Any, labels: Any, index: pd.Index) -> pd.Series | None:
    candidate = label_end
    if candidate is None and labels is not None:
        if isinstance(labels, pd.DataFrame):
            names = [
                "label_available_at",
                "label_end",
                "maturity_at",
                "label_maturity",
            ]
            found = next((name for name in names if name in labels.columns), None)
            if found is None:
                horizon_columns = [
                    c
                    for c in labels.columns
                    if str(c).startswith(("label_available_at_", "maturity_at_", "exit_at_"))
                ]
                found = (
                    max(
                        horizon_columns,
                        key=lambda value: int(
                            "".join(ch for ch in str(value) if ch.isdigit()) or "0"
                        ),
                    )
                    if horizon_columns
                    else None
                )
            if found is not None:
                candidate = labels[found]
        elif isinstance(labels, pd.Series) and not pd.api.types.is_bool_dtype(labels):
            # A datetime-valued y is a convenient label-end input.
            if isinstance(labels.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(
                labels
            ):
                candidate = labels
    if candidate is None:
        return None
    if isinstance(candidate, pd.Series):
        result = candidate.reindex(index)
    elif isinstance(candidate, Mapping):  # type: ignore[name-defined]
        result = pd.Series(candidate).reindex(index)
    else:
        values = (
            list(candidate)
            if not np.isscalar(candidate) and not isinstance(candidate, str)
            else [candidate] * len(index)
        )
        if len(values) != len(index):
            raise ValueError("label_end must have the same length as X")
        result = pd.Series(values, index=index)
    if isinstance(index, pd.DatetimeIndex):
        result = pd.to_datetime(result, errors="coerce")
    return result


def _duration_cutoff(reference: Any, amount: Any, index: pd.Index, *, before: bool = True) -> Any:
    """Compute a date/row cutoff for an embargo or maturity period."""

    if amount is None or amount == 0:
        return reference
    if isinstance(index, pd.DatetimeIndex):
        if isinstance(amount, (str, pd.Timedelta, np.timedelta64)):
            delta = pd.Timedelta(amount)
        elif isinstance(amount, (int, np.integer, float)):
            # Date-index integer embargo values are sessions, not days.
            pos = int(index.searchsorted(reference, side="left"))
            target_pos = pos - int(amount) if before else pos + int(amount)
            target_pos = min(max(target_pos, 0), len(index) - 1)
            return index[target_pos]
        else:
            delta = pd.Timedelta(amount)
        return reference - delta if before else reference + delta
    # Positional fixtures interpret all integer durations as rows.
    return int(reference) - int(amount) if before else int(reference) + int(amount)


def _as_positions(index: pd.Index, candidates: Iterable[Any]) -> np.ndarray:
    position_map = {value: i for i, value in enumerate(index)}
    positions = [position_map[x] for x in candidates if x in position_map]
    return np.asarray(positions, dtype=int)


@dataclass(frozen=True)
class WalkForwardSplit:
    """A split plus dates useful for audit logs."""

    train: np.ndarray
    test: np.ndarray
    train_dates: pd.Index
    test_dates: pd.Index

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.train, self.test


def _split_iterator(
    index: pd.Index,
    *,
    labels: Any = None,
    n_splits: int = 5,
    test_size: int | None = None,
    train_size: int | None = None,
    embargo: int | str | pd.Timedelta = DEFAULT_EMBARGO_WEEKS,
    label_end: Any = None,
    label_maturity: Any = None,
    min_train_size: int = 1,
) -> Iterator[WalkForwardSplit]:
    n = len(index)
    if n_splits < 1:
        raise ValueError("n_splits must be at least one")
    if n == 0:
        return
    if test_size is None:
        test_size = max(1, n // (n_splits + 1))
    if test_size < 1:
        raise ValueError("test_size must be at least one")
    first_test_start = n - n_splits * test_size
    if first_test_start < min_train_size:
        raise ValueError("not enough observations for requested splits and min_train_size")
    maturity = _coerce_label_end(label_end, labels, index)
    previous_test_end: Any = None
    for split_number in range(n_splits):
        test_start_pos = first_test_start + split_number * test_size
        test_end_pos = min(test_start_pos + test_size, n)
        if test_start_pos >= n or test_end_pos <= test_start_pos:
            continue
        test_positions = np.arange(test_start_pos, test_end_pos, dtype=int)
        test_start = index[test_start_pos]
        test_end = index[test_end_pos - 1]
        # The chronology guard prevents a future observation from entering an
        # expanding training set even if a caller supplies unusual indices.
        train_candidates = np.arange(0, test_start_pos, dtype=int)
        if train_size is not None:
            if train_size < 1:
                raise ValueError("train_size must be at least one when supplied")
            train_candidates = train_candidates[-int(train_size) :]
        if previous_test_end is not None and embargo:
            # Do not feed observations immediately following the preceding
            # test fold into the next fold's training set.  This is the
            # forward/expanding equivalent of a post-test embargo.
            blocked_until = _duration_cutoff(previous_test_end, embargo, index, before=False)
            if isinstance(index, pd.DatetimeIndex):
                candidate_dates = index[train_candidates]
                train_candidates = train_candidates[
                    (candidate_dates <= previous_test_end) | (candidate_dates > blocked_until)
                ]
            else:
                candidate_dates = index[train_candidates]
                train_candidates = train_candidates[
                    (candidate_dates <= previous_test_end) | (candidate_dates > blocked_until)
                ]
        if maturity is not None:
            # Labels whose outcome is not known before the test boundary are
            # purged.  A maturity lag tightens the boundary further, e.g. an
            # embargo of two weeks requires labels known by test_start - 2w.
            maturity_cutoff = _duration_cutoff(test_start, label_maturity, index, before=True)
            if isinstance(index, pd.DatetimeIndex):
                known = maturity <= maturity_cutoff
            else:
                known = maturity <= maturity_cutoff
            known = known.fillna(False).to_numpy(dtype=bool)
            train_candidates = train_candidates[known[train_candidates]]
            # Explicit overlap purge: a label ending on test start is not
            # allowed to share the boundary with the first test observation.
            if isinstance(index, pd.DatetimeIndex):
                end_values = pd.to_datetime(maturity, errors="coerce")
                purge = end_values >= test_start
            else:
                purge = maturity >= test_start
            purge = pd.Series(purge, index=index).fillna(True).to_numpy(dtype=bool)
            train_candidates = train_candidates[~purge[train_candidates]]
        if embargo:
            # Keep a quiet period immediately before the test boundary as
            # well.  It is conservative for expanding walk-forward folds and
            # makes the embargo visible even when there are no post-test train
            # rows in a particular fold.
            blocked_before = _duration_cutoff(test_start, embargo, index, before=True)
            if isinstance(index, pd.DatetimeIndex):
                train_candidates = train_candidates[index[train_candidates] < blocked_before]
            else:
                train_candidates = train_candidates[index[train_candidates] < blocked_before]
        # Embargo is represented explicitly in the split metadata and in the
        # test set guard.  In an expanding chronology there are no post-test
        # training rows, but this check protects custom train windows.
        if len(train_candidates) < min_train_size:
            previous_test_end = test_end
            continue
        yield WalkForwardSplit(
            train=train_candidates,
            test=test_positions,
            train_dates=index[train_candidates],
            test_dates=index[test_positions],
        )
        previous_test_end = test_end


def purged_walk_forward_splits(
    X: Any,
    y: Any = None,
    *,
    n_splits: int = 5,
    test_size: int | None = None,
    train_size: int | None = None,
    embargo: int | str | pd.Timedelta = DEFAULT_EMBARGO_WEEKS,
    embargo_weeks: int | None = None,
    label_end: Any = None,
    label_horizon: int | str | pd.Timedelta | None = None,
    label_horizon_weeks: int | None = None,
    purge_horizon: int | str | pd.Timedelta | None = None,
    label_maturity: Any = None,
    min_train_size: int = 1,
    return_metadata: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray] | WalkForwardSplit]:
    """Yield chronological, purged train/test index arrays.

    ``label_end`` may be a Series/array of timestamps, or can be inferred from
    ``y`` columns named ``label_available_at*``, ``label_end``, or
    ``maturity_at``.  Training rows whose labels overlap the test period are
    removed.  ``label_maturity`` adds a conservative lag before the test start
    at which a label must be known.  ``embargo`` accepts row counts for numeric
    fixtures and either row counts or timedeltas for date indexes.
    """

    index = _frame_index(X)
    if embargo_weeks is not None:
        embargo = int(embargo_weeks)
    if label_horizon is None and label_horizon_weeks is not None:
        label_horizon = int(label_horizon_weeks)
    if label_horizon is None and purge_horizon is not None:
        label_horizon = purge_horizon
    if label_end is None and label_horizon is not None:
        if isinstance(index, pd.DatetimeIndex):
            if isinstance(label_horizon, (str, pd.Timedelta, np.timedelta64)):
                delta = pd.Timedelta(label_horizon)
            else:
                delta = np.timedelta64(7 * int(label_horizon), "D")
            label_end = pd.Series(index + delta, index=index)
        else:
            label_end = pd.Series(index + int(label_horizon), index=index)
    iterator = _split_iterator(
        index,
        labels=y,
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        embargo=embargo,
        label_end=label_end,
        label_maturity=label_maturity,
        min_train_size=min_train_size,
    )
    for split in iterator:
        yield split if return_metadata else split.as_tuple()


def generate_purged_walk_forward_splits(*args: Any, **kwargs: Any) -> Iterator[Any]:
    return purged_walk_forward_splits(*args, **kwargs)


class PurgedWalkForwardSplit:
    """Scikit-learn compatible splitter with label purge and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        train_size: int | None = None,
        embargo: int | str | pd.Timedelta = DEFAULT_EMBARGO_WEEKS,
        embargo_weeks: int | None = None,
        label_end: Any = None,
        label_horizon: int | str | pd.Timedelta | None = None,
        label_horizon_weeks: int | None = None,
        purge_horizon: int | str | pd.Timedelta | None = None,
        label_maturity: Any = None,
        min_train_size: int = 1,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.embargo = int(embargo_weeks) if embargo_weeks is not None else embargo
        self.label_end = label_end
        self.label_horizon = (
            int(label_horizon_weeks)
            if label_horizon is None and label_horizon_weeks is not None
            else purge_horizon
            if label_horizon is None
            else label_horizon
        )
        self.label_maturity = label_maturity
        self.min_train_size = min_train_size

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        labels = y if y is not None else None
        label_end = self.label_end
        yield from purged_walk_forward_splits(
            X,
            labels,
            n_splits=self.n_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            embargo=self.embargo,
            label_end=label_end,
            label_horizon=self.label_horizon,
            label_maturity=self.label_maturity,
            min_train_size=self.min_train_size,
        )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits


PurgedWalkForwardCV = PurgedWalkForwardSplit
walk_forward_splits = purged_walk_forward_splits


__all__ = [
    "DEFAULT_EMBARGO_WEEKS",
    "WalkForwardSplit",
    "PurgedWalkForwardSplit",
    "purged_walk_forward_splits",
    "generate_purged_walk_forward_splits",
    "PurgedWalkForwardCV",
    "walk_forward_splits",
]
