from __future__ import annotations

import numpy as np
import pandas as pd

from theme_leadership_os.validation import PurgedWalkForwardSplit, purged_walk_forward_splits


def test_purged_walk_forward_drops_unmatured_labels():
    index = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    features = pd.DataFrame(index=index)
    label_end = pd.Series(index + np.timedelta64(21, "D"), index=index)
    splits = list(
        purged_walk_forward_splits(features, label_end=label_end, n_splits=3, test_size=5)
    )
    assert len(splits) == 3
    for train, test in splits:
        assert max(train) < min(test)
        # Every train label is mature before the test begins.
        assert label_end.iloc[train].max() < index[min(test)]


def test_splitter_supports_metadata_and_embargo_between_test_folds():
    index = pd.RangeIndex(40)
    features = pd.DataFrame(index=index)
    ends = pd.Series(index + 1, index=index)
    splitter = PurgedWalkForwardSplit(n_splits=3, test_size=5, embargo=2, label_end=ends)
    splits = list(splitter.split(features))
    assert len(splits) == 3
    for train, test in splits:
        assert train.dtype.kind in "iu"
        assert test.dtype.kind in "iu"
        assert train.max() < test.min()


def test_label_horizon_can_be_supplied_without_materialized_end_column():
    index = pd.RangeIndex(40)
    features = pd.DataFrame(index=index)
    splits = list(
        purged_walk_forward_splits(
            features,
            n_splits=2,
            test_size=5,
            label_horizon=3,
            embargo=0,
        )
    )
    assert len(splits) == 2
    for train, test in splits:
        assert max(train) + 3 < min(test)
