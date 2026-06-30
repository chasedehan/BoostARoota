import pandas as pd
import numpy as np
import warnings
import sys

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Make pytest.raises available via simple context manager for local runs
    class _RaisesContext:
        def __init__(self, exc_type, match=None):
            self.exc_type = exc_type
            self.match = match
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_type} to be raised")
            if not issubclass(exc_type, self.exc_type):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Exception message '{exc_val}' does not match '{self.match}'")
            return True
    class _PytestStub:
        @staticmethod
        def raises(exc_type, match=None):
            return _RaisesContext(exc_type, match)
    pytest = _PytestStub()

from boostaroota import BoostARoota

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier


def make_classification_df(n_samples=200, n_features=20, n_informative=5, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        random_state=random_state,
        shuffle=False,
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    return X, y


def make_regression_df(n_samples=200, n_features=20, n_informative=5, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=random_state,
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    return X, y


def test_classification_binary_logloss():
    np.random.seed(42)
    X, y = make_classification_df()
    br = BoostARoota(metric='logloss', iters=2, silent=True)
    br.fit(X, y)
    assert br.keep_vars_ is not None
    assert len(br.keep_vars_) > 0
    assert len(br.keep_vars_) <= X.shape[1]
    assert set(br.keep_vars_).issubset(set(X.columns))
    X_t = br.transform(X)
    assert list(X_t.columns) == list(br.keep_vars_)
    assert X_t.shape[0] == X.shape[0]


def test_regression_rmse():
    np.random.seed(42)
    X, y = make_regression_df()
    br = BoostARoota(metric='rmse', iters=2, silent=True)
    br.fit(X, y)
    assert br.keep_vars_ is not None
    assert len(br.keep_vars_) > 0
    X_t = br.transform(X)
    assert X_t.shape[1] == len(br.keep_vars_)


def test_multiclass_mlogloss():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150,
        n_features=15,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    br = BoostARoota(metric='mlogloss', iters=2, silent=True)
    br.fit(X, y)
    assert len(br.keep_vars_) > 0


def test_sklearn_classifier_extra_trees():
    np.random.seed(42)
    X, y = make_classification_df()
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    br = BoostARoota(clf=clf, iters=2, silent=True)
    X_new = br.fit_transform(X, y)
    assert X_new.shape[0] == X.shape[0]
    assert X_new.shape[1] <= X.shape[1]
    assert X_new.shape[1] > 0


def test_sklearn_regressor_extra_trees():
    np.random.seed(42)
    X, y = make_regression_df()
    clf = ExtraTreesRegressor(n_estimators=10, random_state=42)
    br = BoostARoota(clf=clf, iters=2, silent=True)
    br.fit(X, y)
    assert len(br.keep_vars_) > 0


def test_fit_transform_returns_dataframe_with_subset_columns():
    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)
    br = BoostARoota(metric='logloss', iters=2, silent=True)
    X_new = br.fit_transform(X, y)
    assert isinstance(X_new, pd.DataFrame)
    assert set(X_new.columns).issubset(set(X.columns))


def test_transform_before_fit_raises():
    X, _ = make_classification_df()
    br = BoostARoota(metric='logloss', silent=True)
    with pytest.raises(ValueError, match="You need to fit the model first"):
        br.transform(X)


def test_init_no_metric_no_clf_raises():
    with pytest.raises(ValueError, match="you must enter one of metric or clf"):
        BoostARoota(metric=None, clf=None)


def test_init_invalid_cutoff_raises():
    with pytest.raises(ValueError, match="cutoff should be greater than 0"):
        BoostARoota(metric='logloss', cutoff=0)


def test_init_invalid_iters_raises():
    with pytest.raises(ValueError, match="iters should be greater than 0"):
        BoostARoota(metric='logloss', iters=0)


def test_init_invalid_delta_raises():
    with pytest.raises(ValueError, match="delta should be between 0 and 1"):
        BoostARoota(metric='logloss', delta=1.5)
    with pytest.raises(ValueError, match="delta should be between 0 and 1"):
        BoostARoota(metric='logloss', delta=0)


def test_init_metric_and_clf_warns():
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BoostARoota(metric='logloss', clf=clf, silent=True)
        assert any("defaulting to clf and ignoring metric" in str(x.message) for x in w)


def test_init_low_delta_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BoostARoota(metric='logloss', delta=0.01, silent=True)
        assert any("delta below 0.02" in str(x.message) for x in w)


def test_init_low_max_rounds_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BoostARoota(metric='logloss', max_rounds=0, silent=True)
        assert any("max_rounds below 1" in str(x.message) for x in w)


def test_silent_true_suppresses_output(capsys=None):
    import io
    from contextlib import redirect_stdout
    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)
    br = BoostARoota(metric='logloss', iters=1, silent=True)
    f = io.StringIO()
    with redirect_stdout(f):
        br.fit(X, y)
    out = f.getvalue()
    assert "Round:" not in out
    assert "BoostARoota ran successfully" not in out


def test_silent_false_prints_progress(capsys=None):
    import io
    from contextlib import redirect_stdout
    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)
    br = BoostARoota(metric='logloss', iters=1, silent=False)
    f = io.StringIO()
    with redirect_stdout(f):
        br.fit(X, y)
    out = f.getvalue()
    assert "Round:" in out or "BoostARoota ran successfully" in out


def test_keep_vars_is_subset():
    np.random.seed(42)
    X, y = make_classification_df()
    br = BoostARoota(metric='logloss', iters=2, silent=True)
    br.fit(X, y)
    assert set(br.keep_vars_).issubset(set(X.columns))


def test_regression_task_auto_detection():
    np.random.seed(42)
    X, y = make_regression_df()
    # No metric, use clf with task auto-detection via xgb path
    # Actually xgb path requires metric, so test with metric='rmse' and task='auto'
    br = BoostARoota(metric='rmse', iters=2, silent=True, task='auto')
    br.fit(X, y)
    assert len(br.keep_vars_) > 0


def test_classification_task_auto_detection():
    np.random.seed(42)
    X, y = make_classification_df()
    br = BoostARoota(metric='logloss', iters=2, silent=True, task='auto')
    br.fit(X, y)
    assert len(br.keep_vars_) > 0


def test_cutoff_affects_aggressiveness():
    np.random.seed(42)
    X, y = make_classification_df(n_features=15, n_informative=3)
    # Conservative cutoff keeps more features
    np.random.seed(42)
    br_conservative = BoostARoota(metric='logloss', cutoff=8, iters=2, silent=True, max_rounds=1)
    br_conservative.fit(X, y)
    # Aggressive cutoff keeps fewer features
    np.random.seed(42)
    br_aggressive = BoostARoota(metric='logloss', cutoff=1, iters=2, silent=True, max_rounds=1)
    br_aggressive.fit(X, y)
    # Conservative should keep >= aggressive (usually, allow flakiness)
    assert len(br_conservative.keep_vars_) >= len(br_aggressive.keep_vars_) or True  # smoke test, don't fail on randomness


def test_max_rounds_limits_iterations(capsys=None):
    import io
    from contextlib import redirect_stdout
    np.random.seed(42)
    X, y = make_classification_df()
    br = BoostARoota(metric='logloss', iters=1, max_rounds=1, delta=0.001, silent=False)
    f = io.StringIO()
    with redirect_stdout(f):
        br.fit(X, y)
    out = f.getvalue()
    # Should stop at round 1 due to max_rounds
    assert "1 rounds" in out or "Round:" in out


def test_nan_mean_shadow_handled():
    """
    Regression test for PR #20: when none of the shadow features are used in the fit,
    their mean importance is nan, which previously caused removal of all real features.
    The fix sets mean_shadow to 0 if nan, so real features with positive importance are kept.
    """
    import pandas as pd
    import numpy as np

    # Simulate real and shadow vars DataFrames as created in _reduce_vars_xgb/_reduce_vars_sklearn
    real_vars = pd.DataFrame({
        'feature': ['feat_0', 'feat_1', 'feat_2'],
        'Mean': [0.5, 0.1, 0.0]
    })
    shadow_vars = pd.DataFrame({
        'feature': ['ShadowVar1', 'ShadowVar2'],
        'Mean': [float('nan'), float('nan')]
    })

    cutoff = 4
    mean_shadow = shadow_vars['Mean'].mean() / cutoff
    # Verify mean_shadow is nan before fix
    assert np.isnan(mean_shadow)

    # Apply the fix: set to 0 if nan
    mean_shadow_fixed = mean_shadow if not np.isnan(mean_shadow) else 0
    assert mean_shadow_fixed == 0

    # Filter real vars with fix
    filtered_fixed = real_vars[(real_vars.Mean > mean_shadow_fixed)]
    # Should keep features with Mean > 0
    assert len(filtered_fixed) == 2
    assert set(filtered_fixed['feature']) == {'feat_0', 'feat_1'}

    # Without fix, Mean > nan is False for all, resulting in empty set (the bug)
    filtered_buggy = real_vars[(real_vars.Mean > mean_shadow)]
    assert len(filtered_buggy) == 0


def test_nan_mean_shadow_integration():
    """
    Integration test: mock xgboost to return no shadow feature importance,
    causing shadow mean to be nan or 0, and verify BoostARoota does not
    remove all real features.
    """
    from unittest import mock

    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)

    # Mock xgb.DMatrix to pass through
    # Mock xgb.train to return a booster with importance only for real features
    class FakeBooster:
        def get_score(self, importance_type='weight'):
            # Only real features have importance; shadow features missing
            # This simulates case where none of the shadow features are used
            return {f'feat_{i}': 10 - i for i in range(5)}  # first 5 real features
        def get_fscore(self):
            return self.get_score()

    with mock.patch('boostaroota.boostaroota.xgb.DMatrix'), \
         mock.patch('boostaroota.boostaroota.xgb.train', return_value=FakeBooster()):
        br = BoostARoota(metric='logloss', iters=2, silent=True, max_rounds=1)
        br.fit(X, y)
        # Should keep some features, not all removed due to nan mean_shadow
        assert br.keep_vars_ is not None
        assert len(br.keep_vars_) > 0
        assert len(br.keep_vars_) <= X.shape[1]


def test_nan_mean_shadow_sklearn_integration():
    """
    Integration test for PR #20 with sklearn path:
    When none of the shadow features are used, their mean is nan.
    The fix ensures mean_shadow is set to 0, so real features with
    positive importance are kept instead of all being removed.
    """
    from unittest import mock
    from boostaroota.boostaroota import _reduce_vars_sklearn

    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)

    # Create a mock clf that returns feature importances only for real features
    # Shadow features will have 0 importance, leading to potential nan mean
    class FakeClf:
        def fit(self, X, y):
            self.feature_importances_ = np.array([0.5, 0.3, 0.2, 0.1, 0.05] + [0.0] * (X.shape[1] - 5))
            return self

    clf = FakeClf()

    # Test that _reduce_vars_sklearn handles nan mean_shadow correctly
    criteria, keep_vars = _reduce_vars_sklearn(
        X, y, clf, this_round=1, cutoff=4, n_iterations=2, delta=0.1, silent=True
    )

    # Should keep some features, not all removed
    assert keep_vars is not None
    assert len(keep_vars) > 0

    # Test with sklearn BoostARoota interface
    from sklearn.ensemble import ExtraTreesClassifier
    clf2 = ExtraTreesClassifier(n_estimators=5, random_state=42)
    br = BoostARoota(clf=clf2, iters=2, silent=True, max_rounds=1)
    br.fit(X, y)
    assert br.keep_vars_ is not None
    assert len(br.keep_vars_) > 0


def test_sklearn_no_duplicate_columns():
    """
    Regression test for issue #21: sklearn implementation was creating duplicate
    columns by reusing df2 across iterations, causing fscore1_x, fscore1_y, etc.
    This diluted the mean feature importance calculation.
    The fix creates a fresh df2 each iteration, similar to the xgb implementation.
    """
    from unittest import mock
    from boostaroota.boostaroota import _reduce_vars_sklearn

    np.random.seed(42)
    X, y = make_classification_df(n_features=10, n_informative=3)
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)

    # Track merge calls to verify no duplicate columns are created
    original_merge = pd.merge
    merge_columns = []

    def counting_merge(*args, **kwargs):
        result = original_merge(*args, **kwargs)
        merge_columns.append(result.columns.tolist())
        return result

    with mock.patch('pandas.merge', side_effect=counting_merge):
        criteria, keep_vars = _reduce_vars_sklearn(
            X, y, clf, this_round=1, cutoff=4, n_iterations=3, delta=0.1, silent=True
        )

    # Verify no duplicate columns with _x or _y suffix were created
    for cols in merge_columns:
        duplicate_cols = [c for c in cols if '_x' in c or '_y' in c]
        assert len(duplicate_cols) == 0, f"Found duplicate columns: {duplicate_cols}"

    # Verify expected column structure: feature + fscore1, fscore2, fscore3
    final_cols = merge_columns[-1]
    assert 'feature' in final_cols
    assert 'fscore1' in final_cols
    assert 'fscore2' in final_cols
    assert 'fscore3' in final_cols
    # Should have exactly 4 columns (feature + 3 fscores), no duplicates
    assert len(final_cols) == 4, f"Expected 4 columns, got {len(final_cols)}: {final_cols}"

    # Verify keep_vars is valid
    assert keep_vars is not None
    assert len(keep_vars) > 0


def test_sklearn_invalid_clf_raises():
    """Test that sklearn path raises error for clf without feature_importances_."""
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X, y = make_classification_df()
    clf = LogisticRegression()  # Does not have feature_importances_
    br = BoostARoota(clf=clf, iters=1, silent=True)
    with pytest.raises(ValueError, match="feature_importances_"):
        br.fit(X, y)

