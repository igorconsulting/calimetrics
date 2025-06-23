import numpy as np
from sklearn.metrics import (
    make_scorer,
    brier_score_loss,
    log_loss
)
from sklearn.base import ClassifierMixin

class Scorer:
    """
    Scorer is a utility class for evaluating probabilistic classifiers
    using advanced calibration metrics. It provides scorer functions 
    compatible with scikit-learn's `cross_validate` and `GridSearchCV`.

    This class is particularly useful in contexts where not only 
    classification accuracy but also probability calibration is 
    critical, such as risk assessment, medical diagnostics, or 
    decision-theoretic applications.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to use when computing calibration errors (ECE, MCE).

    Methods
    -------
    expected_calibration_error(y_true, y_prob)
        Computes the Expected Calibration Error (ECE) using fixed-width binning.

    max_calibration_error(y_true, y_prob)
        Computes the Maximum Calibration Error (MCE), the worst-case bin gap.

    ece_score_func(estimator, X, y)
        Scoring function for ECE to be used with `make_scorer`.

    mce_score_func(estimator, X, y)
        Scoring function for MCE to be used with `make_scorer`.

    brier_score_func(estimator, X, y)
        Scoring function for negative Brier Score.

    log_loss_func(estimator, X, y)
        Scoring function for negative Log Loss.

    get_ece_scorer()
        Returns a `scorer` object for ECE (to minimize).

    get_mce_scorer()
        Returns a `scorer` object for MCE (to minimize).

    get_brier_scorer()
        Returns a `scorer` object for Brier Score (to minimize).

    get_logloss_scorer()
        Returns a `scorer` object for Log Loss (to minimize).

    get_all_scorers()
        Returns a dictionary of all supported scorers with proper names.

    Examples
    --------
    >>> scorer = Scorer(n_bins=15)
    >>> scorers = scorer.get_all_scorers()
    >>> from sklearn.model_selection import cross_validate
    >>> results = cross_validate(model, X, y, scoring=scorers, cv=5)
    """
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def expected_calibration_error(self, y_true, y_prob):
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1

        bin_total = np.zeros(self.n_bins)
        bin_confidence = np.zeros(self.n_bins)
        bin_accuracy = np.zeros(self.n_bins)

        for b in range(self.n_bins):
            idx = binids == b
            if np.any(idx):
                bin_total[b] = np.sum(idx)
                bin_confidence[b] = np.mean(y_prob[idx])
                bin_accuracy[b] = np.mean(y_true[idx])

        valid = bin_total > 0
        weights = bin_total[valid] / len(y_true)
        return np.sum(weights * np.abs(bin_confidence[valid] - bin_accuracy[valid]))

    def max_calibration_error(self, y_true, y_prob):
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1

        bin_confidence = np.zeros(self.n_bins)
        bin_accuracy = np.zeros(self.n_bins)

        for b in range(self.n_bins):
            idx = binids == b
            if np.any(idx):
                bin_confidence[b] = np.mean(y_prob[idx])
                bin_accuracy[b] = np.mean(y_true[idx])

        return np.nanmax(np.abs(bin_confidence - bin_accuracy))

    def ece_score_func(self, estimator: ClassifierMixin, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return -self.expected_calibration_error(y, proba)

    def mce_score_func(self, estimator: ClassifierMixin, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return -self.max_calibration_error(y, proba)

    def brier_score_func(self, estimator: ClassifierMixin, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return -brier_score_loss(y, proba)

    def log_loss_func(self, estimator: ClassifierMixin, X, y):
        proba = estimator.predict_proba(X)
        return -log_loss(y, proba)

    def get_ece_scorer(self):
        return make_scorer(self.ece_score_func, greater_is_better=False)

    def get_mce_scorer(self):
        return make_scorer(self.mce_score_func, greater_is_better=False)

    def get_brier_scorer(self):
        return make_scorer(self.brier_score_func, greater_is_better=False)

    def get_logloss_scorer(self):
        return make_scorer(self.log_loss_func, greater_is_better=False)
    
    def get_all_scorers(self):
        return {
            "ece": self.get_ece_scorer(),
            "mce": self.get_mce_scorer(),
            "brier": self.get_brier_scorer(),
            "logloss": self.get_logloss_scorer()
        }

