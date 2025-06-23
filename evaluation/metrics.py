from sklearn.metrics import (
    brier_score_loss,
    matthews_corrcoef,
    confusion_matrix,
    fbeta_score
)
import numpy as np
import json

class CalibratorMetrics:
    """
    Compute calibration and classification metrics for probabilistic classifiers.

    This class evaluates the output of binary classifiers in terms of both
    probability calibration and classification performance. It provides
    common metrics such as Brier Score, Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), Matthews Correlation Coefficient (MCC),
    sensitivity, specificity, and F-beta score.

    Parameters
    ----------
    model : estimator object
        A fitted classifier implementing `predict_proba`.

    X : array-like of shape (n_samples, n_features)
        Feature matrix used to evaluate the model.

    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    n_bins : int, default=10
        Number of bins to use for calibration error metrics (ECE and MCE).

    strategy : str, default='uniform'
        Not currently used, but included for future compatibility
        with different binning strategies (e.g., 'quantile').

    beta : float, default=1.0
        Weight of recall in the F-beta score.

    threshold : float, default=0.5
        Classification threshold applied to predicted probabilities
        to produce binary decisions for classification metrics.

    Attributes
    ----------
    _probs : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.

    _preds : ndarray of shape (n_samples,)
        Binary predictions derived from thresholding the probabilities.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from calimetrics.metrics import CalibratorMetrics
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> metrics = CalibratorMetrics(model, X_val, y_val)
    >>> results = metrics.all_metrics()
    >>> metrics.export_json("metrics.json")
    """
    def __init__(self, model, X, y, n_bins=10, strategy='uniform', beta=1.0, threshold=0.5):
        self.model = model
        self.X = X
        self.y = y
        self.n_bins = n_bins
        self.strategy = strategy
        self.beta = beta
        self.threshold = threshold
        self._probs = None
        self._preds = None
        self._get_predictions()

    def _get_predictions(self):
        self._probs = self.model.predict_proba(self.X)[:, 1]
        self._preds = (self._probs >= self.threshold).astype(int)

    def brier_score(self):
        return brier_score_loss(self.y, self._probs)

    def expected_calibration_error(self):
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        binids = np.digitize(self._probs, bin_edges) - 1
        bin_total = np.zeros(self.n_bins)
        bin_prob = np.zeros(self.n_bins)
        bin_true = np.zeros(self.n_bins)

        for b in range(self.n_bins):
            idx = binids == b
            if np.any(idx):
                bin_total[b] = np.sum(idx)
                bin_prob[b] = np.mean(self._probs[idx])
                bin_true[b] = np.mean(self.y[idx])

        valid = bin_total > 0
        weights = bin_total[valid] / len(self.y)
        return np.sum(weights * np.abs(bin_prob[valid] - bin_true[valid]))

    def max_calibration_error(self):
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        binids = np.digitize(self._probs, bin_edges) - 1
        bin_prob = np.zeros(self.n_bins)
        bin_true = np.zeros(self.n_bins)

        for b in range(self.n_bins):
            idx = binids == b
            if np.any(idx):
                bin_prob[b] = np.mean(self._probs[idx])
                bin_true[b] = np.mean(self.y[idx])

        errors = np.abs(bin_prob - bin_true)
        return np.nanmax(errors)

    def mcc(self):
        return matthews_corrcoef(self.y, self._preds)

    def sensitivity(self):
        _, _, fn, tp = confusion_matrix(self.y, self._preds).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan

    def specificity(self):
        tn, fp, _, _ = confusion_matrix(self.y, self._preds).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan

    def fbeta(self):
        return fbeta_score(self.y, self._preds, beta=self.beta)

    def all_metrics(self):
        return {
            "Brier Score": self.brier_score(),
            "ECE": self.expected_calibration_error(),
            "MCE": self.max_calibration_error(),
            "MCC": self.mcc(),
            "Sensitivity": self.sensitivity(),
            "Specificity": self.specificity(),
            f"F_{self.beta}-Score": self.fbeta(),
        }

    def export_json(self, filepath=None):
        metrics = self.all_metrics()
        if filepath:
            with open(filepath, "w") as f:
                json.dump(metrics, f, indent=4)
        return metrics
    