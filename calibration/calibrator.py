from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin
from sklearn.frozen import FrozenEstimator

class Calibrator:
    """
    Calibration wrapper for scikit-learn classifiers.

    This class wraps a fitted classifier and applies post-training probability
    calibration using either Platt scaling (sigmoid) or isotonic regression,
    through scikit-learn's `CalibratedClassifierCV`. The base model is frozen to
    avoid re-fitting during the calibration process.

    Parameters
    ----------
    model : ClassifierMixin
        A fitted scikit-learn classifier implementing `predict_proba`.

    method : {'sigmoid', 'isotonic'}, default='isotonic'
        The calibration method to use. 'sigmoid' refers to Platt scaling,
        while 'isotonic' uses isotonic regression.

    cv : int, cross-validation generator, or iterable, default=None
        Determines the cross-validation splitting strategy for calibration.
        If None, uses the default 5-fold cross-validation.

    n_jobs : int, default=-1
        Number of jobs to run in parallel for cross-validation. `-1` means using all processors.

    ensemble : bool, default=True
        If True, calibrates an ensemble of models fitted on each cross-validation split.
        If False, fits a single calibrated model on the entire training set.

    Attributes
    ----------
    calibrated_model : CalibratedClassifierCV or None
        The calibrated model instance after `build()` or `fit()` is called.

    calibrated_classifiers_ : list of calibrated classifiers
        Available only after the model is fitted. Equivalent to the
        `calibrated_classifiers_` attribute of `CalibratedClassifierCV`.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from calimetrics.calibrator import Calibrator
    >>> X, y = make_classification()
    >>> base_model = LogisticRegression().fit(X, y)
    >>> calibrator = Calibrator(model=base_model, method='sigmoid')
    >>> calibrator.fit(X, y)
    >>> probs = calibrator.predict_proba(X)
    """
    def __init__(
        self,
        model: ClassifierMixin,
        method: str = "isotonic",
        cv=None,
        n_jobs: int = -1,
        ensemble: bool = True
    ):
        self.model = model
        self._method = None
        self._cv = None
        self._n_jobs = None
        self._ensemble = None
        self.calibrated_model = None

        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value: str):
        if value not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self._method = value
        self._invalidate_model()

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value
        self._invalidate_model()

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_jobs must be an integer")
        self._n_jobs = value
        self._invalidate_model()

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("ensemble must be a boolean")
        self._ensemble = value
        self._invalidate_model()

    def _invalidate_model(self):
        """Invalidate calibrated model if parameters change."""
        self.calibrated_model = None

    def build(self):
        """Build and store the calibrated model with frozen base."""
        frozen_model = FrozenEstimator(self.model)
        self.calibrated_model = CalibratedClassifierCV(
            estimator=frozen_model,
            method=self.method,
            cv=self.cv,
            n_jobs=self.n_jobs,
            ensemble=self.ensemble
        )
        return self.calibrated_model

    def fit(self, X, y):
        if self.calibrated_model is None:
            self.build()
        self.calibrated_model.fit(X, y)

    @property
    def calibrated_classifiers_(self):
        if self.calibrated_model is None:
            raise AttributeError("Model has not been calibrated yet.")
        return self.calibrated_model.calibrated_classifiers_

    def predict_proba(self, X):
        if self.calibrated_model is None:
            raise RuntimeError("Calibrated model not yet built or fitted.")
        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        if self.calibrated_model is None:
            raise RuntimeError("Calibrated model not yet built or fitted.")
        return self.calibrated_model.predict(X)