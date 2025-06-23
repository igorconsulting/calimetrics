import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.base import ClassifierMixin
import numpy as np

class CalibrationPlotter:
    """
    Plot calibration curves for one or multiple probabilistic classifiers.

    This class provides utilities to visualize calibration (reliability)
    curves using scikit-learn's `CalibrationDisplay`. It supports plotting
    individual curves or comparing multiple models in a single plot.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to discretize the predicted probability space for
        estimating calibration curves.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the bins:
        - 'uniform': equally spaced bins
        - 'quantile': bins with approximately equal number of samples

    Attributes
    ----------
    displays : list of CalibrationDisplay
        List of `CalibrationDisplay` objects created during plotting.

    Examples
    --------
    >>> from calimetrics.plotting import CalibrationPlotter
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> plotter = CalibrationPlotter(n_bins=10)
    >>> plotter.plot(model, X_test, y_test, label="LogReg")

    >>> models = {
    ...     "Original": base_model,
    ...     "Calibrated": calibrated_model
    ... }
    >>> plotter.compare(models, X_test, y_test)
    """
    def __init__(self, n_bins=10, strategy="uniform"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.displays = []

    def plot(self, model: ClassifierMixin, X, y, label=None):
        """
        Plot the calibration curve for a single classifier.

        This method plots the reliability diagram (calibration curve) using
        the predicted probabilities of a classifier and the true binary labels.

        Parameters
        ----------
        model : ClassifierMixin
            A fitted probabilistic binary classifier that implements 
            `predict_proba`.

        X : array-like of shape (n_samples, n_features)
            Input features to be passed to the model for prediction.

        y : array-like of shape (n_samples,)
            True binary labels (0 or 1).

        label : str, default=None
            Name to display in the legend. If None, the classifier’s default 
            name is used.

        Returns
        -------
        display : CalibrationDisplay
            The scikit-learn CalibrationDisplay object used for plotting.

        Examples
        --------
        >>> from calimetrics.plotting import CalibrationPlotter
        >>> plotter = CalibrationPlotter(n_bins=10)
        >>> plotter.plot(model, X_test, y_test, label="Calibrated SVM")
        """
        display = CalibrationDisplay.from_estimator(
            model,
            X,
            y,
            n_bins=self.n_bins,
            strategy=self.strategy,
            name=label,
        )
        self.displays.append(display)
        return display

    def compare(self, models: dict, X, y):
        """
        Plot calibration curves for multiple classifiers on the same axis.
    
        This method allows for comparative visualization of calibration 
        performance across several models. All curves are overlaid in the
        same reliability diagram.
    
        Parameters
        ----------
        models : dict of str → ClassifierMixin
            Dictionary where keys are model names and values are fitted 
            probabilistic binary classifiers that implement `predict_proba`.
    
        X : array-like of shape (n_samples, n_features)
            Input features to be passed to each model for prediction.
    
        y : array-like of shape (n_samples,)
            True binary labels (0 or 1).
    
        Returns
        -------
        None
    
        Examples
        --------
        >>> models = {
        ...     "Original": base_model,
        ...     "Isotonic Calibrated": iso_model,
        ...     "Sigmoid Calibrated": sig_model
        ... }
        >>> from calimetrics.plotting import CalibrationPlotter
        >>> plotter = CalibrationPlotter(n_bins=15, strategy="quantile")
        >>> plotter.compare(models, X_test, y_test)
        """
        fig, ax = plt.subplots()
        for label, model in models.items():
            display = CalibrationDisplay.from_estimator(
                model,
                X,
                y,
                n_bins=self.n_bins,
                strategy=self.strategy,
                name=label,
                ax=ax
            )
            self.displays.append(display)
        ax.set_aspect("equal")
        ax.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
