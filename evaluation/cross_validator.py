import time
import logging
from typing import Optional
from sklearn.model_selection import cross_validate
from scorer import Scorer

def run_cross_validation(
    model,
    X,
    y,
    scorer_obj: Scorer,
    cv=5,
    groups=None,
    logger: Optional[logging.Logger] = None
):
    """
    Executes cross-validation using multiple scorers, with optional logging.

    Parameters
    ----------
    model : scikit-learn estimator
        Estimator implementing fit/predict_proba.

    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    scorer_obj : Scorer
        Instance providing metric scorers.

    cv : int or CV splitter
        Cross-validation strategy.

    groups : array-like, optional
        Required if using GroupKFold.

    logger : logging.Logger, optional
        Optional logger for auditing execution details.

    Returns
    -------
    results : dict
        Cross-validation metrics per fold.
    """
    if logger:
        logger.info("Cross-validation initiated.")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Scorers: {list(scorer_obj.get_all_scorers().keys())}")
        logger.info(f"CV: {cv.__class__.__name__ if not isinstance(cv, int) else f'{cv}-Fold'}")
        if hasattr(cv, "get_n_splits"):
            try:
                splits = cv.get_n_splits(X, y, groups)
                logger.info(f"Number of folds: {splits}")
            except Exception as e:
                logger.warning(f"Could not determine number of folds: {e}")

    start = time.time()
    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring=scorer_obj.get_all_scorers(),
        cv=cv,
        groups=groups,
        return_train_score=False
    )
    end = time.time()

    if logger:
        logger.info(f"Cross-validation completed in {end - start:.2f} seconds.")
        for key, values in results.items():
            if key.startswith("test_"):
                logger.info(f"{key}: mean={sum(values)/len(values):.4f}")

    return results
