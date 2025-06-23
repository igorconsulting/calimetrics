# Calimetrics

**Calimetrics** is a modular microframework for calibration, evaluation, and diagnostic analysis of probabilistic classifiers. It is designed to support auditable, reproducible, and production-grade model validation workflows.

## Overview

In modern machine learning systems, probabilistic classifiers are frequently used in critical decision-making contexts. However, poorly calibrated probability estimates can lead to significant risk and suboptimal outcomes. **Calimetrics** provides a principled interface to address this issue, offering:

- Post-hoc calibration using frozen base estimators and `CalibratedClassifierCV`
- Support for various cross-validation strategies (`KFold`, `StratifiedKFold`, `GroupKFold`)
- Custom metrics such as Brier Score, Log Loss, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Sensitivity, Specificity, F1-Score, and Matthews Correlation Coefficient (MCC)
- Metric exporting for auditing and reporting
- Comparative calibration curve plotting
- Optional logging for traceability of evaluation configuration

All components are compatible with `scikit-learn` estimators and can be easily integrated into existing pipelines.

## Project Structure

```

calimetrics/
│
├── calibration/         # Calibration logic and plotting
│   ├── calibrator.py
│   └── ploter.py
│
├── evaluation/          # Evaluation execution and custom scorers
│   ├── cross_validator.py
│   └── scorer.py
│   └── metrics.py
│
├── utils/               # Metric implementations and utilities
│   └── metrics.py
│
├── tests/               # Unit and integration tests
│
├── notebooks/           # Interactive examples and demonstrations
│
├── pyproject.toml       # Build and dependency configuration
└── README.md            # Project documentation

````

## Features

- Calibration using frozen estimators to prevent retraining during the calibration process
- Seamless integration with `cross_validate()` and other `scikit-learn` workflows
- Configurable and extensible scoring interface supporting multiple metrics
- Export of results to structured JSON files for reproducibility
- Visualization of multiple calibration curves for comparative analysis
- Logging infrastructure to support experiment tracking and auditability

## Installation

To install the package in development mode:

```bash
git clone https://github.com/your-org/calimetrics.git
cd calimetrics
pip install -e .
````

## Usage Example

### Calibration

```python
from calimetrics.calibration.calibrator import Calibrator
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
calibrator = Calibrator(model=model, method="isotonic", cv=5)
calibrator.fit(X_train, y_train)
probs = calibrator.predict_proba(X_test)
```

### Evaluation

```python
from calimetrics.evaluation.evaluator import run_model_validation
from calimetrics.evaluation.scorer import Scorer
from sklearn.model_selection import StratifiedKFold

scorer = Scorer(n_bins=10)
cv_strategy = StratifiedKFold(n_splits=5)
results = run_model_validation(model, X, y, scorer, cv=cv_strategy)
```

### Exporting Results

```python
from calimetrics.evaluation.evaluator import export_results_to_json

export_results_to_json(results, "metrics_output.json")
```

### Plotting Calibration Curves

```python
from calimetrics.calibration.ploter import CalibrationPlotter

plotter = CalibrationPlotter(n_bins=10)
plotter.compare({"Model A": model_a, "Model B": model_b}, X_val, y_val)
```

## Testing (to-do)

To execute the full test suite:

```bash
pytest tests/
```

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome. Please adhere to clean code principles, write modular and well-tested code, and document all public interfaces. Open issues and submit pull requests via GitHub.


Se desejar, posso também criar o `LICENSE`, `CONTRIBUTING.md`, ou o esqueleto de um `setup.cfg` ou `setup.py` adicionalmente. Deseja isso?
```
