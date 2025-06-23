import json


def export_results_to_json(results: dict, filepath: str):
    """
    Export cross-validation results to a JSON file.

    This function serializes the results dictionary returned by
    `cross_validate` and writes it to disk in JSON format. All array-like
    values are converted to lists of floats to ensure compatibility with
    the JSON specification.

    Parameters
    ----------
    results : dict
        A dictionary containing cross-validation results. Typically the output
        of `sklearn.model_selection.cross_validate`.

    filepath : str
        The path (including filename) where the JSON file will be saved.

    Returns
    -------
    None
        The function writes the file to disk and does not return anything.
    """
    serializable = {
        k: list(map(float, v)) if hasattr(v, '__iter__') else v
        for k, v in results.items()
    }
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=4)