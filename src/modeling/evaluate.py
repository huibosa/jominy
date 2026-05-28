# pyright: basic

from importlib import import_module


def regression_metrics(y_true, y_pred):
    np = import_module("numpy")
    sklearn_metrics = import_module("sklearn.metrics")
    return {
        "mae": float(sklearn_metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))),
        "r2": float(sklearn_metrics.r2_score(y_true, y_pred)),
    }


def monotonic_violation_rate(df) -> float:
    return float((df["j15_pred"] > df["j9_pred"]).mean())


def clipped_delta_metrics(df) -> dict:
    return regression_metrics(df["delta_true"], df["delta_pred_clipped"])
