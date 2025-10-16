from datetime import datetime
import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


def _to_1d_np(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            return np.asarray(x)
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy().ravel()
    return np.asarray(x).ravel()


def _normalize_targets(y_true: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true)
    if yt.ndim > 1:
        yt = np.argmax(yt, axis=1)

    if yt.size:
        uniq = set(np.unique(yt[~np.isnan(yt)]).tolist() if yt.dtype.kind == "f" else np.unique(yt).tolist())
        if uniq.issubset({-1, 0, 1}) and (-1 in uniq or 0 in uniq):
            yt = (yt > 0).astype(int)

    return yt


def _normalize_predictions(y_pred: np.ndarray) -> np.ndarray:
    yp = np.asarray(y_pred)
    if yp.ndim > 1:
        return np.argmax(yp, axis=1)
    if yp.dtype.kind == "f":
        mn, mx = np.nanmin(yp), np.nanmax(yp)
        if 0.0 <= mn <= 1.0 and 0.0 <= mx <= 1.0:
            return (yp >= 0.5).astype(int)
        return (yp >= 0.0).astype(int)
    return yp


def _to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:

    X = df[cols]
    arr = X.to_numpy(copy=False)
    if arr.dtype == object:
        # coerce each column to numeric if any object dtype slipped in
        arr = np.stack(
            [pd.to_numeric(X[c], errors="coerce").to_numpy() for c in cols],
            axis=1,
        )
    return arr.astype(np.float32, copy=False)

@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:

    df: pd.DataFrame = dataset.data
    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    y_true_np = _to_1d_np(df[target_col])
    X_np = _to_numeric_matrix(df, feature_cols)

    
    y_pred = None
    try:
        import torch  # used only for tensor conversion / torch forward
        X_tensor = torch.from_numpy(X_np)

        y_pred = functional_model.predict(X_tensor)

        # Convert torch output to numpy
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
    except TypeError:
        # Some variants accept numpy directly
        try:
            y_pred = functional_model.predict(X_np)
        except Exception:
            y_pred = None
    except Exception:
        y_pred = None

    if y_pred is None and hasattr(model, "predict"):
        try:
            y_pred = model.predict(X_np)
        except Exception:
            y_pred = None

    if y_pred is None:
        try:
            import torch
            X_tensor = torch.from_numpy(X_np)
            with torch.no_grad():
                out = model(X_tensor)  # raw torch forward
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            y_pred = np.asarray(out)
        except Exception:
            y_pred = None

    if y_pred is None:
        raise RuntimeError(
            "accuracy: unable to obtain predictions via functional_model or model."
        )

    # 3) Compute accuracy
    yt = _normalize_targets(y_true_np)
    yp = _normalize_predictions(np.asarray(y_pred))

    n = min(yt.size, yp.size)
    acc = float((yp[:n] == yt[:n]).mean()) if n > 0 else float("nan")

    return [Measure(name="accuracy", score=acc, time=datetime.now())]
