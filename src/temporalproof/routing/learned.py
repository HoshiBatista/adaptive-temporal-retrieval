"""Fold-local logistic fit and safe JSON inference over pre-deep features only."""

import warnings
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.routing.features import FEATURE_VERSION, PreDeepFeatures

TEMPORAL_FEATURES = frozenset(
    {
        "explicit_year_count",
        "query_insertion_known",
        "candidate_insertion_known_fraction",
        "candidate_after_query_fraction",
        "median_abs_insertion_gap_days",
        "candidate_distinct_insertion_days",
    }
)


def feature_names(kind: str) -> tuple[str, ...]:
    names = tuple(PreDeepFeatures.model_fields)
    if kind == "full":
        return names
    if kind == "no_temporal":
        return tuple(name for name in names if name not in TEMPORAL_FEATURES)
    raise ValueError("unknown feature ablation")


def feature_matrix(features: Sequence[PreDeepFeatures], names: Sequence[str]) -> np.ndarray:
    if (
        not names
        or len(set(names)) != len(names)
        or not set(names) <= PreDeepFeatures.model_fields.keys()
    ):
        raise ValueError("only unique pre-deep feature columns are allowed")
    matrix = np.array(
        [
            [np.nan if (v := getattr(row, name)) is None else float(v) for name in names]
            for row in features
        ],
        dtype=np.float64,
    ).reshape(len(features), len(names))
    if np.isinf(matrix).any():
        raise ValueError("nonfinite inference values")
    return matrix


class RouterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    feature_version: Literal[FEATURE_VERSION] = FEATURE_VERSION
    algorithm: Literal["logistic", "constant"]
    feature_names: tuple[str, ...]
    imputer_statistics: tuple[float, ...] = ()
    indicator_indices: tuple[int, ...] = ()
    scaler_mean: tuple[float, ...] = ()
    scaler_scale: tuple[float, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float = 0
    constant_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        feature_matrix([], self.feature_names)
        n = len(self.feature_names)
        p = n + len(self.indicator_indices)
        if self.algorithm == "constant":
            if self.constant_probability is None:
                raise ValueError("constant score is required")
            return self
        if (
            len(self.imputer_statistics) != n
            or len(set(self.indicator_indices)) != len(self.indicator_indices)
            or any(i < 0 or i >= n for i in self.indicator_indices)
            or any(len(v) != p for v in [self.scaler_mean, self.scaler_scale, self.coefficients])
            or any(v <= 0 for v in self.scaler_scale)
            or self.constant_probability is not None
        ):
            raise ValueError("invalid router parameter dimensions")
        return self


def predict_scores(model: RouterModel, features: Sequence[PreDeepFeatures]) -> np.ndarray:
    """Scores from class-weighted logistic fitting are not calibrated probabilities."""
    matrix = feature_matrix(features, model.feature_names)
    if model.algorithm == "constant":
        return np.full(len(features), model.constant_probability, dtype=float)
    missing = np.isnan(matrix)
    filled = np.where(missing, np.asarray(model.imputer_statistics), matrix)
    if model.indicator_indices:
        filled = np.concatenate([filled, missing[:, model.indicator_indices].astype(float)], axis=1)
    scaled = (filled - np.asarray(model.scaler_mean)) / np.asarray(model.scaler_scale)
    logits = scaled @ np.asarray(model.coefficients) + model.intercept
    return np.exp(-np.logaddexp(0, -logits))


def fit_router(
    features: Sequence[PreDeepFeatures],
    labels: Sequence[int],
    *,
    kind: str,
    c: float,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[RouterModel, dict]:
    """Caller passes training rows only; imputer/scaler never inspect validation rows."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    names = feature_names(kind)
    x = feature_matrix(features, names)
    y = np.asarray(labels)
    if len(features) != len(y) or not len(y) or not set(y) <= {0, 1}:
        raise ValueError("invalid binary training rows")
    y = y.astype(int)
    if not np.isfinite([c, tol]).all() or c <= 0 or tol <= 0 or max_iter < 1:
        raise ValueError("invalid logistic configuration")
    info = {
        "training_queries": len(y),
        "positive_count": int(y.sum()),
        "single_class_fallback": len(set(y)) == 1,
    }
    if len(set(y)) == 1:
        return RouterModel(
            algorithm="constant", feature_names=names, constant_probability=float(y[0])
        ), info
    pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=max_iter,
                    tol=tol,
                    random_state=seed,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x, y)
    imp = pipeline.named_steps["impute"]
    scale = pipeline.named_steps["scale"]
    lr = pipeline.named_steps["logistic"]
    model = RouterModel(
        algorithm="logistic",
        feature_names=names,
        imputer_statistics=tuple(imp.statistics_.tolist()),
        indicator_indices=tuple(imp.indicator_.features_.tolist()),
        scaler_mean=tuple(scale.mean_.tolist()),
        scaler_scale=tuple(scale.scale_.tolist()),
        coefficients=tuple(lr.coef_[0].tolist()),
        intercept=float(lr.intercept_[0]),
    )
    if not np.allclose(
        predict_scores(model, features), pipeline.predict_proba(x)[:, 1], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("JSON inference differs from fitted sklearn pipeline")
    info["iterations"] = int(lr.n_iter_[0])
    return model, info
