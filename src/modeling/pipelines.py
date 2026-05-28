# pyright: basic

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED


def _feature_list(feature_names):
    return list(feature_names)


def _scaled_numeric_preprocessor(feature_names):
    features = _feature_list(feature_names)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )


def _imputed_numeric_preprocessor(feature_names):
    features = _feature_list(feature_names)
    return ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)]
    )


def build_ridge_pipeline(feature_names, alpha: float):
    preprocessor = _scaled_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_pls_pipeline(feature_names, n_components: int):
    preprocessor = _scaled_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", PLSRegression(n_components=n_components, scale=False)),
        ]
    )


def build_hist_gbr_pipeline(feature_names):
    preprocessor = _imputed_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=3,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    max_iter=200,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
