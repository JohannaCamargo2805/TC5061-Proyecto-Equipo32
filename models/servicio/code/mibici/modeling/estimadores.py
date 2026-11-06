"""Estimadores propios del proyecto, compatibles con la API de scikit-learn.

Se guardan dentro del modelo registrado en MLflow (``code_paths``), de modo que
``mlflow.pyfunc.load_model("models:/<nombre>@champion")`` funciona sin instalar
el paquete ``mibici``.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid


class NaiveEstacional(BaseEstimator, RegressorMixin):
    """Baseline de P1: predice el valor de la columna ``lag_{lag_dias}`` (mismo día de la semana anterior)."""

    def __init__(self, lag_dias: int = 7):
        self.lag_dias = lag_dias

    def fit(self, x, y=None):
        self.columna_ = f"lag_{self.lag_dias}"
        return self

    def predict(self, x):
        return np.asarray(x[f"lag_{self.lag_dias}"], dtype="float64")


class Log1pColumnas(BaseEstimator, TransformerMixin):
    """Aplica ``log1p`` a las columnas indicadas y conserva el orden de las demás.

    Reproduce el preprocesamiento de P3 (``preparar_features``): las variables de
    intensidad (salidas y llegadas por día) entran en escala logarítmica.
    """

    def __init__(self, columnas=("salidas_dia", "llegadas_dia")):
        self.columnas = columnas

    def fit(self, x, y=None):
        self.feature_names_in_ = np.asarray(list(x.columns), dtype=object)
        return self

    def transform(self, x):
        x = pd.DataFrame(x, columns=self.feature_names_in_).copy()
        for c in self.columnas:
            x[c] = np.log1p(x[c].astype("float64"))
        return x

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class AglomerativoConCentroides(BaseEstimator, ClusterMixin):
    """Clustering jerárquico aglomerativo con asignación de nuevas estaciones.

    ``AgglomerativeClustering`` no tiene ``predict``; para poder evaluar en TEST y
    servir el modelo, cada estación nueva se asigna al centroide más cercano de
    los grupos obtenidos en TRAIN.
    """

    def __init__(self, n_clusters: int = 4, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, x, y=None):
        x = np.asarray(x, dtype="float64")
        self.labels_ = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage
        ).fit_predict(x)
        self.asignador_ = NearestCentroid().fit(x, self.labels_)
        self.cluster_centers_ = self.asignador_.centroids_
        return self

    def predict(self, x):
        return self.asignador_.predict(np.asarray(x, dtype="float64"))


class MezclaGaussiana(BaseEstimator, ClusterMixin):
    """``GaussianMixture`` con ``labels_`` para usarlo igual que K-Means."""

    def __init__(self, n_clusters: int = 4, covariance_type: str = "diag", n_init: int = 5,
                 random_state: int | None = None):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, x, y=None):
        x = np.asarray(x, dtype="float64")
        self.gmm_ = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(x)
        self.labels_ = self.gmm_.predict(x)
        self.cluster_centers_ = self.gmm_.means_
        return self

    def predict(self, x):
        return self.gmm_.predict(np.asarray(x, dtype="float64"))
