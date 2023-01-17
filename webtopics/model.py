from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler


class TopicModel(BaseEstimator, ClusterMixin):
    def __init__(self, reduce_dims=.8, min_cluster_size=15, cluster_eps=.25, distance_metric='cosine'):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dim_red = reduce_dims
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_eps
        )
        self.scaler = StandardScaler()
        self.metric = distance_metric

    def fit(self, X, y=None):
        self.features_in_ = X.copy()
        self.embeddings_ = self.encoder.encode(X, show_progress_bar=True)
        self.reducer_ = umap.UMAP(n_components=int(self.embeddings_.shape[1] * (1-self.dim_red)), metric=self.metric)
        self.reduced_embeddings_ = self.scaler.fit_transform(self.reducer_.fit_transform(X))
        self.labels_ = self.clusterer.fit_predict(self.reduced_embeddings_)
        return self

    def predict(self, X):
        raise NotImplementedError

    def get_results(self):
        return pd.DataFrame({
            'X': self.features_in_,
            'label': self.labels_
        })
    