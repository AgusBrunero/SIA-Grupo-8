import numpy as np


class PCA:
    """PCA por autodescomposición de la matriz de covarianza (sobre datos ya estandarizados
    es la matriz de correlación)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        cov = np.cov(X - self.mean, rowvar=False, bias=True)
        valores, vectores = np.linalg.eigh(cov)
        orden = np.argsort(valores)[::-1]
        self.eigenvalues = np.clip(valores[orden], 0, None)
        # Columna j = autovector de PCj. Signo fijado para que la carga más grande sea positiva.
        vectores = vectores[:, orden]
        signos = np.sign(vectores[np.abs(vectores).argmax(axis=0), range(vectores.shape[1])])
        self.components = vectores * signos
        self.explained_variance_ratio = self.eigenvalues / self.eigenvalues.sum()
        return self

    def transform(self, X, n_components=None):
        return (np.asarray(X, dtype=float) - self.mean) @ self.components[:, :n_components]

    def loadings(self):
        """Cargas = autovector × √autovalor: correlación de cada variable con cada PC
        (si los datos están estandarizados)."""
        return self.components * np.sqrt(self.eigenvalues)
