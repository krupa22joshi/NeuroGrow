from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_dataset(name):
    if name == "Moons":
        X, y = make_moons(n_samples=100, noise=0.1)
    elif name == "Circles":
        X, y = make_circles(n_samples=100, noise=0.1)
    elif name == "Classification":
        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    elif name == "Sensor (Simulated)":
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
    else:
        X, y = make_moons(n_samples=100, noise=0.1)

    y = y.reshape(-1, 1)
    return X, y
