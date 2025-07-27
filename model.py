import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def load_dataset(name, custom_file=None):
    if name == "Custom" and custom_file:
        try:
            df = pd.read_csv(custom_file)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y
        except Exception as e:
            print(f"Error loading custom file: {e}")
            return None, None
    
    if name == "Moons":
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=100, noise=0.1, random_state=42)
    elif name == "Classification":
        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                                 n_redundant=0, n_clusters_per_class=1, random_state=42)
    elif name == "Sensor (Simulated)":
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
    else:
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model.best_estimator_, accuracy

def generate_plot(X, y, model):
    plt.figure(figsize=(8, 6))
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode('ascii')
