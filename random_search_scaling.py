import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing # for non-tree based models-> scaling
from sklearn import decomposition
from sklearn import pipeline

if __name__ == '__main__':
    X = df.drop("response", axis=1).values
    y = df.response.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("rf", rf)])

    param_grid = {
        "pca__n_components": np.arange(5, 10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1, 20),
        "rf__criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=rf,
        param_distribution=param_grid,
        n_iter=10
        scoring='f1_macro',
        n_jobs=1,
        cv=10
    )
    model.fit(X, y)
    print(model.best_score)
    print(model.best_estimator.get_params())