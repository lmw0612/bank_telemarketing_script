import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


if __name__ == '__main__':
    X = df.drop("response", axis=1).values
    y = df.response.values

    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20),
        "criterion": ["gini", "entropy"]
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
