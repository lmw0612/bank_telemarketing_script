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
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [1, 3, 5, 7],
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1_macro',
        n_jobs=1,
        cv=10
    )
    model.fit(X, y)
    print(model.best_score)
    print(model.best_estimator.get_params())

