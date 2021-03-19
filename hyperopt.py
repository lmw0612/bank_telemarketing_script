import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from functools import partial
from skopt import space
from skopt import gp_minimize
from hyperopt import hp, fmin, tpe, Trials


def optimize(params, x, y):
    model = ensemble.RandomForestClassifier(**params)
    kf - model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        X_train = x[train_idx]
        y_train = y[train_idx]

        X_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_acc = metrics.f1_score(y_test, pred)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)

if __name__ == '__main__':
    X = df.drop("response", axis=1).values
    y = df.response.values

    param_space = [
        space.Intger(3, 15, name="max_depth"),
        space.Intger(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    print(dict(param_names, result.x))