import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

def run_feature_ablation(model, X_train, y_train, X_test, y_test, feature_names):
    results = []
    base = clone(model).fit(X_train, y_train)
    base_mae = mean_absolute_error(y_test, base.predict(X_test))
    results.append({"setting":"full","mae":float(base_mae)})

    for j, name in enumerate(feature_names):
        Xt = np.delete(np.asarray(X_train), j, axis=1)
        Xv = np.delete(np.asarray(X_test), j, axis=1)
        m = clone(model).fit(Xt, y_train)
        mae = mean_absolute_error(y_test, m.predict(Xv))
        results.append({"setting":f"minus:{name}","mae":float(mae)})
    return results
