"""Existing independent item-selection model; synthetic CPU examples only."""
import numpy as np
from scipy.optimize import minimize


def probabilities(features, parameters):
    x, w = np.asarray(features, float), np.asarray(parameters, float)
    z = x @ w
    return 1.0 / (1.0 + np.exp(-z))


def event_probability(features, parameters, membership):
    p = probabilities(features, parameters)
    m = np.asarray(membership)
    if m.shape != p.shape or not np.isin(m, [0, 1]).all():
        raise ValueError("The existing model requires every membership to be known")
    return float(np.prod(np.where(m == 1, p, 1 - p)))


def objective_gradient(features, parameters, records, ridge=0.1):
    x, w = np.asarray(features, float), np.asarray(parameters, float)
    p = probabilities(x, w)
    value, grad = ridge * float(w @ w) / 2, ridge * w.copy()
    for row in records:
        y, weight = np.asarray(row['membership'], float), row.get('weight', 1)
        value += weight * float(np.sum(np.logaddexp(0, x @ w) - y * (x @ w)))
        grad += weight * (x.T @ (p - y))
    return value, grad


def fit(features, records, initial, ridge=0.1):
    result = minimize(lambda w: objective_gradient(features, w, records, ridge),
                      np.asarray(initial, float), jac=True, method='L-BFGS-B',
                      bounds=[(-2.0, 2.0)] * len(initial))
    return {'parameters': result.x, 'objective': float(result.fun),
            'status': 'optimizer_success' if result.success else 'stopped'}
