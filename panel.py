"""Existing ordinary independent-score panel selection."""
import numpy as np


def choose(scores, k, costs=None, budget=None):
    scores = np.asarray(scores, float)
    if k < 0 or k > len(scores):
        raise ValueError('Invalid panel size')
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected = order[:k]
    costs = np.ones(len(scores), int) if costs is None else np.asarray(costs, int)
    if budget is not None and int(sum(costs[i] for i in selected)) > budget:
        return {'status': 'top_k_over_budget', 'selected': selected}
    return {'status': 'selected', 'selected': selected,
            'score': float(sum(scores[i] for i in selected)),
            'cost': int(sum(costs[i] for i in selected))}


def select_from_model(features, parameters, k, costs=None, budget=None):
    from set_model import probabilities
    actual_scores = probabilities(features, parameters)
    return {'scores': actual_scores.tolist(), 'panel': choose(actual_scores, k, costs, budget)}
