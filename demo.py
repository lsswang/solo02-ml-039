import json
from set_model import fit, event_probability
from panel import select_from_model

if __name__ == '__main__':
    features = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    records = [{'membership': [1, 0, 0], 'weight': 3}, {'membership': [0, 1, 1], 'weight': 1}]
    model = fit(features, records, [0.0, 0.0])
    print(json.dumps({'model': {**model, 'parameters': model['parameters'].tolist()},
                      'event_probability': event_probability(features, model['parameters'], [1, 0, 0]),
                      'selection': select_from_model(features, model['parameters'], 2)}, indent=2))
