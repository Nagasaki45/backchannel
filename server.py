"""
An http wrapper for the code in `predict.py`.
"""

import json
import pickle

from flask import Flask, request
from flask.ext.runner import Runner
import numpy as np

import predict


# So many globals! :(
with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

predictor = predict.BackchannelPredictor(clf)

app = Flask(__name__)
runner = Runner(app)


@app.route('/', methods=['POST'])
def backchannel_handler():
    new_data = request.json
    type_ = new_data.get('type')  # Not mandatory
    listeners = new_data['listeners']
    ids = []
    samples = []
    for key, value in listeners.items():
        ids.append(int(key))
        samples.append(value)
    predictions = predictor.predict(ids, np.array(samples), type_=type_)
    return json.dumps(dict(zip(ids, predictions)))


if __name__ == '__main__':
    runner.run()
