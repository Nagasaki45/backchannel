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
    ids = []
    samples = []
    for key, value in new_data.items():
        ids.append(int(key))
        samples.append(value)
    predictions = predictor.predict(ids, np.array(samples))
    # tolist to fix serialization issue http://bugs.python.org/issue18303
    return json.dumps(dict(zip(ids, predictions.tolist())))


if __name__ == '__main__':
    runner.run()
