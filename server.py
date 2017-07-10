"""
An http wrapper for the code in `predict.py`.
"""

import pickle

from flask import Flask, request
from flask.ext.runner import Runner
import numpy as np

import predict


# So many globals! :(
with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

data = {}

app = Flask(__name__)
runner = Runner(app)


@app.route('/', methods=['POST'])
def backchannel_handler():
    json = request.json
    ids = []
    samples = []
    for key, value in json.items():
        ids.append(int(key))
        samples.append(value)
    response = predict.predict(clf, data, ids, np.array(samples))
    return ','.join(str(x) for x in response)


if __name__ == '__main__':
    runner.run()
