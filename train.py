"""
Training an ML algorithm to predict listeners backchannel behaviours (head nods)
"""

import pickle
import random

import pandas as pd
from sklearn import neighbors
from sklearn import model_selection
from sklearn import metrics

import training


print('Preparing data')
with pd.HDFStore('data.hdf') as store:
    X, y = training.prepare_training_data(store)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

print('Training model')
clf = neighbors.KNeighborsClassifier(n_jobs=-1)  # Utilize all cores
clf.fit(X_train, y_train)

print('Evaluating model')
yhat = clf.predict(X_test)
print(metrics.classification_report(y_test, yhat))

print('Measuring performance')
sample = X_train[random.randint(0, len(X_train) - 1), :]
performance = training.timeit(lambda: clf.predict([sample]), iterations=100)
print(f'{performance:.02f} ms/prediction')

print('Persisting')
with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)
