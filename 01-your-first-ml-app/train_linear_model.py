#!/usr/bin/env python

import numpy	as np
import tensorflow	as tf

from sklearn import datasets, metrics


# 1. Collect data
iris	= datasets.load_iris()

# 2. Pick the model
feature_columns	= [tf.contrib.layers.real_valued_column('', dimension=4)]

classifier = tf.contrib.learn.LinearClassifier(
	feature_columns	= feature_columns,
	n_classes	= 3,
	model_dir	= 'linear_model',
)

# 3. Train the model
classifier.fit(
	x	= iris.data,
	y	= iris.target,
	steps	= 2000,
)

# 4. Test the model
score	= metrics.accuracy_score(iris.target, classifier.predict(iris.data))

print("Accuracy: %f" % score)
