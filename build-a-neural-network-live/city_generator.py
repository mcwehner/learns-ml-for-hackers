#!/usr/bin/env python3

import os
import ssl
import tflearn

from tflearn.data_utils	import *

# Retrieve source data from:
# https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt

# Input data filename
path	= 'US_cities.txt'

# Maximum city name length
maxlen	= 20

# Vectorize the text file
X, Y, char_idx	= textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

# Build graph
graph	= tflearn.input_data(shape=[None, maxlen, len(char_idx)])
graph	= tflearn.lstm(graph, 512, return_seq=True)
graph	= tflearn.dropout(graph, 0.5)
graph	= tflearn.lstm(graph, 512)
graph	= tflearn.dropout(graph, 0.5)
graph	= tflearn.fully_connected(graph, len(char_idx), activation='softmax')
graph	= tflearn.regression(graph, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Generate cities
model = tflearn.SequenceGenerator(
	graph,
	dictionary	= char_idx,
	seq_maxlen	= maxlen,
	clip_gradients	= 5.0,
	checkpoint_path	= 'model_us_cities',
)

# Training
for i in range(40):
	seed	= random_sequence_from_textfile(path, maxlen)

	model.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id='us cities')

	for temperature in (1.2, 1.0, 0.5):
		print('temperature: %f' % temperature)
		print(model.generate(30, temperature=temperature, seq_seed=seed))
