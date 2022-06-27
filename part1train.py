import nltk
import ssl

# This part only needed to be run once to enable nltk to be downloaded hence its commented out for reruns

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
stemmer = LancasterStemmer()

## PREPROCESSING DATA IN INTENTS JSON FILE

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

#Load intents using own.json

with open("/Users/hlabs/Desktop/hoop/voicebot/Real-Time-Voice-Cloning/own.json") as file:
	data = json.load(file)

# We create 4 empty lists to loop through json file
words = []
labels = []
docs_x = []
docs_y = []

#Loop through dictionaries in own.json
#Tokenize intents file (own.json)
#Get root meaning of the word to train the bot

for intent in data['intents']: 
	for pattern in intent['patterns']:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent['tag'])

	if intent['tag'] not in labels:
		labels.append(intent['tag'])

# Lemmatize and stem words in intents, this step allows us to make sure the model understands that words like run and running mean the same thing
# Hence, we are kind of saying should w in words (where we have declared run and running are simply just run) is equal to another word, so run = run we don't cause our list of words to have duplicates
# we eliminate repeats in our list of words from the intents json file

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# bag of words model

for x, doc in enumerate(docs_x):
	bag = []

	wrds= [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = np.array(training)
output = np.array(output)


#tf 2.5.0
tf.compat.v1.reset_default_graph()

# length of training data
net = tflearn.input_data(shape=[None,len(training[0])])

# 2 hidden layers with 8 neurons

net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

#output layer neurons represent each of labels
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# DNN neural network model
model = tflearn.DNN(net)

#for intial run when no model is generated yet remove try exception
#after the 1st run when a model already exists include try except statement with indentation

# try:
# 	model.load("model.tflearn")

# except:
# show data 1000 times to train
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
