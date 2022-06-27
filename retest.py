
import nltk
import ssl

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


#Preprocess Data

import numpy as np
import tflearn
import tensorflow as tf
# tf.to_float = lambda x: tf.cast(x,tf.float32)
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import random
import json
import pickle

with open("own.json") as file:
	data = json.load(file)

try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data['intents']: 
		for pattern in intent['patterns']:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent['tag'])

		if intent['tag'] not in labels:
			labels.append(intent['tag'])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

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

	with open("data.pickle", "wb") as f:
	 	pickle.dump((words, labels, training, output), f)

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

try:
	model.load("model.tflearn")
except:
	#show data 1000 times to train
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")



def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for sentence in s_words:
		for i, w in enumerate(words):
			if w == sentence:
				bag[i] = 1
	
	return np.array(bag)


## whatsapp
import os
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_session import Session
from twilio.twiml.messaging_response import Message, MessagingResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
sess = Session()
app.static_folder = 'static'

def chat2(text):
	while True:
			query = text
			results = model.predict([bag_of_words(query, words)])
			results_index = np.argmax(results)
			tag = labels[results_index]

			for tg in data['intents']:
					if tg['tag'] == tag:
						responses = tg['responses']
			x=random.choice(responses)
	
			return str(x)

def respond(message):
    response = MessagingResponse()
    response.message(message)
    return str(response)

def respondv():
    response = VoiceResponse()
    response.gather()
    return response


# whatsapp reply
@app.route('/message', methods=['POST'])

def reply():
    message = request.form.get('Body').lower()
    if message:

        bot = chat2(str(message))

        
        return respondv(bot) #and redirect(url_for('get'))

    
if __name__ == "__main__":
    app.run(debug=True)

