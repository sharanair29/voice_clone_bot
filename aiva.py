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

## end of bot



## Flask APP
import os
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_session import Session
from twilio.twiml.messaging_response import Message, MessagingResponse
from twilio.rest import Client

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


ls=[]

ls2=[]



# udemy

@app.route("/")

def index():
	user_logged_in = False
	name = "Akshara"
	letters = list(name)
	pup_dictionary = {'pup_name':'Sammy'}
	return render_template("basic.html", name = name, letters = letters, pup_dictionary = pup_dictionary, user_logged_in =user_logged_in)

@app.route('/puppy/<name>')
def puppy(name):
	return "100th letter: {}".format(name[100])

# Home Page Login

@app.route("/home")

def home():
	return render_template("pages/index.html")



# About Page Login

@app.route("/about")

def about():
	return render_template("pages/about.html")

# Register Page

@app.route("/register")

def register():
	return render_template("pages/register.html")


#whatsapp web
import webbrowser

@app.route("/web")

def web():
	return redirect("https://web.whatsapp.com/")

#QUERY

@app.route("/received")

def received():
	comp_name = request.args.get('Company Name')
	comp_email = request.args.get('Company Email')
	industry = request.args.get('Industry')
	preferred_channel = request.args.get('Preferred Channel')
	purpose = request.args.get('Purpose')

	print(comp_name)
	print(comp_email)
	print(industry)
	print(preferred_channel)
	print(purpose)
	return render_template("pages/received.html")

# Post Register Page

@app.route("/success")

def success():
	first_name = request.args.get('first_name')
	last_name = request.args.get('last_name')
	username = request.args.get('username')
	email = request.args.get('email')
	password = request.args.get('password')
	password2 = request.args.get('password2')
	
	print(first_name)
	print(last_name)
	print(username)
	print(email)
	print(password)
	print(password2)
	return render_template("pages/success.html")


# Login Page

@app.route("/login")

def login():
	return render_template("pages/login.html")



##Landing Page Action : Choose SMS or WhatsApp

@app.route("/landpage")

def land():
	return render_template("pages/landpage.html")

#Create Conversation Template (Intent builder)

@app.route("/intents")

def intents():
	return render_template("pages/intents.html")

## load contacts csv

@app.route("/load")

def load():
	return render_template("pages/load.html")

## send bulk

@app.route("/send_bulk")

def send_bulk():
	return render_template("pages/send_bulk.html")

		

## CHATBOTS
# bot browser
@app.route("/botapp")

def botapp():
    return render_template("chatbot/index2.html") # to send content to html


@app.route("/bot2")

def bot2():
    return render_template("chatbot/bot.html") # to send content to html

@app.route("/get", methods=['GET'])

def get_bot_response():
    userText = request.args.get("msg") # data from input
    ls2.append(f'User: {userText}')
    bot_reply = chat2(userText)
    ls2.append(f'BrowserBot: {bot_reply}')
    print(ls2)
    return (bot_reply)


# whatsapp bot
@app.route('/whatsapp', methods=['GET', 'POST'])

def reply():
    message = request.form.get('Body')
    if message:
        ls.append(f'User: {message}')
        # print(message)
        bot = chat2(str(message))
        ls.append(f'WhatsappBot: {bot}')
        # print(bot)
        # print(ls)
    return (respond(bot))  #and redirect(url_for('get'))




@app.route('/whatsapp_convo', methods=['GET', 'POST'])
def wapage():
    return render_template("chatbot/wa.html", ls = ls)


@app.route('/convos')

def convos():
    return render_template("chatbot/whatsweb.html")





## register conversation to database

@app.route("/db_log")
def db_log():
    ls # to send content to html





# mycursor.execute("INSERT INTO WHATSAPP_CHATS (CONVO_LIST) VALUES ((%s))", json.dumps(ls2,))
# db.commit()

if __name__ == "__main__":
    app.run(debug=True)

