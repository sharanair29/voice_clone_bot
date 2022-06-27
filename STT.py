#pyttsx3 has been imported
#this can be used to replace the speech function altogether for testing purposes

import speech_recognition as sr
from merge3 import *

#This function simply records the audio input and uses Google Speech Recognition API
#The try statement simply states that if speech input is not recognized to be transcribed to text, 
#then the bot will say "Say that again please" using the speech function 

def takeMic():
    
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("Listening...")
		r.pause_threshold = 1
		audio = r.listen(source)
	try:
		print("Recognizing...")
		query = r.recognize_google(audio, language = "en")
		print(query)

	except Exception as e:
		print(e)
		
		TTS.speech("Say that again please...")
		return "none"
	return query


