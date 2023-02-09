# load and evaluate a saved model
import pickle

from numpy import loadtxt
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import os

def predictrating(comment):
	with open(os.path.abspath('tokenizer.pickle'), 'rb') as handle:
		loaded_tokenizer = pickle.load(handle)
	txt = comment
	seq = loaded_tokenizer.texts_to_sequences([txt])
	max_len=50
	padded = pad_sequences(seq, maxlen=max_len)
	model = load_model(os.path.abspath('model.h5'))
	model.summary()
	Testresults = model.predict(padded)
	df = pd.DataFrame(Testresults, columns=['1', '2', '3', '4', '5'])
	a = list(df)
	result = a.index(max(a))
	print('star rating :', result)
	return result