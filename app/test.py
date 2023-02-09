# load and evaluate a saved model
import pickle

from keras.models import load_model
import pandas as pd
from keras_preprocessing.sequence import pad_sequences

import numpy as np
import os


def scoreresumeall(self, resumetext):
	# Convert all strings to lowercase
	text = resumetext.lower()

	# Remove numbers
	text = re.sub(r'\d+', '', text)

	# Remove punctuation
	text = text.translate(str.maketrans('', '', string.punctuation))
	# Create dictionary with industrial and system engineering key terms by area
	terms = {
		'1': ['very disappointed', 'disappointed', "aren't  good", "aren't great", "isn't good", "isn't great",
			  'not too overwhelming', 'bad packaging', 'wet', 'worst product', 'wastage', 'nothing happened',
			  'stickier', 'greasy', 'aisa kyu hai ?????????', 'extremely dry', 'waste', 'nt good', 'rough', 'expensive',
			  'fail', 'worst', 'bad', 'ugly', 'baddest', 'inferior', 'poor', 'least', 'minor', 'poorest', 'secondary',
			  'unimportant', 'worst', 'incorrect', 'not right', 'atrocious', 'awful', 'cheap', 'crummy', 'dreadful',
			  'lousy', 'rough', 'sad', 'unacceptable', 'blah', 'bummer', 'diddly', 'downer', 'garbage', 'gross',
			  'imperfect', 'inferior', 'junky', 'synthetic', 'abominable', 'amiss', 'bad news', 'beastly', 'careless',
			  'cheesy', 'crappy', 'cruddy', 'defective', 'deficient', 'dissatisfactory', 'erroneous', 'fallacious',
			  'faulty', 'godawful', 'grody', 'grungy', 'icky', 'inadequate', 'incorrect', 'not good', 'off', 'raunchy',
			  'slipshod', 'stinking', 'substandard', 'the pits', 'unsatisfactory'],
		'2': ['bought', 'buy', 'care', 'contains', 'normal', 'works', 'without messing', 'improve', 'pass',
			  'average quality', 'satisfactory', 'OK', 'average', 'customer', 'right', 'true', 'acceptable', 'mediocre',
			  'moderate', 'ordinary', 'regular', 'boilerplate', 'common', 'commonplace', 'fair', 'familiar', 'garden',
			  'general', 'humdrum', 'intermediate', 'mainstream', 'medium', 'middling', 'nowhere', 'plastic',
			  'standard', 'customary', 'dime a dozen', 'everyday', 'fair to middling', 'garden-variety',
			  'middle-of-the-road', 'passable', 'run of the mill', 'so-so', 'tolerable', 'undistinguished',
			  'unexceptional', 'usual'],
		'3': ['never disappointed', 'handy', 'positive result', 'nicely', 'affordable', 'natural', 'gud', 'better',
			  'good', 'pleasing', 'sophisticated', 'positive', 'big', 'improved', 'choice', 'exceeding', 'fit',
			  'preferred', 'sharpened', 'sophisticated', 'surpassing', 'big', 'high quality', 'appropriate',
			  'desirable', 'fitting', 'select', 'suitable', 'useful', 'valuable', 'preferable', 'worthy'],
		'4': ['never disappointed', 'glad', 'worth try', 'really handy', 'happy', 'pleasing', 'blends like butter',
			  'easily', 'enriched', 'enjoyed', 'safe', 'thankyou', 'longlasting', 'quickly', 'gently', 'shiny',
			  'soothing', 'smooth', 'soft', 'works well', 'well', 'moisturizes', 'helps', 'reasonable', 'gentle',
			  'easily ', 'absorb', 'absorbs', 'blends', 'repairs', 'repair', 'save', 'saving', 'decent', 'reduce',
			  'fall stop', 'promote', 'friendly', 'manageable', 'very smooth', 'very light weighted', 'impressed',
			  'travel-friendly', 'liked', 'fresh', 'smooth silky', 'multipurpose', 'live without', 'baby',
			  'smooth nd shiny', 'care completely ', 'favourite', 'must buy', 'really helps', 'really nicely', 'pure',
			  'shine', 'progressively', 'moderately', 'easily absorbs', 'bst', 'professionally', 'smoother',
			  'greatlook', 'proper', 'long lasting', 'very good', 'finer', 'bigger', 'larger', 'higher',
			  'higher quality', 'cool', 'valuable', 'favorable', 'greater', 'satisfying', 'superb', 'wonderful''lucky',
			  'effective', 'beneficial', 'benevolent', 'honest', 'profitable', 'reputable', 'undecayed', 'upright',
			  'worthier', 'super', 'sound', 'fitter', 'more appropriate', 'more desirable', 'more fitting',
			  'more select', 'more suitable', 'more useful', 'more valuable', 'prominent', 'souped-up'],
		'5': ['never disappointed', 'awsm', 'highly recommend', 'must try', 'really amazing',
			  'very very light weighted', 'much smooth', 'really really amazing', 'super impressed', '❤', 'shimmery',
			  'loved', 'Highly recommended', 'worth buying', 'absolute favourite', 'five stars', 'smoothest', 'thumbs',
			  'fabulous', 'wow', 'champion', 'amazing', 'awesome', 'very very good', 'highest quality', 'exceptional',
			  'excellent', 'marvelous', 'incredible', 'biggest', 'love', 'lovely', 'bestest', 'beautiful', 'nicest',
			  'finest', 'first', 'first-rate', 'leading', 'outstanding', 'perfect', 'terrific', '10', 'ace', 'boss',
			  'capital', 'champion', 'culminating', 'nonpareil', 'optimum', 'premium', 'prime', 'primo', 'principal',
			  'super', 'superlative', 'tops', 'A-1', 'beyond compare', 'choicest', 'first-class', 'foremost',
			  'greatest', 'highest', 'incomparable', 'inimitable', 'matchless', 'number 1', 'out-of-sight', 'paramount',
			  'peerless', 'preeminent', 'sans pareil', 'second to none', 'supreme', 'transcendent', 'unequaled',
			  'unparalleled', 'unrivaled', 'unsurpassed']}

	# Initializie score counters for each area
	quality = 0
	operations = 0
	supplychain = 0
	project = 0
	data = 0
	healthcare = 0

	# Create an empty list where the scores will be stored
	scores = []

	# Obtain the scores for each area
	for area in terms.keys():

		if area == '1':
			for word in terms[area]:
				if word in text:
					quality += 1
			scores.append(quality)

		elif area == '2':
			for word in terms[area]:
				if word in text:
					operations += 1
			scores.append(operations)

		elif area == '3':
			for word in terms[area]:
				if word in text:
					supplychain += 1
			scores.append(supplychain)

		elif area == '4':
			for word in terms[area]:
				if word in text:
					project += 1
			scores.append(project)

		elif area == '5':
			for word in terms[area]:
				if word in text:
					data += 1
			scores.append(data)

		else:
			for word in terms[area]:
				if word in text:
					healthcare += 1
			scores.append(healthcare)

	# Create a data frame with the scores summary
	summary = pd.DataFrame(scores, index=terms.keys(), columns=['score']).sort_values(by='score', ascending=False)
	summary['rating'] = summary.index
	summary

	return summary
model = load_model(os.path.abspath('model.h5'))
def predictrating(comment):
	with open(os.path.abspath('tokenizer.pickle'), 'rb') as handle:
		loaded_tokenizer = pickle.load(handle)
	txt = comment
	seq = loaded_tokenizer.texts_to_sequences([txt])
	max_len=50
	padded = pad_sequences(seq, maxlen=max_len)
	model = load_model(os.path.abspath('model.h5'))
	# model.summary()
	Testresults = model.predict(padded)
	df = pd.DataFrame(Testresults, columns=['1', '2', '3', '4', '5'])
	a = df.iloc[0]
	result = a.idxmax()
	print('star rating :', result, a)
	int(result)
	return result



a = predictrating('Average experience')
print(a)