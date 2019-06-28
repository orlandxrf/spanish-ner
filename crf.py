from sklearn_crfsuite import CRF, metrics # CRF model and metrics for individual tags evaluation
from seqeval.metrics import classification_report # for the composed tags evaluation
from sklearn.model_selection import train_test_split
import dataprocess as dt # (dataprocess.py) library with useful functions

def executeCRF(filename, RANDOM, path):
	"""Execute CRF model"""

	# Prepare data
	getter = dt.Sentences(filename) # instance of the class Sentences
	tool = dt.Tools() # instance of the class Tools
	sentences = getter.getSentences()
	X = [getter.sent2features(s) for s in sentences] # convert the tuples list into a features dictionary for each word of the sentence.
	y = [getter.sent2labels(s) for s in sentences] # create a list that represents the tags of the sentence
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM) # split train an test

	# Training the CRF model
	crf = CRF(
		algorithm='lbfgs', # Gradient descent using the L-BFGS method
		c1 = 0.1,
		c2 = 0.1,
		max_iterations = 50,
		all_possible_transitions = True,
		# verbose = True,
	)
	model = crf.fit(X_train, y_train)
	labels = list(model.classes_) # get classes
	labels.remove('O') # remove class O (other)
	n_labels = sorted(labels, key=lambda x: x.split("-")[1]) # sort labels

	# Evaluation
	y_pred = model.predict(X_test)
	metric1 = metrics.flat_classification_report(y_test, y_pred, labels=n_labels) # individual tags evaluation
	metric2 = classification_report(y_test, y_pred) # composed tags evaluation

	# Print results
	ln = '-'*100
	print (metric1)
	print (ln)
	print (metric2)

	# Save results in file
	# testwords = getter.xtest2wordsCRF(X_test) # translate data in test to words sentences
	# tool.saveEval(path+'crf_res.txt', '{}\n{}\n{}\n'.format(metric1, metric2, ln), 'a')
	# tool.saveEval(path+'crf_eva.txt', '{}\t{}\t{}'.format(testwords, y_test, y_pred), 'a')



path = 'data/'
filename = path+'split1.mx-news.txt'
RANDOM = 1 # fix random to split train & test
executeCRF(filename, RANDOM, path)
