
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from seqeval.metrics import classification_report
import dataprocess as dt
import numpy as np

def executeBiLSTMELMo(embedding_matrix, filename, SENTENCE_LENGTH, RANDOM, EMBEDDIND_SIZE, UNITS, BATCH_SIZE, EPOCHS, learnrate, dropout, recdropout, path, imgmodel):
	getter = dt.Sentences(filename)
	sentences = getter.getSpecificLengthSentences(SENTENCE_LENGTH)
	vocabulary_size = len(getter.words)
	tags_size = len(getter.tags)

	# create vector sentences using their indexes instead of words / tags
	X = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=getter.X, padding='post', value=vocabulary_size-1)
	y = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=getter.y, padding='post', value=getter.tag2idx['O'])
	# converts a class vector (integers) to binary class matrix
	y = [to_categorical(i, num_classes=tags_size) for i in y]
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM)

	# hyperparameters from Neural Network
	input = Input(shape=(SENTENCE_LENGTH,))
	model = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDIND_SIZE, input_length=SENTENCE_LENGTH, weights=[embedding_matrix], trainable=True)(input)
	model = Bidirectional(LSTM(units=UNITS, return_sequences=True, dropout=dropout, recurrent_dropout=recdropout))(model)
	out = TimeDistributed(Dense(tags_size, activation="softmax"))(model)
	model = Model(input, out)
	# model.summary()
	# plot_model(model, to_file=path+imgmodel, show_shapes=True, show_layer_names=True)
	opt = RMSprop(lr=learnrate, decay=0.0)
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
	history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle=True, verbose=2)

	# predicted data
	test_pred = model.predict(X_te, verbose=0)
	idx2tag = {i: w for w, i in getter.tag2idx.items()}
	def pred2label(pred):
		out = []
		for pred_i in pred:
			out_i = []
			for p in pred_i:
				p_i = np.argmax(p)
				out_i.append(idx2tag[p_i].replace('<pad>', 'O'))
			out.append(out_i)
		return out

	pred_labels = pred2label(test_pred)
	test_labels = pred2label(y_te)

	# Prepare results
	testx, predx = [], []
	for k in range(len(test_labels)): testx += test_labels[k]
	for k in range(len(pred_labels)): predx += pred_labels[k]
	labels = getter.tags.copy()
	labels.remove('O') # remove class 'O'
	labels = sorted(labels, key=lambda x: x.split('-')[1]) # sort labels
	metric1 = metrics.classification_report(testx, predx, labels=labels)
	metric2 = classification_report(test_labels, pred_labels)

	# Print results
	ln = '-'*100
	print (metric1)
	print (ln)
	print (metric2)

	# Save results
	testwords = getter.xtest2wordsBiLSTM(X_te)
	tool = dt.Tools()
	tool.saveEval(path+'bilstm-elmo_res.txt', '{}\n{}\n{}'.format(metric1, metric2, ln), 'a')
	tool.saveEval(path+'bilstm-elmo_eva.txt', '{}\t{}\t{}'.format(testwords, test_labels, pred_labels), 'a')
	tool.saveEval(path+'bilstm-elmo_his.txt', '{}\t{}\t{}\t{}'.format(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']), 'a')

	# Plot results: accuracy and loss
	viz = dt.PlotData()
	viz.plotAccAndLoss(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'], title='Bi-LSTM - Accuracy and Loss ', show=True, prefix='', imgName=path+'bilstm-elmo_accuracy_loss', png=True)

dict_vocsize = { # vocabularies length from ELMo models
	'test-a.conll-2002.elmo.txt': 9647,
	'test-b.conll-2002.elmo.txt': 9087,
	'train.conll-2002.elmo.txt': 26100,
	'ensemble.conll-2002.elmo.txt': 31406,
	'split1.mx-news.elmo.txt': 7629,
	'split2.mx-news.elmo.txt': 7727,
	'split3.mx-news.elmo.txt': 7665,
	'ensemble.mx-news.elmo.txt': 13294,
	'mx1_elmo.txt': 7629,
}


path = 'data/'
elmopath = 'path_of_elmo_models_pre-trained' # download and unzip
filename = 'split1.mx-news.txt'
SENTENCE_LENGTH = 50
RANDOM = 1
EMBEDDIND_SIZE = 1024
BATCH_SIZE = 50
EPOCHS = 20
UNITS = 200
learnrate = 0.001
dropout = 0.01
recdropout = 0.3
imgmodel = 'bilstm-elmo_model.png'

# load ELMo modelo from file
elmofile = '{}{}'.format(elmopath, filename)
we = dt.ProcessELMo()
embedding_matrix = we.loadELMoFromFile(elmofile, vocab_size=dict_vocsize[filename], vb=False) # load elmo model from file

executeBiLSTMELMo(embedding_matrix, path+filename, SENTENCE_LENGTH, RANDOM, EMBEDDIND_SIZE, UNITS, BATCH_SIZE, EPOCHS, learnrate, dropout, recdropout, path, imgmodel)

