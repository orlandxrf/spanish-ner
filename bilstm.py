from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
from sklearn import metrics
import numpy as np
import dataprocess as dt

def executeBiLSTM(filename, SENTENCE_LENGTH, RANDOM, EMBEDDIND_SIZE, UNITS, BATCH_SIZE, EPOCHS, learnrate, dropout, recdropout, path, imgmodel):
	getter = dt.Sentences(filename)
	sentences = getter.getSpecificLengthSentences(SENTENCE_LENGTH)
	vocabulary_size = len(getter.words)
	tags_size = len(getter.tags)

	# create vector sentences using their indexes instead of words/tags
	X = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=getter.X, padding='post', value=vocabulary_size-1)
	y = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=getter.y, padding='post', value=getter.tag2idx['O'])
	# converts a class vector (integers) to binary class matrix
	y = [to_categorical(i, num_classes=tags_size) for i in y]
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM)

	# hyperparameters and compile from Neural Network
	input = Input(shape=(SENTENCE_LENGTH,))
	model = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDIND_SIZE, input_length=SENTENCE_LENGTH)(input)
	model = Bidirectional(LSTM(units=UNITS, return_sequences=True, dropout=dropout, recurrent_dropout=recdropout))(model)
	out = TimeDistributed(Dense(tags_size, activation="softmax"))(model)
	model = Model(input, out)
	# model.summary()
	# plot_model(model, to_file=path+imgmodel, show_shapes=True, show_layer_names=True) # plot NN diagram
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
	metric1 = metrics.classification_report(testx, predx, labels=labels) # individual tags
	metric2 = classification_report(test_labels, pred_labels) # composed tags

	# Print results
	ln = '-'*100
	print (metric1)
	print (ln)
	print (metric2)

	# Save results
	# testwords = getter.xtest2wordsBiLSTM(X_te) # translate data matrix to words sentences
	# tool = dt.Tools()
	# tool.saveEval(path+'bilstm_res.txt', '{}\n{}\n{}'.format(metric1, metric2, ln), 'a')
	# tool.saveEval(path+'bilstm_eva.txt', '{}\t{}\t{}'.format(testwords, test_labels, pred_labels), 'a')
	# tool.saveEval(path+'bilstm_his.txt', '{}\t{}\t{}\t{}'.format(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']), 'a')

	# Plot accuracy and loss from Model
	# viz = dt.PlotData()
	# viz.plotAccAndLoss(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'], title='Bi-LSTM - Accuracy and Loss ', show=False, prefix='', imgName=path+'bilstm_accuracy_loss', png=True)


path = 'data/'
filename = path+'split1.mx-news.txt'
SENTENCE_LENGTH = 50
RANDOM = 1 # fix split on train and test
EMBEDDIND_SIZE = 300
BATCH_SIZE = 50
EPOCHS = 20
UNITS = 100
learnrate = 0.001
dropout = 0.01
recdropout = 0.3
imgmodel = 'bilstm-model.png'

executeBiLSTM(filename, SENTENCE_LENGTH, RANDOM, EMBEDDIND_SIZE, UNITS, BATCH_SIZE, EPOCHS, learnrate, dropout, recdropout, path, imgmodel)

