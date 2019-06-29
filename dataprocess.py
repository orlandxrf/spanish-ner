class Sentences(object):
	"""The class provides functions to get sentences and features from words inside the sentence"""

	def __init__(self, filename):
		self.filename = filename # corpus filename
		self.words = [] # vocabulary
		self.tags = [] # tags (classes)
		self.word2idx = {} # dict of indexes of words
		self.tag2idx = {} # dict of indexes of tags
		self.X = [] # sentences represented by the words index
		self.y = [] # sentences represented by the tags index

	def getSentences(self):
		"""To obtain sentences"""
		tmp_sent = []
		sentences = []
		with open(self.filename, 'r') as f:
			for i, line in enumerate(f):
				if i == 0: continue # skip the header of file
				line = line.replace('\n','').split('\t')[1:]
				if len(line) == 0: # end to sentence
					sentences.append( tmp_sent )
					tmp_sent = []
				else: # words of the sentence
					if line[0] not in self.words: self.words.append(line[0]) # get all words (tokens)
					if line[2] not in self.tags: self.tags.append(line[2]) # get all tags
					tmp_sent.append(tuple(line))

			sentences.append( tmp_sent ) # add last sentence
		f.close()
		self.tags = sorted(self.tags, key=lambda x:x[1:] )
		return sentences

	def getSpecificLengthSentences(self, maxlength):
		"""Obtener una lista de lista de oraciones a partir de un corpus.
		max_length: longitud de palabras que contendra cada oración como máximo.
		"""
		tmp_sent = []
		sentences = []
		with open(self.filename, 'r') as f:
			for i, line in enumerate(f):
				if i == 0: continue # encabezado del archivo (omitir)
				line = line.replace('\n','').split('\t')[1:]
				if len(line) == 0: # separador de oraciones
					if len(tmp_sent) > maxlength:
						nlength = len(tmp_sent) // maxlength
						resto = len(tmp_sent) % maxlength
						for j in range(nlength):
							sentences.append( tmp_sent[ j*maxlength : maxlength+(j*maxlength) ] )
						if resto > 0:
							sentences.append( tmp_sent[-resto:] )
					else:
						sentences.append( tmp_sent )
					tmp_sent = []
				else:
					if line[0] not in self.words: self.words.append(line[0]) # get all words (tokens)
					if line[2] not in self.tags: self.tags.append(line[2]) # get all tags
					tmp_sent.append( tuple(line) )
		f.close()
		self.words.append('<pad>') # this token is used to fill sentences, if it's necessary
		self.tags.remove('O') # remove temporary to sort the others tags
		self.tags = sorted(self.tags, key=lambda x: x.split('-')[1])
		self.tags.append('O') # insert class 'O' other
		self.__getWordIndexes()
		self.__sentencesToIndexes(sentences)
		return sentences

	def __getWordIndexes(self):
		"""Obtiene el índice de palabras en un diccionario a partir de una vocabulario (lista de palabras)"""
		self.word2idx = {w: i for i, w in enumerate(self.words)}
		self.tag2idx = {w: i for i, w in enumerate(self.tags)}

	def __sentencesToIndexes(self, sentences):
		self.X = [[self.word2idx[w[0]] for w in s] for s in sentences]
		self.y = [[self.tag2idx[w[2]] for w in s] for s in sentences]

	def word2features(self, sent, i):
		"""
		Extract features from words.
		source: https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features
		"""
		word = sent[i][0]
		postag = sent[i][1]

		features = {
		    'bias': 1.0,
		    'word.lower()': word.lower(),
		    'word[-3:]': word[-3:],
		    'word.isupper()': word.isupper(),
		    'word.istitle()': word.istitle(),
		    'word.isdigit()': word.isdigit(),
		    'postag': postag,
		    'postag[:2]': postag[:2],
		}
		if i > 0:
		    word1 = sent[i-1][0]
		    postag1 = sent[i-1][1]
		    features.update({
		        '-1:word.lower()': word1.lower(),
		        '-1:word.istitle()': word1.istitle(),
		        '-1:word.isupper()': word1.isupper(),
		        '-1:postag': postag1,
		        '-1:postag[:2]': postag1[:2],
		    })
		else:
		    features['BOS'] = True

		if i < len(sent)-1:
		    word1 = sent[i+1][0]
		    postag1 = sent[i+1][1]
		    features.update({
		        '+1:word.lower()': word1.lower(),
		        '+1:word.istitle()': word1.istitle(),
		        '+1:word.isupper()': word1.isupper(),
		        '+1:postag': postag1,
		        '+1:postag[:2]': postag1[:2],
		    })
		else:
		    features['EOS'] = True

		return features

	def sent2features(self, sent):
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self, sent):
		return [label for token, postag, label in sent]

	def sent2tokens(self, sent):
		return [token for token, postag, label in sent]

	def verifyTestDataBalanceCRF(self, y_test):
		"""check tags (classes) balance from dataset in test"""
		from seqeval.metrics.sequence_labeling import get_entities
		lst = [ls for sublist in y_test for ls in sublist]
		tags = set( [tg[0] for tg in get_entities(lst)] )
		tags = list(tags)
		tags.sort()
		print ('{}\t{}'.format(len(tags), tags))

	def verifyTestDataBalanceBiLSTM(self, y_te):
		"""check tags (classes) balance from dataset in test"""
		from seqeval.metrics.sequence_labeling import get_entities
		import numpy as np
		idx2tag = {i: w for w, i in self.tag2idx.items()}
		def pred2label(pred):
			out = []
			for pred_i in pred:
				out_i = []
				for p in pred_i:
					p_i = np.argmax(p)
					out_i.append(idx2tag[p_i].replace('<pad>', 'O'))
				out.append(out_i)
			return out
		test_labels = pred2label(y_te)
		lst = [ls for sublist in test_labels for ls in sublist]
		tags = set( [tg[0] for tg in get_entities(lst)] )
		tags = list(tags)
		tags.sort()
		print ('{}\t{}'.format(len(tags), tags))

	def xtest2wordsCRF(self, X_test):
		"""Transform data test to word sentences"""
		sentences = []
		for feat in X_test:
			tmplst = []
			for tok in feat:
				word = ''
				if tok['word.isupper()']: word = tok['word.lower()'].upper()
				elif tok['word.istitle()']: word = tok['word.lower()'].title()
				else: word = tok['word.lower()']
				tmplst.append(word)
			sentences.append(tmplst)
		return sentences

	def xtest2wordsBiLSTM(self, X_te):
		"""Transform data test to word sentences"""
		sentences = []
		for snts in X_te:
			sentences.append([self.words[word] if self.words[word]!='<pad>' else 'O' for word in snts])
		return sentences

class ProcessELMo(object):
	"""Clase para crear embeddings con ELMo en español y usar estos embeddings para Redes en keras"""

	def trainELMoToFile(self, elmoModelPath, filename, words, vocab_size):
		"""Build ELMo word embeddings

		filename: filename to save word embeddings
		words: word list (vocabulary)
		vocab_size: length of word list (vocabulary)

		visit to https://github.com/HIT-SCIR/ELMoForManyLangs to download Spanish model and set up some stuffs.
		"""
		from elmoformanylangs import Embedder
		import numpy as np
		e = Embedder(elmoModelPath) # path from loaded model (e.g. /home/user/project/ELMo.es/)
		embedding_matrix = np.zeros((vocab_size, 1024))

		for i, word in enumerate(words):
			aux_elmo = e.sents2elmo( [[word]] )
			with open(filename, 'a') as g:
				strnums = [str(num) for num in aux_elmo[0][0].tolist()]
				strnums = ' '.join(strnums)
				g.write( "{} {}\n".format(word, strnums ) )
				# print ("Processing \t{} of {}...".format( i+1, len(words)) )
			g.close()


	def loadELMoFromFile(self, filename, vocab_size, emb_length=1024, vb=True):
		"""Load ELMo model and return the embeddings matrix

		filename: ELMo model filename
		vocab_size: vocabulary length
		emb_length: embeddings length (default 1024)
		vb: show loading of vectors
		"""
		import sys
		import numpy as np
		print ("")
		embedding_matrix = np.zeros((vocab_size, emb_length))
		with open(filename, 'r') as f:
			for i, line in enumerate(f):
				values = line.split()
				coefs = np.asarray(values[1:], dtype='float32')
				embedding_matrix[i] = coefs
				if vb:
					sys.stdout.write( "\tLoadings {} vectors of {} ...\r".format(i+1, vocab_size) )
					sys.stdout.flush()
			f.close()
		print ('-'*80, "\n\tELMo matrix embeddings {} has been loaded.\n".format(embedding_matrix.shape), '-'*80)
		return embedding_matrix

class ConfusionMatrix(object):
	"""Class to build confusion matrix"""

	def __init__(self):
		self.tags = {}
		self.diagonal = {}
		self.__matrix = None

	def getTags(self, y_data, tags={}):
		"""Receive a list of test and predicted data and return a dict of tags inside of data"""
		countTag = 0
		for ls in y_data:
			if isinstance(ls, list): # list contains lists of classes
				for tg in ls:
					if tg != 'O': # when class is different to 'O' (other)
						tg = tg.split('-')[1]
						if tg not in tags:
							tags[tg] = countTag
							countTag += 1
					else: # when is the class 'O' (other)
						if tg not in tags:
							tags[tg] = countTag
							countTag += 1
			else: # lists of classes
				if ls != 'O': # when class is different to 'O' (other)
					ls = ls.split('-')[1]
					if ls not in tags:
						tags[ls] = countTag
						countTag += 1
				else: # when is the class 'O' (other)
					if ls not in tags:
						tags[ls] = countTag
						countTag += 1
		return tags

	def percentagesOnDiagonal(self, matrix):
		diagonal = {}
		for i in range(len(matrix)):
			if sum(matrix[i][:]) == 0: total = 0
			else:
				total = ( float(matrix[i][i]) / float(sum(matrix[i][:])) ) * 100
				totalInt = int('{:.0f}'.format(total))
				if str(total).split('.')[1] == '0': totalFlt = int(total)
				else: totalFlt = '{:.1f}'.format(total)
				matrix[i][i] = totalInt
				for tg in self.tags:
					if tags[tg] == i:
						diagonal[tg] = totalFlt
		return (diagonal)


	def multilabelConfusionMatrix(self, y_test, y_pred, percentage=False):
		"""Build confusion matrix
		y_true: test dataset
		y_pred: predicted dataset
		"""
		import numpy as np
		tags = {}
		tags = self.getTags(y_test, tags) # get tags from y_test
		tags = self.getTags(y_pred, tags) # get tags from y_test and y_pred
		tags = sorted(tags.items(), key=lambda x:x[0], reverse=True) # descendent sort
		tags = {t[0]:i for i, t in enumerate(tags)} # change and set new index
		self.tags = tags.copy()
		matrix = [[0]*len(tags) for i in range(len(tags))] # build confusion matrix with zeros
		for k, ls in enumerate(y_test):
			if isinstance(ls, list): # list contains lists of classes
				for i in range(len(ls)):
					# compare y_test Vs y_pred
					if y_test[k][i] == 'O' or y_pred[k][i] == 'O':
						if y_test[k][i] != 'O': tg_true = y_test[k][i].split('-')[1]
						else: tg_true = y_test[k][i]
						if y_pred[k][i] != 'O': tg_pred = y_pred[k][i].split('-')[1]
						else: tg_pred = y_pred[k][i]
						matrix[tags[ tg_true ]][tags[ tg_pred ]] += 1
					else:
						tg_true = y_test[k][i].split('-')[1]
						tg_pred = y_pred[k][i].split('-')[1]
						matrix[ tags[tg_true] ][ tags[tg_pred] ] += 1
			else: # lists of classes
				# compare y_test Vs y_pred
				if y_test[k] == 'O' or y_pred[k] == 'O':
					matrix[tags[y_test[k]]][tags[y_pred[k]]] += 1
				else:
					y_test[k] = y_test[k].split('-')[1]
					y_pred[k] = y_pred[k].split('-')[1]
					matrix[tags[y_test[k]]][tags[y_pred[k]]] += 1

		if percentage:
			for i in range(len(matrix)):
				if sum(matrix[i][:]) == 0: total = 0
				else:
					total = ( float(matrix[i][i]) / float(sum(matrix[i][:])) ) * 100
					totalInt = int('{:.0f}'.format(total))
					if str(total).split('.')[1] == '0': totalFlt = int(total)
					else: totalFlt = '{:.1f}'.format(total)
					matrix[i][i] = totalInt
					for tg in tags:
						if tags[tg] == i:
							self.diagonal[tg] = totalFlt
		return matrix

class PlotData(object):
	"""Class to plot data using confusion matrix, bars and lines charts"""

	def plot_confusion_matrix(self, cm, classes, normalize=False, fontsize=13, title='Confusion matrix', show=True, imgName='cm', png=False, pdf=False, wi=11, he=9, diagonal={}):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		import numpy as np
		import matplotlib.pyplot as plt
		import matplotlib.colors as colors
		import itertools
		# cmap = plt.cm.Blues
		# cmap = plt.cm.Greens
		color_list = []
		for i in range(10):
			i += 1
			color_list.append((1.0, 1.0, 1.0))
		color_map = colors.ListedColormap(list(reversed(color_list)))
		plt.figure(figsize=(wi,he))
		plt.imshow(cm, interpolation='nearest', cmap=color_map)
		# plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		# plt.colorbar()

		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, fontsize=fontsize) # , rotation=45
		plt.yticks(tick_marks, classes, fontsize=fontsize)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		thresh = cm.max() / 2.

		nfontsize = fontsize
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if i==j:
				mytext = '{}\n{}%'.format(cm[i, j], diagonal[i])
				fntsze = fontsize
			else:
				mytext = str(int(cm[i, j]))
				fntsze = fontsize
			plt.text(j, i, mytext,
						horizontalalignment="center",
						color="darkblue" if i==j else ("red" if int(mytext) > 0 else "black"),
						fontsize=fntsze,
						weight = 549
						# fontsize=fontsize
					)

		plt.tight_layout()
		plt.ylabel('Clases Verdaderas', fontsize=fontsize)
		plt.xlabel('Clases Predecidas', fontsize=fontsize)
		if png: plt.savefig(imgName+'.png', bbox_inches='tight')
		if pdf: plt.savefig(imgName+'.pdf', bbox_inches='tight')
		if show: plt.show()

	def plotXMetric(self, result1, result2, metrics, title1, title2, supertitle='', metric='', imgName='image', show=True, png=False, eps=False):
		import matplotlib.pyplot as plt
		indexes = list(range(0, len(result1[0][0])))
		# labels = [str(i+1)+'\nClass' if i==0 else str(i+1)+'\nClasses' for i in indexes]
		labels = [str(i+1) for i in indexes]
		ylim = 1.05

		markers = ['o','s','X','*']
		# colors = ['k','k','k','k']
		colors = ['red','darkgreen','m','dodgerblue']

		linestyles = [':', '-.', '--', '-']

		# dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)] # para tener multiples linestyle

		plt.figure(figsize=(15,8))
		plt.subplot(2, 2, 1)
		# plt.ylim(0, ylim)
		for i, arr in enumerate(result1[0]): plt.plot(arr, label=metrics[i], color=colors[i], linestyle=linestyles[i])
		plt.title('Evaluación$_i$ ' + title1, fontsize='16')
		plt.xticks(indexes, labels)
		plt.ylabel(metric+'$_i$', fontsize='14')
		# plt.xlabel('classes')
		plt.xlabel('clases', fontsize='14')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplot(2, 2, 2)
		# plt.ylim(0, ylim)
		for i, arr in enumerate(result1[1]): plt.plot(arr, label=metrics[i], color=colors[i], linestyle=linestyles[i])
		plt.title('Evaluación$_c$ ' + title1, fontsize='16')
		plt.xticks(indexes, labels)
		plt.ylabel(metric+'$_c$', fontsize='14')
		# plt.xlabel('classes')
		plt.xlabel('clases', fontsize='14')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplot(2, 2, 3)
		# plt.ylim(0, ylim)
		for i, arr in enumerate(result2[0]): plt.plot(arr, label=metrics[i], color=colors[i], linestyle=linestyles[i])
		plt.title('Evaluación$_i$ ' + title2, fontsize='16')
		plt.xticks(indexes, labels)
		plt.ylabel(metric+'$_i$', fontsize='14')
		# plt.xlabel('classes')
		plt.xlabel('clases', fontsize='14')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplot(2, 2, 4)
		# plt.ylim(0, ylim)
		for i, arr in enumerate(result2[1]): plt.plot(arr, label=metrics[i], color=colors[i], linestyle=linestyles[i])
		plt.title('Evaluación$_c$ ' + title2, fontsize='16')
		plt.xticks(indexes, labels)
		plt.ylabel(metric+'$_c$', fontsize='14')
		# plt.xlabel('classes')
		plt.xlabel('clases', fontsize='14')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplots_adjust(top=0.91)
		# plt.suptitle(supertitle) # or fig.suptitle('Main title')
		# plt.savefig('CRF_less2many.png', bbox_inches='tight')
		# plt.show()

		if png: plt.savefig('{}.png'.format(imgName), bbox_inches='tight', dpi=200)
		if eps: plt.savefig('{}.eps'.format(imgName), bbox_inches='tight', dpi=200)
		if show: plt.show()

	def plotAccAndLoss(self, trainA, validationA, trainL, validationL, title='Título', show=True, prefix='no-prefix_', imgName='plotmultiplefunction', png=False, pdf=False):
		"""Plot the accuracy and loss of the train and validation"""
		import matplotlib.pyplot as plt

		markers = ['^', 'o', 's']
		plt.figure(figsize=(18,8))
		plt.subplot(1, 2, 1)
		tl = plt.plot(trainA, marker='o', color='blue', label='train')
		vl = plt.plot(validationA, marker='s', color='orange', label='validation')
		plt.title('Model train vs validation loss')
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.legend([tl[0], vl[0]], ['train', 'validation'], loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplot(1, 2, 2)
		tl = plt.plot(trainL, marker='o', color='blue', label='train')
		vl = plt.plot(validationL, marker='s', color='orange', label='validation')
		plt.title('Model train vs validation loss')
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.legend([tl[0], vl[0]], ['train', 'validation'], loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplots_adjust(top=0.91)
		plt.suptitle(title) # or fig.suptitle('Main title')
		if png: plt.savefig('{}.png'.format(prefix+imgName))
		if pdf: plt.savefig('{}.pdf'.format(prefix+imgName))
		if show: plt.show()




class Tools(object):
	"""class with some useful functions"""

	def saveEval(self, filename, data, mode='w'):
		"""To save results in file, later it will be use to analyze results"""
		g = open(filename, mode)
		g.write('{}\n'.format(data))
		g.close()


