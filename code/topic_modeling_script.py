from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import datetime
import os
import re
import nltk
import time


path = '/Users/guillembp/Dropbox/Text Mining/omiros/assignment'
os.chdir(path)

from sklearn.datasets import fetch_20newsgroups

def display_topics_lda(model, feature_names, no_top_words):
	'''Outputs topic to console and file'''
	with open('LDA_output.txt', 'w') as f:
		for topic_idx, topic in enumerate(model.components_):
			print("Topic %d:" % (topic_idx), file = f)
			print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]), file = f)

	for topic_idx, topic in enumerate(model.components_):
		print("Topic %d:" % (topic_idx))
		print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 	1:-1]]))

def display_topics_nmf(model, feature_names, no_top_words):
	'''Outputs topic to console and file'''
	with open('NMF_output.txt', 'w') as f:
		for topic_idx, topic in enumerate(model.components_):
			print("Topic %d:" % (topic_idx), file = f)
			print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]), file = f)

	for topic_idx, topic in enumerate(model.components_):
		print("Topic %d:" % (topic_idx))
		print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 	1:-1]]))

def clean_words(s):
	'''omits unwanted words and encoding errors'''
	s = s.lower() # homogenize letter case
	s = s.replace('\n', ' ').replace('\r', '') #remove line breaks
	emoji_pattern = re.compile(
		u"(\ud83d[\ude00-\ude4f])|"  # emoticons
		u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
		u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
		u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
		u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
		"+", flags=re.UNICODE)
	s = re.sub("\S*\d\S*", "", s).strip() # remove mistakenly encoded characters
	return emoji_pattern.sub(r'', s) # no emoji

# def removeStopWords(li):
# 	return [l for l in li if l not in ENGLISH_STOP_WORDS]
#
# def stemmed_words(doc):
# 	stemmer = EnglishStemmer()
# 	analyzer = CountVectorizer().build_analyzer()
# 	return (stemmer.stem(w) for w in analyzer(doc))


dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

type(dataset)

documents = dataset.data

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, max_df=0.95, max_features=no_features, lowercase = True, decode_error='ignore', preprocessor = clean_words)

tfidf_vectorizer

tfidf = tfidf_vectorizer.fit_transform(documents)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_df=0.95, max_features=no_features, lowercase = True, decode_error='ignore', preprocessor = clean_words)

tf = tf_vectorizer.fit_transform(documents)

tf_feature_names = tf_vectorizer.get_feature_names()

component_nmbr = 14

####### Non-negative Matrix Factorization
start = time.time()
nmf = NMF(n_components=component_nmbr, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
end = time.time()
print("NMF took:", end - start)

####### Latent Dirichlet Allocation
start = time.time()
lda = LatentDirichletAllocation(n_components=component_nmbr, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
end = time.time()
print("LDA took:", end - start)


no_top_words = 10

####### Run
display_topics_nmf(nmf, tfidf_feature_names, no_top_words)
display_topics_lda(lda, tf_feature_names, no_top_words)
