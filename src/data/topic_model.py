import ipdb
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, LdaMulticore

#nltk.download('stopwords')
from nltk.corpus import stopwords

class TopicModel:
	def __init__(self, n_topics=3):
		self.n_topics = n_topics
		self.punctuation = "!\"#$%&()*+,./:;<=>?@[\\]^`{|}~"
		self.punct_table = str.maketrans(dict.fromkeys(self.punctuation))

	def sent_to_words(self, sentences):
		for sentence in sentences:
			# deacc=True removes punctuations
			yield(simple_preprocess(str(sentence), deacc=True))

	def remove_stopwords(self, texts):
		stop_words = stopwords.words('english')
		stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'would'])
		return [[word for word in simple_preprocess(str(doc)) 
				 if word not in stop_words] for doc in texts]

	def find_topics(self, text_df):
		"""
		Input:
			text_df: dataframe of text, each row represents the textual content of a response
		Output:
			topics: dictionary of topics (key: topic_id, value: word distribution)
		"""
		text_df = text_df.map(lambda x: x.translate(self.punct_table))
		text_df = text_df.map(lambda x: x.lower())

		text = text_df.tolist()
		text_words = list(self.sent_to_words(text))
		text_words = self.remove_stopwords(text_words)

		id2word = corpora.Dictionary(text_words)
		corpus = [id2word.doc2bow(text) for text in text_words]
		corpus_size = len(id2word)

		## LDA
		topics = {}
		lda_model = LdaModel(
			corpus=corpus, 
			id2word=id2word, 
			num_topics=self.n_topics, 
			dtype=np.float64
		) #LdaMulticore(corpus=corpus, id2word=id2word, num_topics=self.n_topics)
		for topic_id, word_dist in lda_model.show_topics(num_words=corpus_size, formatted=False):
			#print("Topic ID: {}, {}".format(topic_id, word_dist))
			topics[topic_id] = word_dist

		return topics

def build_topics(args):
	"""
	topics = []
	for line in open("test.json"):
		topics.append(json.loads(line.strip().rstrip()))
	ipdb.set_trace()
	"""

	## Load dataset
	data_df = pd.read_csv("{}/{}/data.csv".format(args.dataset_root, args.dataset_name))

	## Build topic model
	topic_model = TopicModel(n_topics=args.n_topics)

	## For each thread
	topics_threads = {}
	fout = open("{}/{}/topics.json".format(args.dataset_root, args.dataset_name), "w")
	for src_id, group in tqdm(data_df.groupby("source_id")):
		topics = topic_model.find_topics(group["text"])
		topics_threads[src_id] = topics
	fout.write(json.dumps(topics_threads))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Rumor Detection")
	parser.add_argument("--dataset_name", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--n_topics", type=int, default=3)
	args = parser.parse_args()

	build_topics(args)

"""
import re
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## Others
	parser.add_argument("--dataset", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4,comp", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	args = parser.parse_args()

	return args

def sent_to_words(sentences):
	for sentence in sentences:
		# deacc=True removes punctuations
		yield(simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
	stop_words = stopwords.words('english')
	stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'would'])
	return [[word for word in simple_preprocess(str(doc)) 
			 if word not in stop_words] for doc in texts]

def main(args):
	data_path = "{}/{}/data.csv".format(args.data_root, args.dataset)
	data_df = pd.read_csv(data_path)

	## Get source tweet content
	src_df = data_df[data_df["source_id"] == data_df["tweet_id"]]
	src_df = src_df.drop(columns=["tweet_id", "parent_idx", "self_idx", "num_parent", "max_seq_len", "veracity", "stance"])

	## Preprocess text
	src_df["text_processed"] = src_df["text"].map(lambda x: re.sub("[,\\.!?]", "", x))
	src_df["text_processed"] = src_df["text_processed"].map(lambda x: x.lower().replace("<end>", "").replace("url", ""))

	## Prepare data for LDA analysis
	data = src_df["text_processed"].tolist()
	data_words = list(sent_to_words(data))
	data_words = remove_stopwords(data_words)

	id2word = corpora.Dictionary(data_words)
	texts = data_words
	corpus = [id2word.doc2bow(text) for text in texts]

	## LDA
	num_topics = 8
	lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
	for topic in lda_model.print_topics():
		print(topic)

	## Visualization
	#pyLDAvis.enable_notebook()
	LDAvis_data_filepath = "{}/{}/lda/ldavis_prepared_{}".format(args.data_root, args.dataset, num_topics)
	if True:
		LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
		with open(LDAvis_data_filepath, 'wb') as f:
			pickle.dump(LDAvis_prepared, f)

	with open(LDAvis_data_filepath, 'rb') as f:
		LDAvis_prepared = pickle.load(f)

	pyLDAvis.save_html(LDAvis_prepared, "{}/{}/lda/ldavis_prepared_{}.html".format(args.data_root, args.dataset, num_topics))

if __name__ == "__main__":
	args = parse_args()
	main(args)
"""