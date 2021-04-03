import fileinput
import gensim
import nltk
import sys
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import *
import pickle
from nltk.stem import SnowballStemmer
import time
import multiprocessing


time_start=time.time()
answer_corpora = pickle.load(open('../data/titles.pkl','rb'))


print('start word segmentation')

texts = [WordPunctTokenizer().tokenize(answer.lower()) for answer in answer_corpora]

print('start stemming')

stemmer = SnowballStemmer('english')

for i,text in enumerate(texts):
    texts[i] = [SnowballStemmer('english').stem(word) for word in text]


print('start training w2v')

model = gensim.models.Word2Vec(texts, sg=1,size=100, window=5,min_count=5, workers=multiprocessing.cpu_count())

time_end=time.time()
print('totally cost',time_end-time_start)
model.save('../data/skip_w2v_model_stemmed')


#
#
# sys.setdefaultencoding('utf8')
#
# sentences = []
# stemmer = PorterStemmer()
# cc = 0
# for line in fileinput.input("data/answercorpora"):
#
#     line = line.decode('utf-8', 'ignore')
#     line = line.strip()
#     if line == "":
#         continue
#     line = line.lower()
#     sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     sens = sent_tokenizer.tokenize(line)
#
#     for sent in sens:
#         words = WordPunctTokenizer().tokenize(sent)
#         words = [stemmer.stem(word) for word in words]
#         sentences.append(words)
#
#     cc = cc+1
#     if cc%100000==0:
#         print cc
#     #if cc>10000:
#         #break
#
# # sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
#
# print "read sentences over"
#
# model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
# model.save('data/w2v_model_stemmed')
#

