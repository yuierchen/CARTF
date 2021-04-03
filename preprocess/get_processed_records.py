#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import logging
import sys
import pickle
sys.path.append("..")
import numpy as np
import pandas as pd
from scipy.io import mmread
import tensorly as tl
import tensorflow as tf
import os
from jpype import *
import os.path
import gensim
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from algorithm import similarity
from jpype import *
import Levenshtein
import json
import csv
import util

csvfile = open("../data/records.csv", 'w', newline="",encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(["method_annotation", "method_api_sequence"])
w2v = gensim.models.Word2Vec.load('../data/skip_w2v_model_stemmed')  # pre-trained word embedding
idf = pickle.load(open('../data/my_idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary
records = pickle.load(open("../data/records3.pkl", 'rb'))
records = util.preprocess_all_records_new(records, idf, w2v)
for record in records:
    writer.writerow([record.title,record.method_api_sequence])
with open("../data/records_final3.pkl", "wb") as f:
    pickle.dump(records, f, 4)
logging.info("has finished records.pkl")
csvfile.close()