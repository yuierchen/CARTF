#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pickle
import  codecs
import sys
sys.path.append("..")
path='../data/api_questions_pickle_new'
titles=list()
with codecs.open(path,'rb') as f:
    data=pickle.load(f)
    # for i in data:
    for question in data:
        title=question.title
        titles.append(title)
pickle.dump(titles,open('../data/titles.pkl','wb'))


