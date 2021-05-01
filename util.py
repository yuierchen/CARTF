#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import logging
import sys
import pickle
sys.path.append("..")

import os
import os.path
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import codecs
import domain
from algorithm import similarity
from jpype import *
import json


def preprocess_all_records(records, idf, w2v):
    processed_records = list()
    jarpath = os.path.join(os.path.abspath('.'), "D:/")
    startJVM("E:/Java/jdk1.8.0_181/jre/bin/server/jvm.dll", "-ea",
             "-Djava.class.path=%s" % (jarpath + "StackOverFlowFilter.jar"))
    JDClass = JClass("cn.edu.nuaa.xin.match.parser.CodeParser")
    JFClass=JClass("cn.edu.cc.traverseblock.GetFlattenTree")
    for record in records:
        # class_description_words = WordPunctTokenizer().tokenize(record.class_description.lower())
        # method_annotation_words = WordPunctTokenizer().tokenize(record.method_annotation.lower())
        fullmethod_annotation_words = WordPunctTokenizer().tokenize(record.fullmethod_annotation.lower())
        # class_description_words = [SnowballStemmer('english').stem(word) for word in class_description_words]
        # method_annotation_words = [SnowballStemmer('english').stem(word) for word in method_annotation_words]
        fullmethod_annotation_words = [SnowballStemmer('english').stem(word) for word in fullmethod_annotation_words]
        # record.class_description_words = class_description_words
        # record.method_annotation_words = method_annotation_words
        record.fullmethod_annotation_words = fullmethod_annotation_words
        jd = JDClass()
        jf=JFClass()
        jd.parseAPI(record.method_block)
        apiSequence=list()
        for i in jd.apiSequence:
            apiSequence.append(i)
        record.method_api_sequence=apiSequence
        method_block_flat=list()
        for i in jf.getFlattenTree(record.method_block):
            method_block_flat.append(i)
        record.method_block_flat=method_block_flat
        record.fullmethod_annotation_matrix = similarity.init_doc_matrix(record.fullmethod_annotation_words, w2v)
        record.fullmethod_annotation_idf_vector = similarity.init_doc_idf_vector(record.fullmethod_annotation_words, idf)
        processed_records.append(record)
    return processed_records

def preprocess_all_records_new(records, idf, w2v):
    processed_records = list()
    for record in records:
        # class_description_words = WordPunctTokenizer().tokenize(record.class_description.lower())
        method_annotation_words = WordPunctTokenizer().tokenize(record.method_annotation.lower())
        decompose_methodname_words=WordPunctTokenizer().tokenize(record.decompose_methodname.lower())
        fullmethod_annotation_words = WordPunctTokenizer().tokenize(record.fullmethod_annotation.lower())
        # class_description_words = [SnowballStemmer('english').stem(word) for word in class_description_words]
        method_annotation_words = [SnowballStemmer('english').stem(word) for word in method_annotation_words]
        decompose_methodname_words=[SnowballStemmer('english').stem(word) for word in decompose_methodname_words]
        fullmethod_annotation_words = [SnowballStemmer('english').stem(word) for word in fullmethod_annotation_words]
        # record.class_description_words = class_description_words
        # record.method_annotation_words = method_annotation_words
        record.method_annotation_words = method_annotation_words
        record.decompose_methodname_words=decompose_methodname_words
        record.fullmethod_annotation_words=fullmethod_annotation_words
        record.method_annotation_matrix = similarity.init_doc_matrix(method_annotation_words, w2v)
        record.method_annotation_idf_vector = similarity.init_doc_idf_vector(method_annotation_words,idf)
        record.decompose_methodname_matrix=similarity.init_doc_matrix(decompose_methodname_words, w2v)
        record.decompose_methodname_idf_vector=similarity.init_doc_idf_vector(decompose_methodname_words, w2v)
        record.fullmethod_annotation_matrix=similarity.init_doc_matrix(fullmethod_annotation_words, w2v)
        record.fullmethod_annotation_idf_vector=similarity.init_doc_idf_vector(fullmethod_annotation_words,idf)
        processed_records.append(record)
    return processed_records




def get_experiments():
    load_f = codecs.open("../data/0%/experiment.json", 'r', encoding="utf-8")
    experiments = list()
    #  method_annotation, now_method_flat, true_api
    for line in load_f:
        try:
            load_dict = json.loads(line)
            method_annotation = load_dict["methodannotation"]
            decompose_methodname=load_dict["decomposemethodname"]
            now_method_flat = []
            block_flat = load_dict["nowflat"].split()
            for i in block_flat:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    now_method_flat.append(i)
            # method_block_flat=load_dict["methodflat"]
            true_api = []
            now_api=[]
            api_sequence = load_dict["groudtruth"].split()
            for i in api_sequence:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    true_api.append(i)
            # method_api_sequence=load_dict["apisequence"]
            # 去除没有api序列的
            now_api_sequence=load_dict["nowapisequence"].split()
            for i in now_api_sequence:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    now_api.append(i)
            if len(set(true_api)-set(now_api)) == 0:
                continue
            experiment = domain.Experiment(method_annotation, now_method_flat, true_api,decompose_methodname,now_api)
            experiments.append(experiment)
        except Exception as e:
            print(e)
    return experiments

def get_class_experiments():
    load_f = codecs.open("../data/0%/experiment.json", 'r', encoding="utf-8")
    experiments = list()
    #  method_annotation, now_method_flat, true_api
    for line in load_f:
        try:
            load_dict = json.loads(line)
            method_annotation = load_dict["methodannotation"]
            decompose_methodname=load_dict["decomposemethodname"]
            now_method_flat = []
            block_flat = load_dict["nowflat"].split()
            for i in block_flat:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    now_method_flat.append(i)
            true_api = []
            now_api=[]
            api_sequence = load_dict["groudtruth"].split()
            for i in api_sequence:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    true_api.append(i)
            # 去除没有api序列的
            now_api_sequence=load_dict["nowapisequence"].split()
            for i in now_api_sequence:
                if "java.io.PrintStream.println" != i or "java.io.PrintStream.print" != i:
                    now_api.append(i)
            true_api_class = [api.split('.')[-2] for api in true_api]

            now_api_class = [api.split('.')[-2] for api in now_api]

            if len(set(true_api_class)-set(now_api_class)) == 0:
                continue
            experiment = domain.Experiment(method_annotation, now_method_flat, true_api, decompose_methodname, now_api)
            experiments.append(experiment)
        except Exception as e:
            print(e)
    return experiments
