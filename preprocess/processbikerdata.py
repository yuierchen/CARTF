#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pickle
import  codecs
import sys
import os
from jpype import *
from bs4 import BeautifulSoup
from domain import *
import util
sys.path.append("..")
path='../data/api_questions_pickle_new'
jarpath = os.path.join(os.path.abspath('.'), "D:/")
startJVM("F:/Java/jdk1.8.0_181/jre/bin/server/jvm.dll", "-ea",
             "-Djava.class.path=%s" % (jarpath + "StackOverFlowFilter.jar"))
JDClass = JClass("cn.edu.nuaa.xin.match.parser.MyCodeParser")
records=list()
with open(path,'rb') as f:
    data=pickle.load(f)
    # for i in data:
    for question in data:
        title=question.title
        for answer in question.answers:
            if int(answer.score.strip())>0 or answer.id==question.accepted_answer_id:
                answerbody=answer.body
                print(answer.id)
                soup = BeautifulSoup(answer.body, 'html.parser', from_encoding='utf-8')
                codes = soup.find_all('code')
                methodapisequence=""
                methodflat=""
                methodname=""
                links = soup.find_all('a')
                for link in links:
                    link = link['href']
                    if 'docs.oracle.com/javase/' in link and '/api/' in link and 'html' in link:
                        pair = util.parse_api_link(link)  # pair[0] is class name, pair[1] is method name
                        if pair[1] != '':
                            api = pair[0] + '.' + pair[1]
                            methodapisequence=methodapisequence+" "+api
                for code in codes:
                    jd=JDClass()
                    code = code.get_text()
                    try:
                        jd.parseAPI(code)
                    except Exception as e:
                        print(e)
                    methodapisequence=methodapisequence+" "+str(jd.needData.getMethodapisequences())
                    methodflat=methodflat+" "+str(jd.needData.getMethodflat())
                    methodname=methodname+" "+str(jd.needData.getMethodname())
                arecord=Record(title,methodname,methodflat,methodapisequence,methodname)
                records.append(arecord)
pickle.dump(records,open('../data/records_final.pkl','wb'))





