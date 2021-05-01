import re
import sys
import codecs
import json
import domain
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from algorithm import similarity
def parse_api_link(url):

    #example intput: http://docs.oracle.com/javase/7/docs/api/java/text/NumberFormat.html#getInstance(java.util.Locale)
    #example output: (NumberFormat,getInstance)

    #print url

    url = url.split(".html")

    tokens = url[0].split("/")

    i = tokens.index('api')
    class_name = tokens[i+1]
    i = i+2
    while i < len(tokens):
        class_name = class_name+'.'+tokens[i]
        i = i+1

    #class_name = url[0].split("/")[-1]
    method_name = url[1]

    if method_name != '':
        method_name = method_name[1:]
        for i,ch in enumerate(method_name):
            if not method_name[i].isalpha():
                method_name = method_name[:i]
                break

    #print class_name,method_name

    return (class_name,method_name) #Note that class_name already contains package name


def normalize_dict(dic):

    min_value = sys.maxint
    max_value = -1

    for (k,v) in dic.items():
        min_value = min(min_value,v)
        max_value = max(max_value,v)

    for k in dic:
        dic[k] = (dic[k]-min_value+1)*1.0/(max_value-min_value+1)

def get_experiments():
    load_f = codecs.open("../data/stackoverflow_experiment_expand80%.json", 'r', encoding="utf-8")
    experiments = list()
    #  method_annotation, now_method_flat, true_api
    for line in load_f:
        try:
            load_dict = json.loads(line)
            annotation = load_dict["methodannotation"].strip()
            if annotation[0:2] == '**':
                continue
            annotation = annotation[7:]
            atitle = annotation.split('\n')[0]
            # print(atitle)
            if atitle[-1] == '?' or atitle[-1] == '.':
                atitle = atitle[:-1]
                atitle = atitle.strip()
            title = atitle
            now_method_flat = load_dict["nowflat"].split()

            # method_block_flat=load_dict["methodflat"]

            api_sequence = load_dict["groudtruth"].split()
            # method_api_sequence=load_dict["apisequence"]
            # 去除没有api序列的
            now_api_sequence=load_dict["nowapisequence"].split()
            if len(set(api_sequence)-set(now_api_sequence)) == 0:
                continue
            experiment = domain.Experiment(title, now_method_flat, api_sequence,now_api_sequence)
            experiments.append(experiment)
        except Exception as e:
            print(e)
    return experiments

def get_class_experiments():
    load_f = codecs.open("../data/stackoverflow_experiment_expand0%.json", 'r', encoding="utf-8")
    experiments = list()
    #  method_annotation, now_method_flat, true_api
    for line in load_f:
        try:
            load_dict = json.loads(line)
            annotation = load_dict["methodannotation"].strip()
            if annotation[0:2] == '**':
                continue
            annotation = annotation[7:]
            atitle = annotation.split('\n')[0]
            # print(atitle)
            if atitle[-1] == '?' or atitle[-1] == '.':
                atitle = atitle[:-1]
                atitle = atitle.strip()
            title = atitle

            now_method_flat = load_dict["nowflat"].split()
            api_sequence = load_dict["groudtruth"].split()

            # 去除没有api序列的
            now_api_sequence=load_dict["nowapisequence"].split()

            true_api_class = [api.split('.')[-2] for api in api_sequence]

            now_api_class = [api.split('.')[-2] for api in now_api_sequence]

            if len(set(true_api_class)-set(now_api_class)) == 0:
                continue
            experiment = domain.Experiment(title, now_method_flat, api_sequence, now_api_sequence)
            experiments.append(experiment)
        except Exception as e:
            print(e)
    return experiments


def preprocess_all_records_new(records, idf, w2v):
    processed_records = list()
    for record in records:
        # class_description_words = WordPunctTokenizer().tokenize(record.class_description.lower())
        title_words = WordPunctTokenizer().tokenize(record.title.lower())
        title_words = [SnowballStemmer('english').stem(word) for word in title_words]
        # record.class_description_words = class_description_words
        # record.method_annotation_words = method_annotation_words
        record.title_words = title_words
        record.title_matrix = similarity.init_doc_matrix(title_words, w2v)
        record.title_idf_vector = similarity.init_doc_idf_vector(title_words,idf)
        method_api_sequence=record.method_api_sequence.split()
        final_method_api_sequence=list()
        for api in method_api_sequence:
            if api[0]=="." or api[:api.find(".")].lower=="missing":
                continue
            final_method_api_sequence.append(api)
        record.method_api_sequence=final_method_api_sequence
        record.method_block_flat=record.method_block_flat.split()
        if len(record.method_api_sequence)<=0:
            continue
        processed_records.append(record)
    return processed_records