#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import logging
import sys
import pickle
sys.path.append("..")
import tensorly as tl
import os
import os.path
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import gensim
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ncp import NCP_BCU
from tensorD.demo.DataGenerator import *
from jpype import *
import csv
from algorithm import similarity
import util
from algorithm import calculateNDCG
import nimfa
import time

def main():
    w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')  # pre-trained word embedding
    idf = pickle.load(open('../data/idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary
    records = pickle.load(open("../data/records.pkl", 'rb'))
    #获取需要推荐的问题
    experiments =util.get_experiments()
    print(len(experiments))
    csvfile_path = os.path.join(args.output_path, "topmethod.csv")#输出结果
    csvfile = open(csvfile_path, 'w', newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["question_title","top5", "is_exist","is_global_exist","ground_truth_intersection","true_apis"])
    #所有问题的api的集合，看这个集合里面是否有答案存在
    global_record_api = list()
    for record in records:
        for api in record.method_api_sequence:
            if api not in global_record_api:
                global_record_api.append(api)

    #统计能进行推荐的问题个数，推荐出来的问题的个数
    recommend_num=0
    recommend_success_num=0
    true_num = 0
    global_true_num = 0
    processnum=0
    #统计指标
    mrr = 0.0
    map = 0.0
    precision = 0
    recall = 0
    ndcg = 0.0

    rec_num = args.rec_num
    start = time.clock()
    for experiment in experiments:
        experiment_method_annotation=experiment.method_annotation
        experiment_now_method_flat=experiment.now_method_flat
        experiment_true_api=experiment.true_api
        experiment_now_api=experiment.now_api
        experiment_true_api=set(experiment_true_api)-set(experiment_now_api)
        experiment_decompose_methodname=experiment.decompose_methodname
        full_query=experiment_method_annotation+" "+experiment_decompose_methodname
        full_query_words = WordPunctTokenizer().tokenize(full_query.lower())
        full_query_words = [SnowballStemmer('english').stem(word) for word in full_query_words]
        full_query_matrix = similarity.init_doc_matrix(full_query_words, w2v)
        full_query_idf_vector = similarity.init_doc_idf_vector(full_query_words, idf)
        query_words = WordPunctTokenizer().tokenize(experiment_method_annotation.lower())
        query_words = [SnowballStemmer('english').stem(word) for word in query_words]

        #获取相似的TOP-N问题
        top_questions = similarity.get_topk_questions(full_query_matrix, full_query_idf_vector, records, 50, 0.0)
        #获取得到问题的长度
        print(top_questions)
        similar_questions_length=len(top_questions)
        print("similar_questions_length:",similar_questions_length)
        #查看现有问题是否在相似问题中，如果不在则加入，否则直接根据相似问题构建张量
        flag=False

        similar_records_list=list(top_questions.keys())
        for record in similar_records_list:
            if(record.method_annotation_words ==query_words):
                flag=True
        processnum+=1
        #现有问题在相似问题里面
        record_method_annotation_words = list()
        record_method_flat = list()
        record_api = list()
        for record in similar_records_list:
            if record.method_annotation_words not in record_method_annotation_words:
                record_method_annotation_words.append(record.method_annotation_words)
            if record.method_block_flat not in record_method_flat:
                record_method_flat.append(record.method_block_flat)
            for api in record.method_api_sequence:
                if api not in record_api:
                    record_api.append(api)
        for now_api in experiment_now_api:
            if now_api not in record_api:
                record_api.append(now_api)
        api_rec_all = []
        api_rec = []
        if flag==True:
            recommend_num+=1
            print(len(record_method_annotation_words), len(record_method_flat), len(record_api))
            record_method_annotation_words_dict = dict(zip(range(len(record_method_annotation_words)), record_method_annotation_words))
            record_method_flat_dict = dict(zip(range(len(record_method_flat)), record_method_flat))
            record_api_dict = dict(zip(range(len(record_api)), record_api))
            tensor = np.zeros((len(record_method_annotation_words), len(record_method_flat), len(record_api)), dtype=int)
            for record in similar_records_list:
                for concrete_api in record.method_api_sequence:
                    tensor[list(record_method_annotation_words_dict.keys())[
                               list(record_method_annotation_words_dict.values()).index(
                                   record.method_annotation_words)],
                           list(record_method_flat_dict.keys())[
                               list(record_method_flat_dict.values()).index(record.method_block_flat)],
                           list(record_api_dict.keys())[list(record_api_dict.values()).index(concrete_api)]] = 1
            for api in experiment_now_api:
                if api in record_api_dict.values():
                    tensor[list(record_method_annotation_words_dict.keys())[
                               list(record_method_annotation_words_dict.values()).index(
                                   query_words)],
                          :,
                           list(record_api_dict.keys())[list(record_api_dict.values()).index(api)]] = 1
            #处理不是张量的情况
            one = query_words
            if(len(record_method_annotation_words) ==1 or len(record_method_flat) ==1 or len(record_api) ==1):
                if(len(record_method_annotation_words) ==1 and len(record_method_flat) ==1
                        or len(record_method_flat) ==1 and len(record_api) ==1
                        or len(record_api) ==1 and len(record_method_annotation_words) ==1):
                    api_rec_all=record_api
                    for m in set(experiment_now_api):
                        if m in api_rec_all:
                            api_rec_all.remove(m)
                    api_rec = api_rec_all[:rec_num]
                elif(len(record_api)==1):
                    api_rec_all = record_api
                    for m in set(experiment_now_api):
                        if m in api_rec_all:
                            api_rec_all.remove(m)
                    api_rec = api_rec_all[:rec_num]
                else:
                    if(len(record_method_annotation_words)==1):
                        matrix = tl.unfold(tensor,mode=1)
                        nmf = nimfa.Nmf(matrix, max_iter=200, rank=round(min(matrix.shape)), update='euclidean',
                                        objective='fro')
                        nmf_fit = nmf()
                        W = nmf_fit.basis()
                        H = nmf_fit.coef()
                        matrix = np.dot(W, H)
                        two = list(
                            similarity.get_topk_method_flat(experiment_now_method_flat,
                                                            list(record_method_flat_dict.values()), 1, 1,
                                                            -1, 1).values())[0]
                        rec_combine_api_key = np.argsort(
                            -matrix[list(record_method_flat_dict.keys())[list(record_method_flat_dict.values()).index(two)]
                            , :]).tolist()[0]
                        api_rec_all = [record_api_dict[i] for i in rec_combine_api_key]
                        for m in set(experiment_now_api):
                            if m in api_rec_all:
                                api_rec_all.remove(m)
                        api_rec = api_rec_all[:rec_num]
                    elif(len(record_method_flat)==1):
                        matrix = tl.unfold(tensor, mode=0)
                        nmf = nimfa.Nmf(matrix, max_iter=200, rank=round(min(matrix.shape)), update='euclidean',
                                        objective='fro')
                        nmf_fit = nmf()
                        W = nmf_fit.basis()
                        H = nmf_fit.coef()
                        matrix = np.dot(W, H)
                        rec_combine_api_key = np.argsort(
                            -matrix[list(record_method_annotation_words_dict.keys())[
                                             list(record_method_annotation_words_dict.values()).index(one)]
                            , :]).tolist()[0]
                        api_rec_all = [record_api_dict[i] for i in rec_combine_api_key]
                        for m in set(experiment_now_api):
                            if m in api_rec_all:
                                api_rec_all.remove(m)
                        api_rec = api_rec_all[:rec_num]

            else:
            #张量分解
                tf.reset_default_graph()
                tensor = tl.tensor(tensor).astype(np.float32)
                data_provider = Provider()
                data_provider.full_tensor = lambda: tensor
                env = Environment(data_provider, summary_path='/tensor/ncp_ml')
                ncp = NCP_BCU(env)
                arg = NCP_BCU.NCP_Args(rank=round(min(len(record_method_annotation_words), len(record_method_flat), len(record_api))/2), validation_internal=1)
                ncp.build_model(arg)
                loss_hist = ncp.train(100)
                factor_matrices = ncp.factors
                full_tensor = tl.kruskal_to_tensor(factor_matrices)

                two = list(
                    similarity.get_topk_method_flat(experiment_now_method_flat, list(record_method_flat_dict.values()), 1, 1,
                                                            -1, 1).values())[0]

                rec_combine_api_key = np.argsort(
                    -full_tensor[list(record_method_annotation_words_dict.keys())[
                                     list(record_method_annotation_words_dict.values()).index(one)]
                    , list(record_method_flat_dict.keys())[list(record_method_flat_dict.values()).index(two)]
                    , :]).tolist()
                # 推荐的API列表,去除情境中已经含有的api
                api_rec_all=[record_api_dict[i] for i in rec_combine_api_key]
                for m in set(experiment_now_api):
                    if m in api_rec_all:
                        api_rec_all.remove(m)
                api_rec = api_rec_all[:rec_num]

        #现有问题不在相似问题里面
        else:
            similar_questions_length+=1

            #去除找不到相似问题的问题
            if similar_questions_length==1:
                continue
            recommend_num+=1
            #添加新来的query
            record_method_annotation_words.append(query_words)
            print(len(record_method_annotation_words), len(record_method_flat), len(record_api))
            #构建张量
            record_method_annotation_words_dict = dict(zip(range(len(record_method_annotation_words)), record_method_annotation_words))
            record_method_flat_dict = dict(zip(range(len(record_method_flat)), record_method_flat))
            record_api_dict = dict(zip(range(len(record_api)), record_api))
            tensor = np.zeros((len(record_method_annotation_words), len(record_method_flat), len(record_api)), dtype=int)
            for record in similar_records_list:
                for concrete_api in record.method_api_sequence:
                    tensor[list(record_method_annotation_words_dict.keys())[
                            list(record_method_annotation_words_dict.values()).index(record.method_annotation_words)],
                               list(record_method_flat_dict.keys())[
                                   list(record_method_flat_dict.values()).index(record.method_block_flat)],
                               list(record_api_dict.keys())[list(record_api_dict.values()).index(concrete_api)]] = 1

            for api in experiment_now_api:
                if api in record_api_dict.values():
                    tensor[list(record_method_annotation_words_dict.keys())[
                               list(record_method_annotation_words_dict.values()).index(
                                   query_words)],
                          :,
                           list(record_api_dict.keys())[list(record_api_dict.values()).index(api)]] = 1
            #处理不是张量分解
            one = query_words
            if (len(record_method_annotation_words) == 1 or len(record_method_flat) == 1 or len(record_api) == 1):
                if (len(record_method_annotation_words) == 1 and len(record_method_flat) == 1
                        or len(record_method_flat) == 1 and len(record_api) == 1
                        or len(record_api) == 1 and len(record_method_annotation_words) == 1):
                    api_rec_all = record_api
                    for m in set(experiment_now_api):
                        if m in api_rec_all:
                            api_rec_all.remove(m)
                    api_rec = api_rec_all[:rec_num]
                elif (len(record_api) == 1):
                    api_rec_all = record_api
                    for m in set(experiment_now_api):
                        if m in api_rec_all:
                            api_rec_all.remove(m)
                    api_rec = api_rec_all[:rec_num]
                else:
                    if (len(record_method_annotation_words) == 1):
                        matrix = tl.unfold(tensor, mode=1)
                        nmf = nimfa.Nmf(matrix, max_iter=200, rank=round(min(matrix.shape)), update='euclidean',
                                        objective='fro')
                        nmf_fit = nmf()
                        W = nmf_fit.basis()
                        H = nmf_fit.coef()
                        matrix = np.dot(W, H)
                        two = list(
                            similarity.get_topk_method_flat(experiment_now_method_flat,
                                                            list(record_method_flat_dict.values()), 1, 1,
                                                            -1, 1).values())[0]
                        rec_combine_api_key = np.argsort(
                            -matrix[
                             list(record_method_flat_dict.keys())[list(record_method_flat_dict.values()).index(two)]
                            , :]).tolist()[0]
                        api_rec_all = [record_api_dict[i] for i in rec_combine_api_key]
                        for m in set(experiment_now_api):
                            if m in api_rec_all:
                                api_rec_all.remove(m)
                        api_rec = api_rec_all[:rec_num]
                    elif (len(record_method_flat) == 1):
                        matrix = tl.unfold(tensor, mode=0)
                        nmf = nimfa.Nmf(matrix, max_iter=200, rank=round(min(matrix.shape)), update='euclidean',
                                        objective='fro')
                        nmf_fit = nmf()
                        W = nmf_fit.basis()
                        H = nmf_fit.coef()
                        matrix = np.dot(W, H)
                        rec_combine_api_key = np.argsort(
                            -matrix[list(record_method_annotation_words_dict.keys())[
                                        list(record_method_annotation_words_dict.values()).index(one)]
                            , :]).tolist()[0]
                        api_rec_all = [record_api_dict[i] for i in rec_combine_api_key]
                        for m in set(experiment_now_api):
                            if m in api_rec_all:
                                api_rec_all.remove(m)
                        api_rec = api_rec_all[:rec_num]

            else:
            # 张量分解
                tf.reset_default_graph()
                tensor = tl.tensor(tensor).astype(np.float32)
                data_provider = Provider()
                data_provider.full_tensor = lambda: tensor
                env = Environment(data_provider, summary_path='/tensor/ncp_ml')
                ncp = NCP_BCU(env)
                arg = NCP_BCU.NCP_Args(rank=round(min(len(record_method_annotation_words), len(record_method_flat), len(record_api))/2), validation_internal=1)
                ncp.build_model(arg)
                loss_hist = ncp.train(100)
                factor_matrices = ncp.factors
                full_tensor = tl.kruskal_to_tensor(factor_matrices)
                # one = query_words
                two = list(similarity.get_topk_method_flat(experiment_now_method_flat,list(record_method_flat_dict.values()), 1, 1,
                                                            -1, 1).values())[0]

                rec_combine_api_key = np.argsort(
                    -full_tensor[list(record_method_annotation_words_dict.keys())[list(record_method_annotation_words_dict.values()).index(one)]
                    , list(record_method_flat_dict.keys())[list(record_method_flat_dict.values()).index(two)]
                    , :]).tolist()
                #推荐的API列表
                api_rec_all = [record_api_dict[i] for i in rec_combine_api_key]
                for m in set(experiment_now_api):
                    if m in api_rec_all:
                        api_rec_all.remove(m)
                api_rec = api_rec_all[:rec_num]
        #判断结果在相似的问题中有没有出现

        appearance_flag=True
        if(len(set(experiment_true_api).intersection(set(record_api)))==0):
            appearance_flag=False
        if(appearance_flag==True):
            true_num+=1
        global_appearance_flag=True
        if (len(set(experiment_true_api).intersection(set(global_record_api))) == 0):
            global_appearance_flag = False
        if(global_appearance_flag==True):
            global_true_num+=1

        pos = -1
        tmp_map=0.0
        hits=0.0
        vector = list()
        for i,api in enumerate(api_rec_all[:rec_num]):
            if api in set(experiment_true_api) and pos==-1:
                pos=i+1
            if api in set(experiment_true_api):
                vector.append(1)
                hits+=1
                tmp_map+=hits/(i+1)
            else:
                vector.append(0)

        tmp_map/=len(set(experiment_true_api))
        tmp_mrr=0.0
        if pos!=-1:
            tmp_mrr=1.0/pos
        map+=tmp_map
        mrr+=tmp_mrr
        ndcg += calculateNDCG.ndcg_at_k(vector[:rec_num], rec_num)
        ground_truth_intersection=set(api_rec).intersection(set(experiment_true_api))
        if(len(ground_truth_intersection)>0):
            recommend_success_num+=1
        precision+=len(ground_truth_intersection)/rec_num
        recall+=len(ground_truth_intersection)/len(set(experiment_true_api))
        writer.writerow([experiment_method_annotation,
                         api_rec,
                         appearance_flag,
                         global_appearance_flag,
                         ground_truth_intersection,
                         experiment_true_api])

    writer.writerow(["recommend_num","recommend_success_num","exist_in_similar_questions_num","exist_in_global_num"])
    writer.writerow([recommend_num,recommend_success_num,true_num,global_true_num])
    writer.writerow(["mrr/recommend_num", "recommend_num", "map/recommend_num", "success_rate@N", "precision@N/recommend_num",
                     "recall@N/recommend_num","ndcg/recommend_num"]),
    writer.writerow([mrr /recommend_num, recommend_num, map / recommend_num, recommend_success_num / recommend_num, precision / recommend_num,
                     recall / recommend_num,ndcg/recommend_num])
    csvfile.close()
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))





    logging.info("Finish")







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', dest="output_path",
                        default='../data/0%/',
                        help='Path to output file ')
    parser.add_argument('-r', dest="r_rank", default=10, type=int,
                        help="The rank of the decomposition matirx. Default: 10")
    parser.add_argument('--m', dest="rec_num",
                        default=10, help='the number of recommend')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()

