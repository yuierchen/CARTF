from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import gensim
import pickle
import numpy as np
from algorithm import smithwaterman
from sklearn import preprocessing
import Levenshtein as lst

def init_doc_matrix(doc,w2v):

    matrix = np.zeros((len(doc),100)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[word])
    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)
    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))
    return matrix

def init_doc_idf_vector(doc,idf):
    idf_vector = np.zeros((1,len(doc)))  # word embedding size is 100
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word][1]

    return idf_vector



def sim_doc_pair(matrix1,matrix2,idf1,idf2):

    sim12 = (idf1*(matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()

    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()

    s = float(sim12 + sim21)
    if s == 0:
        return 0
    else:
        return 2 * sim12 * sim21 / s
    # return 2 * sim12 * sim21 / (sim12 + sim21)
    # return sim12+sim21
    # total_len = matrix1.shape[0] + matrix2.shape[0]
    # return sim12 * matrix2.shape[0] / total_len + sim21 * matrix1.shape[0] / total_len

def sim_doc_mypair(matrix1,matrix2):
    ab12=matrix1.dot(matrix2.T).max(axis=1)
    ab21=matrix2.dot(matrix1.T).max(axis=1)
    sim12=ab12.sum()/len(ab12[ab12!=0])
    sim21=ab21.sum()/len(ab21[ab21!=0])
    return 2 * sim12 * sim21 / (sim12 + sim21)
    # return sim12+sim21

def lst(str1,str2):
    return lst.ratio(str1,str2)


def sim_method_flat_pair(seq1,seq2,mS, mmS, w1):
    return 2*smithwaterman.Smith_Waterman(seq1, seq2, mS, mmS, w1)/(len(seq1)+len(seq2))
    # return smithwaterman.Smith_Waterman(seq1, seq2, mS, mmS, w1)
    # return smithwaterman.Smith_Waterman(seq1, seq2, mS, mmS, w1) / len(seq1)

def get_topk_questions(query_matrix,query_idf_vector,questions,topk,similar_threshold):
    relevant_questions = list()
    for question in questions:

        sim = sim_doc_pair(query_matrix, question.fullmethod_annotation_matrix, query_idf_vector,
                                                         question.fullmethod_annotation_idf_vector)
        relevant_questions.append((question, question.method_annotation_words, sim))

    list_relevant_questions = sorted(relevant_questions, key=lambda question: question[2], reverse=True)

    # get the ids of top-k most relevant questions
    top_questions = dict()
    for i, item in enumerate(list_relevant_questions):
        if(item[2]<similar_threshold):
            break
        top_questions[item[0]] = item[1]
        if i+1 == topk:
            break
    return top_questions

def get_topk_method_flat(now_method_flat,record_method_flat,topk,mS, mmS, w1):
    relevant_method_flat=list()
    for method_flat in record_method_flat:
        # print(api_sequence)
        sim=sim_method_flat_pair(now_method_flat,method_flat,mS, mmS, w1)
        relevant_method_flat.append((method_flat,sim))
    list_relevant_api_sequence=sorted(relevant_method_flat, key=lambda method_flat: method_flat[1], reverse=True)
    top_method_flat = dict()
    for i, item in enumerate(list_relevant_api_sequence):
        top_method_flat[item[1]] = item[0]
        if i + 1 == topk:
            break

    return top_method_flat

if __name__ == "__main__":
    w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')

    idf = pickle.load(open('../data/idf','rb'))



    question1 = 'intialize all elements in an ArrayList as a specific integer'
    question1 = WordPunctTokenizer().tokenize(question1.lower())
    question1 = [SnowballStemmer('english').stem(word) for word in question1]

    question2 = 'set every element of a list to the same constant value'
    question2 = WordPunctTokenizer().tokenize(question2.lower())
    question2 = [SnowballStemmer('english').stem(word) for word in question2]

    matrix1 = init_doc_matrix(question1,w2v)
    matrix2 = init_doc_matrix(question2,w2v)
    matrix1_trans = matrix1.T
    matrix2_trans = matrix2.T

    idf1 = init_doc_idf_vector(question1,idf)
    idf2 = init_doc_idf_vector(question2,idf)

    #print sim_question_api(question1, question2, idf, w2v)
    print(sim_doc_pair(matrix1, matrix2, idf1, idf2))
