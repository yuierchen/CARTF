import  pickle
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer
import math

answer_corpora = pickle.load(open('../data/titles.pkl','rb'))



texts = [WordPunctTokenizer().tokenize(answer.lower()) for answer in answer_corpora]

idf = {}

stemmer = SnowballStemmer('english')
cc = 0
for text in texts:

    tmp_set = set()

    cc += 1

    for word in text:
        tmp_set.add(word)

    for word in tmp_set:
        word = stemmer.stem(word)
        if word in idf:
            idf[word] += 1
        else: idf[word] = 1


    if cc%10000 == 0:
        print(cc)

for key in idf:
    idf[key] = (idf[key], math.log(len(texts)*1.0 / idf[key]))


pickle.dump(idf,open('../data/my_idf','wb'))


idf = pickle.load(open('../data/my_idf','rb'))

cc = 0
for key in idf:
    value = idf[key]
    if(value[0]<=10): continue
    cc = cc + 1
    print(key,value[0],value[1])

print(cc)