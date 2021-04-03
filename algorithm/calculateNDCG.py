import numpy as np


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    #r1 = [1, 1, 1, 1, 1]
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

if __name__ == "__main__":
    # get dataset from txt
    with open('F:/test_result/test_localization_Einstein/5@ndcg/app.txt', 'r')as f:
        rows = f.readlines()
    for j in range(0, len(rows)):
        rows[j] = rows[j].rstrip('\n')
    # print len(rows)
    vector = np.zeros([len(rows), 1])
    n = 0
    for row in rows:
        r = []
        for i in range(5):
            r.append(int(row[i]))
            k = 5
        #print ndcg_at_k(r, k)
        vector[n] = ndcg_at_k(r, k)
        print(vector[n])
        n = n + 1
    vector_average = np.mean(np.mean(vector, axis = 0))
    np.savetxt("F:/test_result/test_localization_Einstein/5@ndcg/app_ndcg.txt", vector, fmt='%0.17f', newline='\n')
    print(vector_average)