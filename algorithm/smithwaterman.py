#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from Bio import SeqIO

def compare(m, n, match, n_match):
    if m == n:
        return match
    else:
        return n_match

def Smith_Waterman(seq1, seq2, mS, mmS, w1):
    print(seq1)
    print(seq2)
    path = {}
    S = np.zeros([len(seq1) + 1, len(seq2) + 1], int)

    for i in range(0, len(seq1) + 1):
        for j in range(0, len(seq2) + 1):
            if i == 0 or j == 0:
                path['[' + str(i) + ', ' + str(j) + ']'] = []
            else:
                if seq1[i - 1] == seq2[j - 1]:
                    s = mS
                else:
                    s = mmS
                L = S[i - 1, j - 1] + s
                P = S[i - 1, j] - w1
                Q = S[i, j - 1] - w1
                S[i, j] = max(L, P, Q, 0)
                path['[' + str(i) + ', ' + str(j) + ']'] = []
                if L == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j - 1) + ']')
                if P == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j) + ']')
                if Q == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i) + ', ' + str(j - 1) + ']')

    print("S = ", S)
    end = np.argwhere(S == S.max())
    match_max_length=[]
    for i in end:
        key = str(list(i))
        value = path[key]
        result = [key]
        match_max_length.append(traceback(path, S, value, result, seq1, seq2))
        # print("----------------")
        # print(match_max_length)
    return max(match_max_length)

def Smith_Waterman_aff(seq1, seq2, match, n_match, u, v):
    a = len(seq1)
    b = len(seq2)
    path = {}
    S = np.zeros((a + 1, b + 1))
    L = np.zeros((a + 1, b + 1))
    P = np.zeros((a + 1, b + 1))
    Q = np.zeros((a + 1, b + 1))
    seq1 = " " + seq1[:]
    seq2 = " " + seq2[:]
    for r in range(1, b + 1 if a > b else a + 1):
        for c in range(r, b + 1):
            L[r, c] = S[r - 1, c - 1] + compare(seq1[r], seq2[c], match, n_match)
            P[r, c] = max(np.add(S[0:r, c], -(np.arange(r, 0, -1) * u + v)))
            Q[r, c] = max(np.add(S[r, 0:c], -(np.arange(c, 0, -1) * u + v)))
            S[r, c] = max([0, L[r, c], P[r, c], Q[r, c]])
        for c in range(r + 1, a + 1):
            L[c, r] = S[c - 1, r - 1] + compare(seq1[c], seq2[r], match, n_match)
            P[c, r] = max(np.add(S[0:c, r], -(np.arange(c, 0, -1) * u + v)))
            Q[c, r] = max(np.add(S[c, 0:r], -(np.arange(r, 0, -1) * u + v)))
            S[c, r] = max([0, L[c, r], P[c, r], Q[c, r]])
        for i in range(0, len(seq1)):
            for j in range(0, len(seq2)):
                if i == 0 or j == 0:
                    path['[' + str(i) + ', ' + str(j) + ']'] = []
                else:
                    path['[' + str(i) + ', ' + str(j) + ']'] = []
                    if L[i,j] == S[i, j]:
                        path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j - 1) + ']')
                    if P[i,j] == S[i, j]:
                        path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j) + ']')
                    if Q[i,j] == S[i, j]:
                        path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i) + ', ' + str(j - 1) + ']')
    print("S = ", S)
    end = np.argwhere(S == S.max())
    print (S)
    # match_max_length=0
    for i in end:
        key = str(list(i))
        value = path[key]
        result = [key]
        match_max_length=traceback(path, S, value, result, seq1, seq2)
    #     print("----------------")
    #     print(match_max_length)
    # return match_max_length
def traceback(path, S, value, result, seq1, seq2):
    i=0
    j=0
    if value != []:
        key = value[0]
        result.append(key)
        value = path[key]
        print(key)
        i = int((key.split(',')[0]).strip('['))
        j = int((key.split(',')[1]).strip(']'))
    if S[i, j] == 0:
        x = 0
        y = 0
        s1 = ''
        s2 = ''
        md = ''
        for n in range(len(result)-2, -1, -1):
            point = result[n]
            i = int((point.split(',')[0]).strip('['))
            j = int((point.split(',')[1]).strip(']'))
            if i == x:
                s1 += '-'
                s2 += seq2[j-1]
                md += ' '
            elif j == y:
                s1 += seq1[i-1]
                s2 += '-'
                md += ' '
            else:
                s1 += seq1[i-1]
                s2 += seq2[j-1]
                md += '|'
            x = i
            y = j
        print('alignment result:')
        print('s1: %s'%s1)
        print('    '+md)
        print('s2: %s'%s2)
        # print(max(len(s1),len(s2)))
        return max(len(s1),len(s2))
    else:
         return traceback(path, S, value, result, seq1, seq2)


# f1 = 'D:/edit/biology informatics/FJ215665.fasta'
# f2 = 'D:/edit/biology informatics/AY049983.2.fasta'
# print "\nFILE: " + f1
# print open(f1, 'r').read()
# print "\nFILE: " + f2
# print open(f2, 'r').read()
# fr1 = open(f1, 'r')
# fr2 = open(f2, 'r')
# seq1 = SeqIO.read(fr1, "fasta")
# seq2 = SeqIO.read(fr2, "fasta")
# fr1.close()
# fr2.close()
# m=Smith_Waterman(['1','[','2',']','3','5'], ['1','1',']','1','1','[','2','3','2','3','3'], 1, -1/3, 1)
# m=Smith_Waterman(['[', 'k', '[', ']', '[', '#', ']', 'o', 'y', '#', '[', 'J', '[', '?', '[', '$', '#', ']', '[', '>', '#', '[', '0', '[', '#', '#', '[', 'z', '#', '#', ']', ']', '&', ']', ']', ']', '[', '4', '[', 'I', '#', '[', 'java.lang.String.format', '#', '#', '[', '#', '#', '[', 'z', '#', '#', ']', ']', '[', '#', '#', ']', ']', ']', ']', ']', ']']
# , ['[', 'k', '[', ']', 'o', 'y', '#', '[', 'J', '[', '?', 'y', '[', '>', '#', '[', '[', 'z', '#', '#', ']', '#', ']', ']', ']', '[', '4', '[', 'java.util.logging.Logger.log', '[', 'java.util.logging.Logger.getLogger', '#', '#', '&', ']', '#', '[', 'z', '#', '#', ']', '&', '#', ']', ']', '[', '@', 'L', ']', ']', ']'], 1, -1/3, 1)
# print('length',m)
# Smith_Waterman_aff(['1','2','3','5'], ['1','1','1','1','2','2','3','3'], 1, -1/3, 1, 1/3)
