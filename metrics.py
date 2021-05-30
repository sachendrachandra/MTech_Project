import sys
import nltk
from nltk.translate.bleu_score import *
import nltk.translate.gleu_score as gleu
import sys
import numpy as np

def edits(hyp, ref, print_matrix=False):
  N = len(hyp)
  M = len(ref)
  L = np.zeros((N,M))
  for i in range(0, N):
    for j in range(0, M):
      if min(i,j) == 0:
        L[i,j] = max(i,j)
      else:
        deletion = L[i-1,j] + 1
        insertion = L[i,j-1] + 1
        sub = 1 if hyp[i] != ref[j] else 0
        substitution = L[i-1,j-1] + sub
        L[i,j] = min(deletion, min(insertion, substitution))

  if print_matrix:
    print("WER matrix ({}x{}): ".format(N, M))
    print(L)
  return int(L[N-1, M-1])

def nltk_bleu(hypotheses, references):
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
    print(total_score)
    avg_score = total_score / count
    print ('avg_bleu_score: %.4f' % avg_score)
    return corpus_bleu, avg_score

def nltk_gleu(hypotheses, references):
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        score = gleu.sentence_gleu([hyp], ref, min_len=1,max_len=4)
        total_score += score
        count += 1
    print(total_score)
    avg_score = total_score / count
    print ('avg_gleu_score: %.4f' % avg_score)
    return corpus_bleu, avg_score


def evaluate():

    hypotheses = []
    print('start evaluation')
    with open(sys.argv[2], 'r') as file:
        for line in file:
            hypotheses.append(line.strip())

    references = []
    with open(sys.argv[3], 'r') as file:
        for line in file:
            references.append(line.strip())

    nltk_bleu(hypotheses, references)
    nltk_gleu(hypotheses, references)
    k=0
    m1=m2=m3=0
    for i in range(len(hypotheses)):
      l = edits(hypotheses[i],references[i])
      if (l==0):
        k=k+1
      if (l==1):
        m1=m1+1
      if (l==2):
        m2=m2+1
      if (l==3):
        m3 = m3 + 1


    wer_score(hypotheses,references)
    print("perfect predictions: ",k)
    print("1 , 2, 3 edits: ",m1,m2,m3)

evaluate()


 
