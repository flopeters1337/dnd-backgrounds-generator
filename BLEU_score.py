# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:31:48 2020

@author: Elodie
"""

#https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
import nltk
import pickle 
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu



infile = open('Preprocessing_file','rb')
new_output = pickle.load(infile)
infile.close()




reference = [new_output[0], new_output[1]]
candidate = ['the', 'gargoyles','chasing','the', 'him', '.', '_end_']
score = sentence_bleu(reference, candidate)
print(score)