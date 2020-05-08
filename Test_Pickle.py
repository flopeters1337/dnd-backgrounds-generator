# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:56:08 2020

@author: Elodie
"""
import pickle 


infile = open('Preprocessing_file','rb')
new_output = pickle.load(infile)
infile.close()
print(new_output)