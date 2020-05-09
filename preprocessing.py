###################
#Import libraries
import pandas as pd
import re
import string
#import nltk
#from nltk.tokenize import word_tokenize
import pickle
import os

#Import dataset 
dataset = pd.read_excel(os.path.join('Data', 'dd_bios.xls'), encoding='latin1')
#print(dataset)
dataset.columns = ['Time', 'Name', 'Race', 'Class', 'Backstory']

#Delete rows with missing elements
dataset=dataset.dropna()
dataset=dataset.reset_index(drop=True)
#print(dataset)


#Remove punctuation 
#This function removes punctuation and English quotation marks from a string called text. However it keeps the hyphen in the text. 
def remove_punct(text):
    text = text.replace("“", "")
    text = text.replace("”", "")
    words = text.split()
    remove=string.punctuation
    remove = remove.replace("-", "")
    table = str.maketrans('', '', remove)
    stripped = [w.translate(table) for w in words]
    return stripped


#Function to apply before the tokenization 
#This function 
    #-replaces interrogation and exclamation points by points in the text
    #-removes quotation marks in the text
    #-replaces every number by the sring "_Number_" in the text
    #-replaces  the string name in the text by _Name_
    #-adds "_end_" at the end of the text if the option end is activated 
#The inputs are two strings text and name and an option end (either 0 or 1)
def Before_Token(text, name, end):
    text = text.replace("?", ".")
    text = text.replace("!", ".")
    text = text.replace(".", " .")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = re.sub("\d+", "_number_", text)
    if name in text: #if the string name is in the text
        text = re.sub(name, "_name_", text) # replaces name by _Name_
    else: #if not, look if one part of the string is in the text and if yes, replace it by _Name_
        name=remove_punct(name)
        for i in range(0, len(name)):
            text = re.sub(name[i], "_name_", text)  
    if end==1:
        text = text + " _end_"
    return text
    
#Tokenization, words by word, of the string text by keeping the point. 
#First this function applies the function Before_Token. 
#The inputs are two strings text and name.
def Tokenization_per_backstory(text, name):
   text= Before_Token(text, name, 1)
   words = text.split()
   return [word.lower() for word in words]

def Tokenization_per_sentence(text, name):
   text= Before_Token(text, name,0)
   words = text.split(".")
   return [word.lower() for word in words]

print("#######################################")

#Tokenization of each backstory. 
output_per_backstory = [None]*len(dataset)
sentence = [None]*len(dataset)



###############################    By backstory #################
for i in range(0,len(dataset)):
#    print("############ Index :", i)
#    print(dataset.Backstory[i])
#    print(dataset.Name[i])
    output_per_backstory [i]=Tokenization_per_backstory(dataset.Backstory[i], dataset.Name[i]) 
    print(output_per_backstory [i])
 
###############################    By sentence #################
for i in range(0,len(dataset)):
#    print("############ Index :", i)
#    print(dataset.Backstory[i])
#    print(dataset.Name[i])
    sentence[i]=Tokenization_per_sentence(dataset.Backstory[i], dataset.Name[i]) 
#    print("Sentence i ")
#    print(sentence[i])
    for j in range(0, len(sentence[i])):
        if i==0 and j==0:
#            print(sentence[i][j].split())
            output_per_sentence=[sentence[i][j].split()]
        else: 
            print(sentence[i][j].split())
            output_per_sentence = output_per_sentence+ [sentence[i][j].split()]
#    print("output_per_sentence est ")
#    print(output_per_sentence)      

        
        
        
# end description 

#Save the output it in a file named 'Preprocessing_file'
outfile_per_backstory = open('Preprocessing_per_backstory_file.pkl','wb')
pickle.dump(output_per_backstory,outfile_per_backstory)
outfile_per_backstory.close()

outfile_per_sentence = open('Preprocessing_per_sentence_file.pkl','wb')
pickle.dump(output_per_sentence,outfile_per_sentence)
outfile_per_sentence.close()

