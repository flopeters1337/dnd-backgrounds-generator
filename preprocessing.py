###################
#Import libraries
import pandas as pd
import re
import string
#import nltk
#from nltk.tokenize import word_tokenize
import pickle

#Import dataset 
dataset = pd.read_excel(r'C:\Users\elodi\dox\Documents\Doctorat\Formation doctorale\Cours suivis\Deep learning\Projet Deep Learning\dnd-backgrounds-generator\Data\dd_bios.xls',encoding='latin1')
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
    #-adds "_end_" at the end of the text
#The inputs are two strings text and name. 
def Before_Token(text, name):
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
    text = text + " _end_"
    return text
    
#Tokenization, words by word, of the string text by keeping the point. 
#First this function applies the function Before_Token. 
#The inputs are two strings text and name.
def Tokenization(text, name):
   text= Before_Token(text, name)
   words = text.split()
   return [word.lower() for word in words]

print("#######################################")

#Tokenization of each backstory. 
output = [None]*len(dataset)


#print(output)
#for i in range(2430,len(dataset)):
for i in range(0,len(dataset)):
    print("############ Index :", i)
    print(dataset.Backstory[i])
    print(dataset.Name[i])
    output[i]=Tokenization(dataset.Backstory[i], dataset.Name[i]) 
    print(output[i])
 
# end description 

#Save the output it in a file named 'Preprocessing_file'
outfile = open('Preprocessing_file.pkl','wb')
pickle.dump(output,outfile)
outfile.close()

