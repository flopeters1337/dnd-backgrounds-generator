# Some preprocesssing that will be common to all the text classification methods you will see. 

import csv
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n

import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
#call the nltk downloader
# import these modules 
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 


def remove_punctuations(x): #Replace punctuation by space 
    pattern = r'[^a-zA-z0-9\s]'
    x= re.sub(pattern, ' ', x)
    return x

def anonymous(x):
    pattern = r'[^a-zA-z0-9\s]'
    x= re.sub(pattern, ' ', x)
    return x

def remove_numbers(str):
    string_no_numbers = re.sub("\d+", "#", str)
    return string_no_numbers

def remove_line(str):
    string_= re.sub("\n", " ", str)
    string_= re.sub("\r", " ", string_)
    return string_


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_website(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text

def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def text_preprocessing(text):
    """preprocess text with default option set to true for all steps"""
    text= remove_website(text)
    text = strip_html_tags(text)
    text = text.lower()
    text=remove_punctuations(text)
    text=remove_numbers(text)
    text=remove_line(text)
    text = remove_whitespace(text)
    return text

def unique(list1): 
    x = np.array(list1) 
    print(np.unique(x)) 


dataset = pd.read_excel(r'C:\Users\elodi\dox\Documents\Doctorat\Formation doctorale\Cours suivis\Deep learning\Projet Deep Learning\dnd-backgrounds-generator\Data\dd_bios.xls',encoding='latin1')
print(dataset)
dataset.columns = ['Time', 'Name', 'Race', 'Class', 'Backstory']
dataset= dataset.dropna() #remove nan 

Name=dataset.Name.tolist()
Name = list(map(text_preprocessing, Name))

#print(remove_website("Coucou Elodie voici le site : https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python"))
Backstory=dataset.Backstory.tolist()
print(len(Backstory))
Backstory = list(map(text_preprocessing, Backstory))
        
print(Backstory[0])

for i in range(0, 10):
    print(Name[i])
#    Backstory[i].replace(Name[i],i)
#
#for i in range(0,10):
#    re.sub(Name[i], , str)

# split into words

#tokens = word_tokenize(dataset.Backstory[0])
## convert to lower case
#tokens = [w.lower() for w in tokens]
## remove punctuation from each word
#import string
#table = str.maketrans('', '', string.punctuation)
#stripped = [w.translate(table) for w in tokens]
## remove remaining tokens that are not alphabetic
#words = [word for word in stripped if word.isalpha()]
## filter out stop words
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#words = [w for w in words if not w in stop_words]
#print(words[:100])
#print(dataset.Backstory[0])