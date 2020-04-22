# Some preprocesssing that will be common to all the text classification methods you will see. 

import pandas as pd
import re
import string
import nltk
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize


dataset = pd.read_excel(r'C:\Users\elodi\dox\Documents\Doctorat\Formation doctorale\Cours suivis\Deep learning\Projet Deep Learning\dnd-backgrounds-generator\Data\dd_bios.xls',encoding='latin1')
print(dataset)
dataset.columns = ['Time', 'Name', 'Race', 'Class', 'Backstory']

#Delete rows with missing elements
dataset=dataset.dropna()
dataset=dataset.reset_index(drop=True)
print(dataset)

#def remove_numbers(str):
#    string_no_numbers = re.sub("\d+", "_Number_", str)
#    return string_no_numbers
##
#def remove_line(str):
#    string_= re.sub("\n", " ", str)
#    string_= re.sub("\r", " ", string_)
#    return string_
#
#
#def strip_html_tags(text):
#    """remove html tags from text"""
#    soup = BeautifulSoup(text, "html.parser")
#    stripped_text = soup.get_text(separator=" ")
#    return stripped_text
#
#def remove_website(text):
#    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#    return text
#
#def remove_whitespace(text):
#    """remove extra whitespaces from text"""
#    text = text.strip()
#    return " ".join(text.split())
#
#
#def text_preprocessing(text):
#    """preprocess text with default option set to true for all steps"""
#    text= remove_website(text)
#    text = strip_html_tags(text)
#    text = text.lower()
#    text=remove_punctuations(text)
#    text=remove_numbers(text)
#    text=remove_line(text)
#    text = remove_whitespace(text)
#    return text
#
#def unique(list1): 
#    x = np.array(list1) 
#    print(np.unique(x)) 
#
#



###########
def remove_punct(text):
    text = text.replace("“", "")
    text = text.replace("”", "")
    words = text.split()
    remove=string.punctuation
    remove = remove.replace("-", "")
    table = str.maketrans('', '', remove)
    stripped = [w.translate(table) for w in words]
    return stripped



def Before_Token(text, name):
    text = text.replace("?", ".")
    text = text.replace("!", ".")
    text = text.replace(".", " .")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = re.sub("\d+", "_Number_", text)
    if name in text: 
        text = re.sub(name, "_Name_", text)
    else: 
        name=remove_punct(name)
        for i in range(0, len(name)):
            text = re.sub(name[i], "_Name_", text)
    return text
    
def Tokenization(text, name):
   text= Before_Token(text, name)
   words = text.split()
   return [word.lower() for word in words]

print("#######################################")




output = [None]*len(dataset)
#print(output)
for i in range(0,len(dataset)):
    print("############ Index :", i)
    print(dataset.Backstory[i])
    print(dataset.Name[i])
    output[i]=Tokenization(dataset.Backstory[i], dataset.Name[i]) 
    print(output[i])
 
