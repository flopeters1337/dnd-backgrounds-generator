# Regroup tokenization backstory/sentence
# Remove all the intermediate useless values (make the functions return values and use them as temp features in preprocess
# Put "load dataset" in the constructor
# Comment everything nicely
# Update functions with the correction of ELodie to make sure dictionnary is the same everywhere 



import pandas as pd
import re
import string
import nltk
import torch
import torch.nn as nn

# Preprocessor class returning tensor input for the model
class Preprocessor():
    def __init__(self):
        self.dataset = None # panda dataframe : TODO : faire que ce soit le constructeur qui load le dataset (paramètre)
        self.vocabulary = None # dictionnary of strings with integer indexes
        
        self.sentences_original = None  # list of strings NOT USED FOR NOW 
        self.sentences_tokened = None # list of (list of strings)
        self.sentences_indexed = None # list of (list of integers)
        
        self.descriptions_original = None # list of strings 
        self.descriptions_tokened = None # list of (list of strings)
        self.descriptions_indexed = None # list of (list of integers)
    
    # Function loading the dataset 
    def load_dataset(self, name):
        dataset = pd.read_excel(name, encoding='latin1')
        dataset.columns = ['Time', 'Name', 'Race', 'Class', 'Backstory']
        dataset = dataset[['Name', 'Backstory']]
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)
        self.dataset = dataset
        self.descriptions_original = self.dataset[['Backstory']].values.tolist()
        self.names = self.dataset[['Name']].values.tolist() #USELESS
    
    # Main function outputing the sentences and the descriptions preprocessed. 
    def preprocess(self, min_sentences=4, max_sentences=20, min_descriptions=20, max_descriptions=200):
        self.tokenize_dataset()
        
        self.create_vocabulary()
        self.vocabulary_mapping()
        
        output_sentences = self.size_filtering(self.sentences_indexed, min_sentences, max_sentences)
        output_descriptions = self.size_filtering(self.descriptions_indexed, min_descriptions, max_descriptions)
        
        output_sentences = self.tensors_conversion(output_sentences)
        output_descriptions = self.tensors_conversion(output_descriptions)
        
        return output_sentences, output_descriptions
    
    # Function tokenizing the dataset into backstory and sentences 
    def tokenize_dataset(self):
        descriptions_output = []
        sentences_output = []
        for i in range(0, len(self.dataset)):
            descriptions_output.append(self.tokenization_backstory(self.dataset['Backstory'][i], self.dataset['Name'][i]))
            sentence_temp = self.tokenization_sentences(self.dataset['Backstory'][i], self.dataset['Name'][i])
            sentence_temp = [l.split() for l in sentence_temp]  # Si on veut sentences original c'est ici
            sentences_output.append(sentence_temp)
            
        self.descriptions_tokened = descriptions_output
        self.sentences_tokened = [item for elem in sentences_output for item in elem]
    
    # Function creating the vocabulary with every unique word being given a unique index
    def create_vocabulary(self, backstory = True):
        list_of_words1 = [item for elem in self.descriptions_tokened for item in elem]
        list_of_words2 = [item for elem in self.sentences_tokened for item in elem]
        list_of_words = [y for x in [list_of_words1, list_of_words2] for y in x] # NOT OPTIMAL BUT FOR NOW
        list_of_unique_words = list(set(list_of_words))
        vocabulary = {word:list_of_unique_words.index(word) for word in list_of_unique_words}
        self.vocabulary = vocabulary
    
    # Mapping every word of every description/sentence to its unique index
    def vocabulary_mapping(self):
        outputs = []
        for d in self.descriptions_tokened:
            outputs.append([self.vocabulary[word] for word in d])
        self.descriptions_indexed = outputs
        
        outputs = []
        for s in self.sentences_tokened:
            outputs.append([self.vocabulary[word] for word in s])
        self.sentences_indexed = outputs
    
    # Remove sentences that are too short (and probably noisy) and split too long sentences
    def size_filtering(self, list_of_list, size_min, size_max):
        output = [l for l in list_of_list if size_min <= len(l) <= size_max]
        list_size_too_long = [l for l in list_of_list if len(l) > size_max]
        output_splitted = [self.split_list(l,(len(l)//size_max)+1) for l in list_size_too_long]
        output_splitted = [item for elem in output_splitted for item in elem]
        output.extend(output_splitted)
        return output

    # Split too long lists into two or more shorter lists
    def split_list(self, input_list, n_breaks):
        length = len(input_list)
        splitted_list = [input_list[i*length//n_breaks:(i+1)*length//n_breaks] for i in range(n_breaks)]
        return splitted_list
    
    # Return a tensor with all the descriptions/sentences padded so that it has the same length
    # TODO : return a dataset object instead of a tensor 
    def tensors_conversion(self,x):
        output = list(map(torch.LongTensor, x))
        output = nn.utils.rnn.pad_sequence(output, padding_value=len(self.vocabulary))
        output = torch.transpose(output,0,1)
        return output
        
    ###################### Pre-processing  #####################################    
    
    #Tokenization, words by word, of the string text by keeping the point. 
    #First this function applies the function pre_tokenization. 
    def tokenization_backstory(self, text, name):
        text = self.pre_tokenization(text, name, 1)
        words = text.split()
        return [word.lower() for word in words]
        
    #Tokenization, sentence by sentence, of the string text
    #First this function applies the function pre_tokenization. 
    def tokenization_sentences(self, text, name):
        text = self.pre_tokenization(text, name, 0)
        words = text.split('.')
        return [word.lower() for word in words]

    #Function to apply before the tokenization 
    #This function 
    #-replaces interrogation and exclamation points by points in the text
    #-removes quotation marks in the text
    #-replaces every number by the sring "_Number_" in the text
    #-replaces  the string name in the text by "_Name_"
    #-removes all non alpha-numeric characters except the point
    #-adds "_end_" at the end of the text if the option end is activated 
    #The inputs are two strings text and name and an option end (either 0 or 1)
    def pre_tokenization(self, text, name, end):
        text = text.replace("?", ".")
        text = text.replace("!", ".")
        text = text.replace(".", " .")
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = re.sub(r"[^a-zA-Z0-9\.]+", ' ', text)
        text = re.sub("\d+", "_Number_", text)
        
        if name in text: #if the string name is in the text
            text = re.sub("(^| )" + name + "( |$)", " _Name_ ", text)# replace name by _Name_
        else:  #if not, look if one part of the string is in the text and if yes, replace it by _Name_
            name = self.remove_punctuation(name)
            for i in range(0, len(name)):
                text = re.sub("(^| )" + name[i] + "( |$)", " _Name_ ", text)
        if end == 1:
            text = text + " _end_"
        return text
    

    #Tokenization, words by word, of the string text. All the punctuations, except the hyphen, are removed.
    #This function will be applied to replace all the occurences of the character name in the backstory. 
    def remove_punctuation(self, text):
        text = text.replace("“", "")
        text = text.replace("”", "")
        words = text.split()
        remove = string.punctuation
        remove = remove.replace("-", "")
        table = str.maketrans('', '', remove)
        stripped = [w.translate(table) for w in words]
        return stripped
