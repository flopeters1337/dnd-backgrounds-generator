# Implementation of the Preprocessor class, prepossessing the dataset to allow it to be fed to the training algorithm

import pandas as pd
import re
import string
import torch
import torch.nn as nn


class Preprocessor():
    def __init__(self):

        """
        Constructor
        """
        self.dataset = None  # panda dataframe
        self.vocabulary = None  # dictionary of strings with integer as indexes
        self.names = None  # list of strings

        self.sentences_original = None  # list of strings
        self.sentences_tokened = None  # list of (list of strings)
        self.sentences_indexed = None  # list of (list of integers)

        self.descriptions_original = None  # list of strings
        self.descriptions_tokened = None  # list of (list of strings)
        self.descriptions_indexed = None  # list of (list of integers)

    def load_dataset(self, name):

        """
        Function loading the dataset
        :param name: (string) name of the excel file containing the dataset
        """

        dataset = pd.read_excel(name, encoding='latin1')
        dataset.columns = ['Time', 'Name', 'Race', 'Class', 'Backstory']
        dataset = dataset[['Name', 'Backstory']]
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)
        self.dataset = dataset
        self.descriptions_original = self.dataset[['Backstory']].values.tolist()
        self.names = self.dataset[['Name']].values.tolist()

    def preprocess(self, min_sentences=4, max_sentences=20, min_descriptions=20, max_descriptions=200):

        """
        Main prepossessing function
        :param min_sentences : (int) minimum size of the sentences to not be removed
        :param max_sentences : (int) minimum size of the sentences to not be splitted
        :param min_descriptions : (int) minimum size of the background stories to not be removed
        :param max_descriptions : (int) minimum size of the background to not be splitted
        """

        self.tokenize_dataset()

        self.create_vocabulary()
        self.vocabulary_mapping()

        output_sentences = self.size_filtering(self.sentences_indexed, min_sentences, max_sentences)
        output_descriptions = self.size_filtering(self.descriptions_indexed, min_descriptions, max_descriptions)

        output_sentences = self.tensors_conversion(output_sentences)
        output_descriptions = self.tensors_conversion(output_descriptions)

        return output_sentences, output_descriptions

    def tokenize_dataset(self):

        """
        Tokenization function
        """

        descriptions_output = []
        sentences_output = []
        for i in range(0, len(self.dataset)):
            descriptions_output.append(
                self.tokenization_backstory(self.dataset['Backstory'][i], self.dataset['Name'][i]))
            sentence_temp = self.tokenization_sentences(self.dataset['Backstory'][i], self.dataset['Name'][i])
            sentence_temp = [l.split() for l in sentence_temp]
            sentences_output.append(sentence_temp)

        self.descriptions_tokened = descriptions_output
        self.sentences_tokened = [item for elem in sentences_output for item in elem]

    def create_vocabulary(self):

        """
        Function creating the vocabulary with every unique word being given a unique index
        """

        list_of_words1 = [item for elem in self.descriptions_tokened for item in elem]
        list_of_words2 = [item for elem in self.sentences_tokened for item in elem]
        list_of_words = [y for x in [list_of_words1, list_of_words2] for y in x]
        list_of_unique_words = list(set(list_of_words))
        vocabulary = {word: list_of_unique_words.index(word) for word in list_of_unique_words}
        self.vocabulary = vocabulary

    def vocabulary_mapping(self):

        """
        Function mapping every word of every background story/sentence to its unique index
        """

        outputs = []
        for d in self.descriptions_tokened:
            outputs.append([self.vocabulary[word] for word in d])
        self.descriptions_indexed = outputs

        outputs = []
        for s in self.sentences_tokened:
            outputs.append([self.vocabulary[word] for word in s])
        self.sentences_indexed = outputs

    def size_filtering(self, list_of_list, size_min, size_max):

        """
        Function removing strings that are too short and splitting string that are too long
        :param list_of_list : (list of list of items) list of sentences or background stories
        :param size_min : (int) minimum size to not be removed
        :param size_max : (int) maximum size to not be splitted
        """

        output = [l for l in list_of_list if size_min <= len(l) <= size_max]
        list_size_too_long = [l for l in list_of_list if len(l) > size_max]
        output_splitted = [self.split_list(l, (len(l) // size_max) + 1) for l in list_size_too_long]
        output_splitted = [item for elem in output_splitted for item in elem]
        output.extend(output_splitted)
        return output

    # Split too long lists into two or more shorter lists
    def split_list(self, input_list, n_breaks):

        """
        Function performing the splitting
        :param input_list : (list of items) list to be splitted
        :param n_breaks : (int) number of breaks to perform
        """

        length = len(input_list)
        splitted_list = [input_list[i * length // n_breaks:(i + 1) * length // n_breaks] for i in range(n_breaks)]
        return splitted_list

    def tensors_conversion(self, x):

        """
        Convert the output of the prepossessing function to a tensor
        :param x : (list of list of int)
        :return output: (tensor) of dimension len(x) * max (len l for l in x)
        """

        output = list(map(torch.LongTensor, x))
        output = nn.utils.rnn.pad_sequence(output, padding_value=len(self.vocabulary))
        output = torch.transpose(output, 0, 1)
        return output

    def tokenization_backstory(self, text, name):

        """
        Tokenization, words by word, of the string text by keeping the point.
        :param text : string to tokenize
        :param name: name of the character which background story is being processed
        :return : list of strings after tokenization
        """

        text = self.pre_tokenization(text, name, 1)
        words = text.split()
        return [word.lower() for word in words]

    def tokenization_sentences(self, text, name):

        """
        Tokenization, sentence by sentence, of the string text
        :param text : string to tokenize
        :param name: name of the character which background story is being processed
        :return : list of strings after tokenization
        """

        text = self.pre_tokenization(text, name, 0)
        words = text.split('.')
        return [word.lower() for word in words]

    def pre_tokenization(self, text, name, end):

        """
        Function doing some prepossessing work in a text (before tokenization)
        :param text : string to pre-tokenize
        :param name: name of the character which background story is being processed
        :param end : (int) either 0 or 1
        :return : (string) text after some text prepossessing
        """

        text = text.replace("?", ".")
        text = text.replace("!", ".")
        text = text.replace(".", " .")
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = re.sub(r"[^a-zA-Z0-9\.]+", ' ', text)
        text = re.sub("\d+", "_Number_", text)

        if name in text:  # if the string name is in the text
            text = re.sub("(^| )" + name + "( |$)", " _Name_ ", text)  # replace name by _Name_
        else:  # if not, look if one part of the string is in the text and if yes, replace it by _Name_
            name = self.remove_punctuation(name)
            for i in range(0, len(name)):
                text = re.sub("(^| )" + name[i] + "( |$)", " _Name_ ", text)
        if end == 1:
            text = text + " _end_"
        return text

    def remove_punctuation(self, text):

        """
        Function removing punctuation of a text
        :param text : string to pre-tokenize
        :return : (string) text after some text prepossessing
        """

        text = text.replace("“", "")
        text = text.replace("”", "")
        words = text.split()
        remove = string.punctuation
        remove = remove.replace("-", "")
        table = str.maketrans('', '', remove)
        stripped = [w.translate(table) for w in words]
        return stripped
