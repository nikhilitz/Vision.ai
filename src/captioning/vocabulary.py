import nltk
from collections import Counter
import torch

nltk.download('puntk')

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        Initialize the Vocabulary object.

        :param freq_threshold: Minimum number of times a word must appear to be included in vocab
        """
        self.freq_threshold=freq_threshold

        #special tokens
        self.pad_token='<pad>'
        self.sos_token='<sos>'
        self.eos_token='<eos>'
        self.unk_token='<unk>'

        self.stoi={
            self.pad_token:0,
            self.sos_token:1,
            self.eos_token:2,
            self.unk_token:3
        }

        self.itos={idx:token for token,idx in self.stoi.items()}

        self.word_freq=Counter()

    def __len__(self):
        return len(self.stoi)
    
    def tokenize(self,text):
        return nltk.tokenize.word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list):
        """
        Builds vocabulary from list of cleaned captions.
        Only includes words with freq >= threshold.
        :param sentence_list: list of cleaned captions
        """

        for sentence in sentence_list:
            tokens=self.tokenize(sentence)
            self.word_freq.update(tokens)
        
        idx=len(self.stoi)

        for word,freq in self.word_freq.items():
            if freq>=self.freq_threshold:
                if word not in self.stoi:
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1
    
    def numericalize(self, text):
        """
        Converts a sentence into a list of integers using stoi mapping.
        :param text: cleaned caption
        :return: list of token indices
        """
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token,self.stoi[self.unk_token]) for token in tokenized_text]
    
