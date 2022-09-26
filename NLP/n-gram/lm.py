#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Zimeng Qiu Sep 2019

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse
import math
from itertools import chain
from collections import Counter

from utils import *


class LanguageModel(object):
    """
    Base class for all language models
    """
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # write your initialize code below
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform

        flattened = list(chain.from_iterable(corpus))
        c = Counter(flattened)
        if min_freq > 1:
            self.corpus = [token if c[token] >= min_freq else 'UNK' for token in flattened]
        else:
            self.corpus = flattened
        # self.corpus = flattened
    
        self.vocab_size = 0
        self.unique_words = []
        for token in self.corpus:
            if token not in self.unique_words:
                self.unique_words.append(token)
                self.vocab_size += 1
        
        # print(self.corpus)

        self.build()

    def build(self):
        """
        Build LM from text corpus
        """
        ## Uniform distribution
        if self.uniform is True:
            print("Training Uniform Distribution Model.")
            self.freq_list = [(token, 1) for token in self.unique_words]
            self.prob_list = [(token, 1/self.vocab_size) for token in self.unique_words]
        ## Unigram
        elif self.ngram is 1:
            print("Training Unigram Model.")
            c = Counter(self.corpus)
            self.freq_list = [(token, c[token]) for token in self.unique_words]
            # ## no laplace smoothing
            self.prob_list = [(token, c[token]/len(self.corpus)) for token in self.unique_words]
            # ## laplace smoothing
            # self.prob_list = [(token, (1+c[token])/(self.vocab_size+len(self.corpus))) for token in self.unique_words]
        ## Bigram
        elif self.ngram is 2:
            print("Training Bigram Model.")
            bigrams = [(self.corpus[i], self.corpus[i+1]) for i in range(len(self.corpus)-1)]
            c1 = Counter(self.corpus)
            c2 = Counter(bigrams)
            self.freq_list = [("{} {}".format(w1, w2), c2[(w1, w2)]) for (w1, w2) in list(set(bigrams))]
            ## no laplace smoothing
            self.prob_list = [("{} {}".format(w1, w2), c2[(w1, w2)]/c1[w1]) for (w1, w2) in list(set(bigrams))]
            ## laplace smoothing
            # self.prob_list = [("{} {}".format(w1, w2), (1+c2[(w1, w2)])/(self.vocab_size+c1[w1])) for (w1, w2) in list(set(bigrams))]
        ## Trigram
        elif self.ngram is 3:
            print("Training Trigram Model.")
            bigrams = [(self.corpus[i], self.corpus[i+1]) for i in range(len(self.corpus)-1)]
            trigrams = [(self.corpus[i], self.corpus[i+1], self.corpus[i+2]) for i in range(len(self.corpus)-2)]
            c1 = Counter(self.corpus)
            c2 = Counter(bigrams)
            c3 = Counter(trigrams)
            self.freq_list = [("{} {} {}".format(w1, w2, w3), c3[(w1, w2, w3)]) for (w1, w2, w3) in list(set(trigrams))]
            ## no laplace smoothing
            self.prob_list = [("{} {} {}".format(w1, w2, w3), c3[(w1, w2, w3)]/c2[(w1, w2)]) for (w1, w2, w3) in list(set(trigrams))]
            ## laplace smoothing
            # self.prob_list = [("{} {} {}".format(w1, w2, w3), (1+c3[(w1, w2, w3)])/(self.vocab_size+c2[(w1, w2)])) for (w1, w2, w3) in list(set(trigrams))]
        # print(self.freq_list)
        self.prob_dict = dict(self.prob_list)
            
    def most_common_words(self, k):
        """
        Return the top-k most frequent n-grams and their frequencies in sorted order.
        For uniform models, the frequency should be "1" for each token.

        Your return should be sorted in descending order of frequency.
        Sort according to ascending alphabet order when multiple words have same frequency.
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        # Write your own implementation here
        sorted_freq_list = sorted(self.freq_list, key = lambda x: (-x[1], x[0]))
        return sorted_freq_list[:k]

    def count(self):
        return len(self.freq_list)

def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # Write your own implementation here
    pp = 1
    flattened = flattened = list(chain.from_iterable(data))
    adjusted = [token if token in models[0].unique_words else 'UNK' for token in flattened]
    # c = Counter(adjusted)
    # if models[0].min_freq > 1:
    #     adjusted = [token if c[token] >= models[0].min_freq else 'UNK' for token in adjusted]
    for n in range(len(adjusted)):
        interpolated = 0
        for i in range(len(models)):
            def getT(k):
                if models[k].ngram is 1:
                    return adjusted[n]
                elif models[k].ngram is 2:
                    return "{} {}".format(adjusted[n-1], adjusted[n]) if n > 0 else None
                elif models[k].ngram is 3:
                    return "{} {} {}".format(adjusted[n-2], adjusted[n-1], adjusted[n]) if n > 1 else None
                return None
            
            t = getT(i)
            prob = 0
            ## backoff
            if t in models[i].prob_dict:
                prob = models[i].prob_dict[t]
            if prob is 0 and i > 0 and getT(i-1) in models[i-1].prob_dict:
                prob = models[i-1].prob_dict[getT(i-1)]
            if prob is 0 and i > 1 and getT(i-2) in models[i-2].prob_dict:
                prob = models[i-2].prob_dict[getT(i-2)]
            interpolated += coefs[i]*prob
        pp = pp * math.exp(-math.log(interpolated) / len(adjusted))
    return pp


# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    return parser.parse_args()


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)
    print('Uniform counts: %s' % uniform.count())
    print('Uniform top words: %s' % uniform.most_common_words(10))
    print('Unigram counts: %s' % unigram.count())
    print('Unigram top words: %s' % unigram.most_common_words(30))
    print('Bigram counts: %s' % bigram.count())
    print('Bigram top words: %s' % bigram.most_common_words(200))
    print('Trigram counts: %s' % trigram.count())
    print('Trigram top words: %s' % trigram.most_common_words(70))


    # # calculate perplexity on test file
    # ppl = calculate_perplexity(
    #     models=[uniform, unigram, bigram, trigram],
    #     coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
    #     data=test)

    # print("Perplexity: {}".format(ppl))
