#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Utils
Zimeng Qiu Sep 2019

Define model utils here

You don't need to modify this file.
"""


def read_file(file):
    """
    Read text from file
    :param file: input file path
    :return: text - text content in input file
    """
    with open(file, 'r') as f:
        text = f.readlines()
    return text


def load_dataset(files):
    """
    Load dataset from file list
    :param files:
    :return: data - text dataset
    """
    data = []
    for file in files:
        data.extend(read_file(file))
    return data


def preprocess(corpus):
    """
    Extremely simple preprocessing
    You can not suppose use a preprocessor this simple in real world
    :param corpus: input text corpus
    :return: tokens - preprocessed text
    """
    tokens = []
    for line in corpus:
        tokens.append([tok.lower() for tok in line.split()])
    return tokens
