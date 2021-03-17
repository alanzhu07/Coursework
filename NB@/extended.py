import sys
import io
import math
from collections import Counter

class ExtendedNaiveBayes(object):

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        self.punctuations = ['.', ',', '!', '?', ':', ';', '-', '--', '...']

        self.lines = trainingData.splitlines()
        self.classes = {}
        self.classes_count = {}
        self.punct_count = {}
        for line in self.lines:
            toks = line.split()
            clas = toks[0]
            if clas not in self.classes:
                self.classes[clas] = []
            if clas not in self.punct_count:
                self.punct_count[clas] = 0
            for tok in toks[1:]:
                self.classes[clas].append(tok)
                if tok in self.punctuations:
                    self.punct_count[clas] += 1

        self.unique_words = []
        for clas in self.classes:
            self.classes_count[clas] = len(self.classes[clas])
            for token in self.classes[clas]:
                if token not in self.unique_words:
                    self.unique_words.append(token)
        self.vocab_size = len(self.unique_words)
        self.total_count = self.classes_count['RED'] + self.classes_count['BLUE']
        self.total_punct_count = self.punct_count['RED'] + self.punct_count['BLUE']

        
        # print(self.classes["RED"])
        # print(self.classes["BLUE"])
        # print(self.vocab_size)



    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        words = sentence.split()
        log_prob_dict = {'red': 0, 'blue': 0}
        for clas in self.classes:
            c = Counter(self.classes[clas])
            prob = 0
            for word in words:
                wc = c[word]
                log_prob = math.log(wc + 1) - math.log(self.classes_count[clas] + self.vocab_size)
                if word in self.punctuations:
                    punct_prob = math.log(self.punct_count[clas] + 1) - math.log(self.classes_count[clas] + self.vocab_size)
                else:
                    punct_prob = math.log(self.classes_count[clas] - self.punct_count[clas] + 1) - math.log(self.classes_count[clas] + self.vocab_size)
                prob = prob + log_prob + punct_prob
            p_c = math.log(self.classes_count[clas]) - math.log(self.total_count)
            prob += p_c
            if clas == 'RED':
                log_prob_dict['red'] = prob
            elif clas == 'BLUE':
                log_prob_dict['blue'] = prob

        return log_prob_dict

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        ## red - positive, blue - negative
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for line in testData.splitlines():
            toks = line.split()
            # print(toks[0])
            correct_class = toks[0]
            estimated_log_dict = self.estimateLogProbability(" ".join(toks[1:]))
            # print(estimated_log_dict)
            predicted_class = 'RED' if estimated_log_dict['red'] > estimated_log_dict['blue'] else 'BLUE'
            # print('Estimate:{}, Correct:{}'.format(predicted_class, correct_class))
            if correct_class == 'RED':
                if predicted_class == 'RED':
                    tp += 1
                else:
                    fn += 1
            elif correct_class == 'BLUE':
                if predicted_class == 'BLUE':
                    tn += 1
                else:
                    fp += 1
        print("tp:{}, tn:{}, fp:{}, fn:{}".format(tp, tn, fp, fn))
        return {'overall accuracy': (tp + tn)/(tp + tn + fp + fn),
                'precision for red': tp/(tp + fp),
                'precision for blue': tn/(tn + fn),
                'recall for red': tp/(tp + fn),
                'recall for blue': tn/(tn + fp)}

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 extended.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = ExtendedNaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))



