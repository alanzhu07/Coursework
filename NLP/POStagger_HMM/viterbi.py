'''
Yihui Peng
Ze Xuan Ong
Jocelyn Huang
Noah A. Smith
Yifan Xu

Usage: python viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

Apart from writing the output to a file, the program also prints
the number of text lines read and processed, and the time taken
for the entire program to run in seconds. This may be useful to
let you know how much time you have to get a coffee in subsequent
iterations.
'''

import math
import sys
import time

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string


class Viterbi():
    def __init__(self):
        # transition and emission probabilities. Remember that we're not dealing with smoothing 
        # here. So for the probability of transition and emission of tokens/tags that we haven't 
        # seen in the training set, we ignore thm by setting the probability an impossible value 
        # of 1.0 (1.0 is impossible because we're in log space)

        self.transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.emission = defaultdict(lambda: defaultdict(lambda: 1.0))
        # keep track of states to iterate over 
        self.states = set()
        self.POSStates = set()
        # store vocab to check for OOV words
        self.vocab = set()

        # text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into LOG SPACE!
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state)
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)


    # run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")

    # TODO: Implement this

    # run Viterbi algorithm on a line of text 
    # Input: A string representing a sequence of tokens separated by white spaces 
    # Output: A string representing a sequence of POS tags.

    # Things to keep in mind:
    # 1. Probability calculations are done in log space. 
    # 2. Ignore smoothing in this case. For  probabilities of emissions that we haven't seen
    # or  probabilities of transitions that we haven't seen, ignore them. (How to detect them?
    # Remember that values of self.transition and self.emission are default dicts with default 
    # value 1.0!)
    # 3. A word is treated as an OOV word if it has not been seen in the training set. Notice 
    # that an unseen token and an unseen transition/emission are different things. You don't 
    # have to do any additional thing to handle OOV words.
    # 4. There might be cases where your algorithm cannot proceed. (For example, you reach a 
    #     state that for all prevstate, the transition probability prevstate->state is unseen)
    #     Just return an empty string in this case. 
    # 5. You can set up your Viterbi matrix (but we all know it's better to implement it with a 
    #     python dictionary amirite) in many ways. For example, you will want to keep track of 
    #     the best prevstate that leads to the current state in order to backtrack and find the 
    #     best sequence of pos tags. Do you keep track of it in V or do you keep track of it 
    #     separately? Up to you!
    # 6. Don't forget to handle final state!
    # 7. If you are reading this during spring break, mayyyyybe consider taking a break from NLP 
    # for a bit ;)

    def viterbiLine(self, line):
        words = line.split()

        # TODO: Initialize DP matrix for Viterbi here
        V = [defaultdict(lambda: 0) for s in range(len(words)+1)]
        
        # V[0][INIT_STATE] = {'p': 1, 'path': None}

        for (i, word) in enumerate(words):
            # replace unseen words as oov
            if word not in self.vocab:
                word = OOV_WORD

            # TODO: Fill up your DP matrix
            if i is 0:
                for new_state in self.POSStates:
                    if self.transition[INIT_STATE][new_state] == 1.0 or self.emission[new_state][word] == 1.0:
                        V[i][new_state] = {'p': 1.0, 'path': None}
                    else:
                        V[i][new_state] = {'p': self.transition[INIT_STATE][new_state] + self.emission[new_state][word], 'path': None}
                    # print('{} {} {} {}'.format(i, new_state, V[i][new_state]['p'], V[i][new_state]['path']))
            else:
                for new_state in self.POSStates:
                    best_prob = None
                    best_path = None
                    for prev_state in self.POSStates:
                        if V[i-1][prev_state]['p'] == 1.0 or self.transition[prev_state][new_state] == 1.0 or self.emission[new_state][word] == 1.0:
                            prob = None
                            # print(prev_state + '!')
                        else:
                            prob = V[i-1][prev_state]['p'] + self.transition[prev_state][new_state] + self.emission[new_state][word]
                            # print('{} {}'.format(prev_state, prob))
                            if best_prob is None or prob > best_prob:
                                best_prob = prob
                                best_path = prev_state
                    if best_prob is None:
                        V[i][new_state] = {'p': 1.0, 'path': None}
                    else:
                        V[i][new_state] = {'p': best_prob, 'path': best_path}
                    # print('{} {} {} {}'.format(i, new_state, V[i][new_state]['p'], V[i][new_state]['path']))


        # TODO: Handle best final state
        final_prob = None
        final_path = None
        for prev_state in self.POSStates:
            if V[len(words)-1][prev_state]['p'] == 1.0 or self.transition[prev_state][FINAL_STATE] == 1.0:
                prob = None
            else:
                prob = V[len(words)-1][prev_state]['p'] + self.transition[prev_state][FINAL_STATE]
                if final_prob is None or prob > final_prob:
                    final_prob = prob
                    final_path = prev_state
        if final_prob is None:
            return ''
        l = len(words)
        # print('{} {} {} {}'.format(l, FINAL_STATE, V[l][FINAL_STATE]['p'], V[l][FINAL_STATE]['path']))
        V[len(words)][FINAL_STATE] = {'p': final_prob, 'path': final_path}

        # TODO: Backtrack and find the optimal sequence. 
        sequence = ['' for i in range(len(words))]
        i = len(words)
        latest = FINAL_STATE
        while i > 0:
            sequence[i-1] = V[i][latest]['path']
            latest = sequence[i-1]
            i = i-1
    

        return ' '.join(sequence)

if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))

