import sys 
import re

# Open the file in read mode 
text = open(sys.argv[1], "r") 

class UnigramModel:
    def __init__(self, text):
        self.freq_dict = dict()
        self.corpusSize = 0
        for line in text:
            # append stop token to EOLs
            line = line + ' <STOP>'
            # split the line into tokens
            tokens = line.split(" ")
            # iterate through each token
            for token in tokens:
                self.corpusSize += 1
                # remove whitespace and \n characters
                token = token.strip()
                # if token is already in dictionary then increment the count
                if token in self.freq_dict:
                    self.freq_dict[token] = self.freq_dict[token] + 1
                # else initialize that token's key with value 1
                else:
                    self.freq_dict[token] = 1

        self.freq_dict = self.replaceUNKs(self.freq_dict)

    # this function replaces all keys with < 3 frequency with 'unk'
    def replaceUNKs(self, idict):
        fdict = {k:v for (k,v) in idict.items() if v >= 3}
        fdict['unk'] = 0
        for k, v in idict.items():
            if(v < 3): fdict['unk'] = fdict['unk'] + idict[k]
        return fdict

    def calcTokenProb(self, token):
        #The numerator is the frequency of the specified token, return 0 if not found in dict
        unigram_numerator = self.freq_dict.get(token, 0)
        #The denominator is just the size of the corpus
        unigram_denominator = self.corpusSize
        print("Pr(" + token + "):")
        print("numerator = ", unigram_numerator, "denominator = ", unigram_denominator)
        return float(unigram_numerator) / float(unigram_denominator)

    def calcSentenceProb(self, sentence):
        print("Pr(" + str(sentence) + ")")

        #Return 0 for empty sentence
        if not sentence:
            return 0

        prob = 1
        #The MLE for a sentence is the product of the MLEs of its constituent tokens
        for token in sentence:
            prob = prob * self.calcTokenProb(token)

        return prob

# we still need to fill in a function that parses each line in the text
# and implements the unigram language model by calling calc_prob

def main():
    uni = UnigramModel(text)

    #Some tests
    print("Pr(\"sibling\"): " + str(uni.calcTokenProb("sibling")))
    sentence = ["This", "is", "a", "test", ".", "<STOP>"]
    print(uni.calcSentenceProb(sentence))
    print(uni.calcSentenceProb([]))

if __name__ == '__main__':
    main()

