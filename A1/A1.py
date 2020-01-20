import sys 
import re
import math

# Open the file in read mode 
#text = open(sys.argv[1], "r") 

class UnigramModel:
    def __init__(self, text):
        textFile = open(text, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        for line in textFile:
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
        textFile.close()

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

    #Takes a list of sentences and calculates the perplexity of the model given those senetences
    def calcPerplexity(self, sentences):
        logSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)
            sentenceProb = self.calcSentenceProb(sentence)

            #Log is undefined if input is not > 0
            if sentenceProb > 0:
                logProb = math.log(self.calcSentenceProb(sentence), 2)
                logSum += logProb
            #Here we assume Pr = 0, so perplexity is infinite, return -1 to signify this
            else:
                return -1

        #To get perplexity, multiply this sum by the negative reciprocal 
        #of sample size and exponentiate it base 2
        logSum = logSum * (float(-1) / float(sampleSize))
        perplexity = 2 ** logSum

        return perplexity

    #Uses a test file and calculates perplexity based on that sample
    def testModel(self, test):
        testFile = open(test, "r")

        sentences = []

        for line in testFile:
            # append stop token to EOLs
            line = line + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        return self.calcPerplexity(sentences)


# we still need to fill in a function that parses each line in the text
# and implements the unigram language model by calling calc_prob

def main():
    uni = UnigramModel(sys.argv[1])

    #Some tests
    print("Pr(\"sibling\"): " + str(uni.calcTokenProb("sibling")))
    sentence = ["This", "is", "a", "test", ".", "<STOP>"]
    print(uni.calcSentenceProb(sentence))
    print(uni.calcSentenceProb([]))

    print("\n Perplexity: " + str(uni.calcPerplexity([sentence, ["hello", "world"]])))

    print("Files perplexity: " + str(uni.testModel(sys.argv[2])))

if __name__ == '__main__':
    main()

