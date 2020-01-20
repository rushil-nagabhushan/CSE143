import sys 
import re

# Open the file in read mode 
text = open(sys.argv[1], "r") 

class UnigramModel:
    def __init__(self, text):
        self.freq_dict = dict()
        for line in text:
            # append stop token to EOLs
            line = line + ' <STOP>'
            # split the line into tokens
            tokens = line.split(" ")
            # iterate through each token
            for token in tokens:
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

    def calc_prob(self, token):
        unigram_numerator = self.freq_dict.get(token, 0)
        unigram_denominator = len(self.freq_dict)
        print("numerator = ", unigram_numerator, "denominator = ", unigram_denominator)
        return float(unigram_numerator) / float(unigram_denominator)

# we still need to fill in a function that parses each line in the text
# and implements the unigram language model by calling calc_prob

def main():
    uni = UnigramModel(text)

if __name__ == '__main__':
    main()

