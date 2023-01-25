import re
import pandas as pd
import re
import json
from collections import Counter
import matplotlib.pyplot as plt


def plot_word_counts(series_data):
    my_dict = dict(Counter([len(j) for j in series_data.str.split()]))
    sorted_list = sorted(my_dict.items())
    counts = {str(i):j for i,j in sorted_list}
    x,y =counts.keys(),counts.values()
    plt.figure(figsize = (20, 10))
    plt.bar(x,y)
    plt.title("Counts of number of words in sentences")
    plt.xlabel("lenghts")
    plt.ylabel("Counts")
    plt.savefig("/home/Ravikumar/Developer/text_generation/results/"+'plot_word_counts.png')
    plt.show()



def trim_sentence(sentences, trim_size=100):
    trim = list()
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if sentence_len >= trim_size:
            trim.append(" ".join(sentence.split()[:trim_size]))
        else:
            # sen = sentence.split() + (["<PAD>"]* trim_size-len(sentence))
            trim.append(sentence)
    return trim



def shift_word(sequences,params):
    inputs = list()
    outputs = list()
    for _,s in enumerate(sequences):
        for i in range(len(s)-params['SEQ_LEN']):
            inp = s[i:params['SEQ_LEN']+i]
            out =s[i+1:params['SEQ_LEN']+i+1]
            inputs.extend([inp])
            outputs.extend([out])
    return inputs,outputs

