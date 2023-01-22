import glob
import pandas as pd
import re
import config
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import nltk
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import pickle

def get_all_file_path(path,file_format = "*.csv"):
    dir = path + file_format
    csv_file_names = glob.glob(dir)
    return csv_file_names

def create_dataframe(file_names):
    df = list()
    for file_path in file_names:
        d = pd.read_csv(file_path)
        df.append(d)
    return pd.concat(df)

def clean_text(text):
    # Remove puncuation,stopwords,only words,lowercase,lematization
    text = re.sub(r'(\#\w+)'," ",str(text)) # note : nan is float
    text = re.sub(r"br","",text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('[^a-zA-Z]+',' ',text)
    text = " ".join([wordnet_lemmatizer.lemmatize(x) for x in text.split() ])
    text = text.lower()
    return text.strip()    

def trim_sentence(sentences,params):
    trim = list()
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if sentence_len >= params["MAX_SEQ_LEN"]:
            trim.append(" ".join(sentence.split()[:params["MAX_SEQ_LEN"]]))
        else:
            trim.append(sentence)
    return trim

def create_vocab(lines,params):
    all_words = [word for line in lines for word in line.split()]
    unique_words  = sorted(list(set(all_words)))

    n_to_char = {n:char for n, char in enumerate(unique_words)}
    char_to_n = {char:n for n, char in n_to_char.items()}

    vocab_size =  len(n_to_char)

    vocabed = {
        "word2index":char_to_n,
        "index2word":n_to_char,
        "vocab_len" : vocab_size
    }
    json_obj = json.dumps(vocabed)
    with open(params["DATA_DIR"]+"vocab.json", "w") as outfile:
        outfile.write(json_obj)
    return f"vocab created to {params['DATA_DIR']+'vocab.json'}"

def load_metadata():
    with open(config.DATA_DIR+ "data.txt","r") as f:
        lines = f.readlines()
    f.close() 
    random.shuffle(lines)
    return lines

def convert_to_index(words,word2index):
    return [word2index[w] for w in words]
    

def get_tokens(data,params):
    # create inputs and targets (x and y)
    x = list()
    y = list()
    vocab = json.load(open(params["DATA_DIR"]+"vocab.json","r"))
    word2index = vocab['word2index']
    
    inputs = list()
    outputs = list()
    print("Creating data...")
    for _,s in tqdm(enumerate(data),total = len(data)):
        for i in range(len(s.split())-params['MAX_SEQ_LEN']):
            inp = convert_to_index(s.split()[i:params['MAX_SEQ_LEN']+i],word2index)
            out =convert_to_index(s.split()[i+1:params['MAX_SEQ_LEN']+i+1],word2index)
            inputs.append(inp)
            outputs.append(out)
    print("Data prepared.")
    return torch.tensor(inputs),torch.tensor(outputs)



def create_val(test_data_loader,test_custom_data,params):
    for inp,out in test_data_loader:
        break
    inputs  = list()
    outputs = list()
    for i,o in zip(inp,out):
        inputs.append([test_custom_data.index2word[idx.item()] for idx in i])
        outputs.append([test_custom_data.index2word[idx.item()] for idx in o])
    file = open(params["DATA_DIR"] + 'validation_data.pt', 'wb')
    data = {
            "inputs":inputs,
            "outpus":outputs
        }

    pickle.dump(data,file)
    print("validation data created")
