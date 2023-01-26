import config
import torch
import json 
import re
import pandas as pd
from pathlib import Path
from utils import get_tokenizer, plot_word_counts,shift_word,split_method
from spacy.lang.en import English


def save_vocab(unique_words,save_dir,special_char = None):
    unique_words  = sorted(set(unique_words))

    n_to_char = {n:char for n, char in enumerate(unique_words)}
    char_to_n = {char:n for n, char in n_to_char.items()}

    vocab_size =  len(n_to_char)

    vocabed = {
        "word2index":char_to_n,
        "index2word":n_to_char,
        "vocab_len" : vocab_size,
        "special_charcters":special_char
    }
    json_obj = json.dumps(vocabed)
    with open(save_dir+"vocab.json", "w") as outfile:
        outfile.write(json_obj)
    return f"vocab created to {save_dir+'vocab.json'}"

def spacy_create_vocab(raw_data,save_dir,tokenizer):
    all_tokens = split_method(raw_data,tokenizer)
    save_vocab(all_tokens,save_dir)


def custom_create_vocab(raw_data,save_dir):
    special_char = list(set(re.findall("[^a-zA-Z0-9]",raw_data)))
    all_tokens = raw_data.split() + special_char 
    save_vocab(all_tokens,save_dir,special_char)

class PrepareData:
    def __init__(self, params):
        self.params = params
        vocab_data = json.load(open(params['DATA_DIR']+"vocab.json","r"))
        self.word2index = vocab_data['word2index']
        self.index2word = vocab_data['index2word']
        tokenizer_type = self.params.get('TOKENIZER_TYPE','custom')
        
        self.tokenizer = get_tokenizer(tokenizer_type)

    def encode(self,sentence):
        return [self.word2index[word] for word in split_method(sentence,self.tokenizer)]

    def decode(self,sequence):
        return [self.index2word[word] for word in sequence]

    def padded(self,sequence):
        return sequence + (self.params['TRIM_SIZE']-len(sequence))*0
    
    def trim_sequence(self,sequences,trim_size):
        padded_seq = list()
        for sequence in sequences:
            if len(sequence) >= trim_size:
                padded_seq.append(sequence[:trim_size])
            else:
                padded_seq.append(self.padded(sequence))
        return padded_seq

    def load_dataset(self):
        dataset = json.load(open(self.params["DATA_DIR"]+"data.json","r"))
        train_sentences =dataset['data']['train_data']
        val_sentences = dataset['data']['validation_data']

        train_sequences = list()
        val_sequences = list()
        for t_sen,v_sen in zip(train_sentences,val_sentences):

            train_sequences.append(self.encode(t_sen))
            val_sequences.append(self.encode(v_sen))
        return train_sequences,val_sequences


def create_inputs_outputs(params):
    prepare_data = PrepareData(params)
    train_sequences,val_sequences = prepare_data.load_dataset()
    
    train_inputs,train_outputs = shift_word(train_sequences,params)
    val_inputs,val_outputs = shift_word(val_sequences,params)
    
    
    data = {
        "train" :{
            "train_inputs":torch.tensor(train_inputs),
            "train_outputs":torch.tensor(train_outputs)
        },
        "val":{
            "val_inputs":torch.tensor(val_inputs),
            "val_outputs":torch.tensor(val_outputs),
        }
    }
    torch.save(data,params["DATA_DIR"]+"ins_outs.pt")
    print("Train & Validation Data prepared.")

def create_data(args):
    # Read csv data
    df = pd.read_csv(args['DEFAULT_FILE_NAME'])
    raw_data =  " ".join(df["Joke"]) # use print for text in debug mode
    sentences = df['Joke'].tolist()

    tokenizer_type = args.get('TOKENIZER_TYPE','custom')
    tokenizer = get_tokenizer(tokenizer_type)

    if tokenizer_type =='custom':
        custom_create_vocab(raw_data,args["DATA_DIR"])

    elif tokenizer_type =='spacy':
        spacy_create_vocab(raw_data,args["DATA_DIR"],tokenizer)
    else:
        raise NotImplementedError
    
    max_words = plot_word_counts(sentences,tokenizer)
    trim_size = int(input("ENTER TRIM SIZE:"))
    df['trim_flag'] = df['Joke'].apply(lambda x : \
        len(split_method(x,tokenizer))>=trim_size)

    valid_joke = df.loc[df['trim_flag'],"Joke"]
    
    sentences = valid_joke.tolist()
    # Create train & val dataset 
    file = open(args['DATA_DIR']+'data.json', 'w')
    json.dump(
        {
            "data":{
                "train_data":sentences[:round(len(sentences)*0.9)],
                "validation_data":sentences[round(len(sentences)*0.1):]
            },
            "metadata":{
                "max_words" :max_words,
                "trim_size" : trim_size
            }
        },file
    )
    print(f"train_val_data saved to {args['DATA_DIR']+'data.json'}\n")


if __name__=="__main__":
    args =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    my_file = Path(args['DATA_DIR'] + "data.json")
    if not my_file.is_file():
        create_data(args)
    else:
        print("data.json file is exists.")

    create_inputs_outputs(args)

    

















    # vocab,lines = create_load_vocab(params)

    # inputs,outputs = get_tokens(lines,params)
    # print(f"Input size : {inputs.shape}\nOutput size : {outputs.shape}")
    # X_train, X_test, y_train, y_test = train_test_split(inputs,outputs,test_size=0.2,shuffle=True)
    # # print(f"X_train : {X_train.shape} y_train : {y_train.shape} X_test : {X_test.shape} y_test : {y_test.shape}")
    # file = open(params["DATA_DIR"] + 'splitted.pt', 'wb')
    # data = {
    #     "X_train":X_train,
    #     "y_train" : y_train,
    #     "X_test" : X_test,
    #     "y_test":y_test
    # }
    # pickle.dump(data,file)
    # print("done")
    

    