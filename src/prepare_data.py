import config
import torch
import json 
from collections import Counter
from utils import get_all_file_path,create_dataframe,clean_text,trim_sentence,create_vocab,load_metadata,get_tokens
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,params,data_type="train"):
        self.params = params
        words = self.load_words()
        self.unique_words = self.get_unique_words(words)
        vocab = json.load(open(params['DATA_DIR']+"vocab.json","r"))
        self.word2index = vocab['word2index']
        self.index2word = {i:w for w,i in self.word2index.items()}
        self.word_indexes = [self.word2index[w] for w in words]
        
        data = pickle.load(open(params["DATA_DIR"] + 'splitted.pt', 'rb'))

        if data_type=="train":
            dataset = list()
            for inp,lbl in zip(data['X_train'],data['y_train']):
                dataset.append((inp,lbl))
            self.data = dataset
        if data_type=="test":
            dataset = list()
            for inp,lbl in zip(data['X_test'],data['y_test']):
                dataset.append((inp,lbl))
            self.data = dataset

    def load_words(self):
        with open(config.DATA_DIR+ "data.txt","r") as file:
            data =file.read().replace("\n"," ").split()
        return data
        

    def get_unique_words(self,words):
        word_counts = Counter(words)
        sorted_word = sorted(word_counts,key=word_counts.get,reverse=True)
        return sorted_word

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self,index):
        inp,out = self.data[index]
        return (inp.long().to("cuda"),out.long().to("cuda"))
        
        # return 
        #     torch.tensor(self.word_indexes[index:index+self.params['MAX_SEQ_LEN']]),
        #     torch.tensor(self.word_indexes[index+1:index+self.params['MAX_SEQ_LEN']+1])
        # )


def create_load_vocab(params):
    csv_file_names = get_all_file_path(params["DATA_DIR"])
    df = create_dataframe(csv_file_names)
    df = df.dropna()
    df['clean_review'] = df['review'].apply(lambda x : clean_text(x))
    print("Data cleaned")
    df['trim_clean_review'] = trim_sentence(df['clean_review'],params)
    file = open('data.txt','w')
    for data in df['trim_clean_review']:
        file.write(data)
    file.close()
    lines = load_metadata()
    create_vocab(lines,params)
    vocab = json.load(open(config.DATA_DIR+"vocab.json","r"))
    return vocab,lines

if __name__=="__main__":
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}

    vocab,lines = create_load_vocab(params)

    inputs,outputs = get_tokens(lines,params)
    print(f"Input size : {inputs.shape}\nOutput size : {outputs.shape}")
    X_train, X_test, y_train, y_test = train_test_split(inputs,outputs,test_size=0.2,shuffle=True)
    # print(f"X_train : {X_train.shape} y_train : {y_train.shape} X_test : {X_test.shape} y_test : {y_test.shape}")
    file = open(params["DATA_DIR"] + 'splitted.pt', 'wb')
    data = {
        "X_train":X_train,
        "y_train" : y_train,
        "X_test" : X_test,
        "y_test":y_test
    }
    pickle.dump(data,file)
    print("done")
    