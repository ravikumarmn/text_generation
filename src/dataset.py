import torch
import json

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,params,data_type="train"):
        self.params = params
        # words = self.load_words(data_type)
        # self.unique_words = self.get_unique_words(words)
        vocab = json.load(open(params['DATA_DIR']+"vocab.json","r"))
        self.word2index = vocab['word2index']

        self.words = list(self.word2index.keys())
        self.index2word = {i:w for w,i in self.word2index.items()}
        self.word_indexes = [self.word2index[w] for w in self.words]
        
        data = torch.load(open(params["DATA_DIR"] + 'ins_outs.pt', 'rb'))

        if data_type=="train":
            dataset = list()
            for inp,lbl in zip(data['train']['train_inputs'],data['train']['train_outputs']):
                dataset.append((inp,lbl))
            self.data = dataset
        if data_type=="test":
            dataset = list()
            for inp,lbl in zip(data['val']['val_inputs'],data['val']['val_outputs']):
                dataset.append((inp,lbl))
            self.data = dataset

    # def load_words(self):
    #     with open(self.params["DATA_DIR"]+ "ins_outs.txt","r") as file:
    #         data =file.read().replace("\n"," ").split()
    #     return data
        

    # def get_unique_words(self,words):
    #     word_counts = Counter(words)
    #     sorted_word = sorted(word_counts,key=word_counts.get,reverse=True)
    #     return sorted_word

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self,index):
        inp,out = self.data[index]
        return (inp.long().to(self.params['DEVICE']),out.long().to(self.params['DEVICE']))