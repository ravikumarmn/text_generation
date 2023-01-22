import torch
import random
import config
from prepare_data import CustomDataset
from torch.utils.data import DataLoader
from model import Model
import pickle
def predict(word2index,index2word,model,params,next_word = 5):

    data = pickle.load(open(params["DATA_DIR"] + 'validation_data.pt', 'rb'))
    true_inp = list()
    true_out = list()
    predicted = list()

    for inp,out in zip(data['inputs'],data['outpus']):
        sequence = torch.tensor([[word2index[w] for w in inp]]).to("cuda")
        
        y_pred = model(sequence)
        _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
        
        next_word_predicted = list()
        next_word_predicted.append(index2word[indices.item()])
        for _ in range(next_word-1):
            sequence = torch.cat((sequence[:,1:],indices.view(1,-1)),dim = -1)
            y_pred = model(sequence)
            _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
            next_word_predicted.append(index2word[indices.item()])
        
        predicted.append(" ".join(next_word_predicted))
        true_inp.append(" ".join(inp))
        true_out.append(" ".join(out[:next_word]))

    return true_inp,true_out,predicted


def predict_realtime(test_custom_data,model,text=None,num_words_to_predict = 5):
    if not text:
        dataset = test_custom_data.data
        random.shuffle(dataset)
        examples = dataset[:10]

    for inp,out in examples:
        # sequence = torch.tensor([[test_custom_data.word2index[w] for w in inp]]).to("cuda")
        sequence = inp.view(1,-1)
        y_pred = model(sequence)
        _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
        
        next_word_predicted = list()
        next_word_predicted.append(test_custom_data.index2word[indices.item()])
        for _ in range(num_words_to_predict-1):
            sequence = torch.cat((sequence[:,1:],indices.view(1,-1)),dim = -1)
            y_pred = model(sequence)
            _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
            next_word_predicted.append(test_custom_data.index2word[indices.item()])
        
        true_inp = " ".join([test_custom_data.index2word[i.item()] for i in sequence[0]])
        predicted_out = " ".join(next_word_predicted)
        true_out = " ".join([test_custom_data.index2word[i.item()] for i in out][:num_words_to_predict])
        
        print(f"\nInput : {true_inp}\nPredicted : {predicted_out}\nTrue Output : {true_out}\n\n")
        print("="*50)
        # true_inp.append(" ".join(sequence))
        # true_out.append(" ".join(out[:num_words_to_predict]))

    # model.to("cpu")
    # words = text.split()
    # sequence = torch.tensor([[test_custom_data.word2index[w] for w in words]]).to("cpu")
    
    # y_pred = model(sequence)
    # _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
        
    # next_word_predicted = list()
    # next_word_predicted.append(test_custom_data.index2word[indices.item()])
    # for _ in range(num_words_to_predict-1):
    #     sequence = torch.cat((sequence[:,1:],indices.view(1,-1)),dim = -1)
    #     y_pred = model(sequence)
    #     _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
    #     next_word_predicted.append(test_custom_data.index2word[indices.item()])
    # return text +" "+ " ".join(next_word_predicted)


if __name__=="__main__":
    # path = "/home/Ravikumar/Developer/text_generation/checkpoints/"
    model_path = "/home/Ravikumar/Developer/text_generation/checkpoints/batch_size_128-emb_dim_32-hidden_size_128.pt"
    checkpoints = torch.load(model_path,map_location=torch.device('cpu'))

    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    custom_data = CustomDataset(params=params)
    data_loader = DataLoader(custom_data,batch_size=params['BATCH_SIZE'])

    n_vocab = len(custom_data.unique_words)

    model = Model(n_vocab,params)
    model.load_state_dict(checkpoints['model_state_dict'])
    # predict(custom_data.word2index,custom_data.index2word,model,params,text = "there s a family where a")
    text = "there s a family where a"
    predict_realtime(custom_data,model,num_words_to_predict=10)
    print()
    # predict(data,model,model_path=)



"""
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

    """


"""

# def predict(params, model, text, next_words=5):
#     vocab = json.load(open(params["DATA_DIR"]+"vocab.json","r"))

#     model.eval()
#     model.cpu() 
#     words = text.split(' ')
#     hidden = model.init_state()
#     predicted = list()
#     sequence = torch.tensor([[vocab["word2index"][w] for w in words]])

#     for idx in range(sequence.size(1)):
#         y_pred,hidden = model(sequence[:,idx].unsqueeze(0), (hidden[0].cpu(),hidden[1].cpu()))
    
#     p = torch.nn.functional.softmax(y_pred, dim=1)
#     word_index = p.argmax(1)
#     predicted.append(vocab["index2word"][word_index.item()])

#     for _ in range(1,next_words):

#         y_pred,state_h = model(word_index,state_h)
#         p = torch.nn.functional.softmax(y_pred, dim=1)
#         word_index = p.argmax(1)
#         predicted.append(params["index2word"][word_index.item()])
#     print(f"Input : {text}\nOutput : {' '.join(predicted)}")
#     return predicted
# """