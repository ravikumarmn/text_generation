import torch
import config
from dataset import CustomDataset
from models.lstm import Model
from os import listdir
from pathlib import Path
import argparse

def predict(inp,model,index2word,device="cpu",next_word=None):
    sequence = inp.to(device)

    model.to(device)
    y_pred = model(sequence,sequence)
    _,indices = torch.topk(torch.softmax(y_pred[-1],dim = 0),1)
    
    next_word_predicted = list()
    next_word_predicted.append(index2word[indices.item()])
    i = 0
    
    while True:
        i+=1
        sequence = torch.cat((sequence[1:],indices.view(-1)),dim = 0)
        y_pred = model(sequence,sequence)
        _,indices = torch.topk(torch.softmax(y_pred[-1],dim = 0),1)
        predicted_word = index2word[indices.item()]
        next_word_predicted.append(predicted_word)
        if next_word is not None:
            if i == next_word-1:
                break
        elif "." in predicted_word:
            break
    return next_word_predicted

# def predict(inp,model,index2word,device="cpu",next_word=5):
#     sequence = inp

#     model.eval()
#     model.to("cpu")
#     y_pred = model(sequence)
#     _,indices = torch.topk(torch.softmax(y_pred[-1],dim = 0),1)
#     next_word_predicted = list()
#     next_word_predicted.append(index2word[indices.item()])
#     while indices.item()!=0:
#         sequence = torch.cat((sequence[1:],indices.view(-1)),dim = 0)
#         y_pred = model(sequence)
#         _,indices = torch.topk(torch.softmax(y_pred[-1],dim = 0),1)
#         pred_word = index2word[indices.item()]
#         next_word_predicted.append(pred_word)
#         if "." in pred_word:
#             break
#     print(f"lenght : {len(next_word_predicted)}")
    
#     return next_word_predicted

def predict_fn(index2word,model,params,next_word = 5):
    data = torch.load(open(params["DATA_DIR"] + 'ins_outs.pt', 'rb'))
    val_inputs = data['val']['val_inputs'][:5]
    # val_outputs = data['val']['val_outputs'][:5]
    model.eval()
    true_inp = list()
    predicted = list()

    for inp in val_inputs:
        next_word_predicted = predict(inp,model,index2word,params['DEVICE'],next_word)
        predicted.append(" ".join(next_word_predicted))
        true_inp.append(" ".join([index2word[i.item()] for i in inp]))

    return true_inp,predicted


def main(arg_params):
    args =  {k:v for k,v in config.__dict__.items() if "__" not in k}

    filepath = Path(args["CHECKPOINTS_DIR"]+args["EXPERIMENT_NAME"])
    all_dir = listdir(filepath)
    lst_epochs = [int(x.split("_")[-1][:-3]) for x in listdir(filepath)]
    use_idx = lst_epochs.index(max(lst_epochs))
    model_path = str(filepath)+"/"+all_dir[use_idx]
    checkpoints = torch.load(model_path,map_location=torch.device('cpu'))

    n_vocab = checkpoints['n_vocab']

    custom_data = CustomDataset(params=args)

    model = Model(n_vocab,args)
    model.load_state_dict(checkpoints['model_state_dict'])
    text = arg_params.input
    inps = torch.tensor([custom_data.word2index[w] for w in text.split()])
    next_word = arg_params.next_words
    next_word_predicted = " ".join(predict(inps,model,custom_data.index2word,next_word=next_word))
    print(f"\nInput : {text}\nPredicted : {next_word_predicted}\n")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="There were two snowmen standing in a field, one",
        help="give input to for inference to test the model."
    )
    parser.add_argument(
        "--next_words",
        type=int,
        help="How many text word to predict."
    )
    args = parser.parse_args()

    main(args)