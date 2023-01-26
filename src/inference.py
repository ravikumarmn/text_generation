import torch
import config
from dataset import CustomDataset
from models.lstm import Model
from models.transformer import TransformerModel
from os import listdir
from pathlib import Path
import argparse
from utils import get_tokenizer,split_method

def predict(inp,model,index2word,device="cpu",next_word=None):
    sequence = inp.to(device)
    model.to(device)
    model.eval()
    next_word_predicted = list()
    i = 0
    while True:
        i+=1
        y_pred = model(sequence)
        _,indices = torch.topk(torch.softmax(y_pred[0][-1],dim = 0),1)
        predicted_word = index2word[indices.item()]
        next_word_predicted.append(predicted_word)
        sequence = torch.cat((sequence[:,1:],indices.view(-1).unsqueeze(0)),dim = -1)
        if next_word is not None:
            if i == next_word-1:
                break
        elif predicted_word == ".":
            break
    return next_word_predicted

def predict_fn(index2word,model,params,next_word = 5):
    data = torch.load(open(params["DATA_DIR"] + 'ins_outs.pt', 'rb'))
    val_inputs = data['val']['val_inputs'][:5]
    model.eval()
    true_inp = list()
    predicted = list()
    for inp in val_inputs:
        next_word_predicted = predict(inp.unsqueeze(0),model,index2word,params['DEVICE'],next_word)
        predicted.append(" ".join(next_word_predicted))
        true_inp.append(" ".join([index2word[i.item()] for i in inp]))
    return true_inp,predicted

def main(arg_params):
    # Load checkpoint
    filepath = Path("checkpoints",arg_params.model)
    all_dir = listdir(filepath)
    lst_epochs = [int(x.split("_")[-1][:-3]) for x in listdir(filepath)]
    use_idx = lst_epochs.index(max(lst_epochs))
    model_path = str(filepath)+"/"+all_dir[use_idx]
    print(f"Using checkpoint : {all_dir[use_idx]}\n")
    checkpoints = torch.load(model_path,map_location=torch.device('cpu'))
    # Model Instantiate
    n_vocab = checkpoints['n_vocab']
    args = checkpoints['params']
    custom_data = CustomDataset(params=args)
    if arg_params.model =="lstm":
        model = Model(n_vocab,args)
    elif arg_params.model == "transformer":
        model = TransformerModel(n_vocab,args)
    model.load_state_dict(checkpoints['model_state_dict'])
    text = arg_params.input
    tokenizer_type = args.get('TOKENIZER_TYPE','custom')
    tokenizer = get_tokenizer(tokenizer_type=tokenizer_type)
    # create data & predict.
    inps = torch.tensor([custom_data.word2index[str(w)] for w in tokenizer(text)])
    next_word = arg_params.next_words
    next_word_predicted = " ".join(predict(inps.unsqueeze(0),model,custom_data.index2word,next_word=next_word))
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
    parser.add_argument(
        "--model",
        default='transformer',
        required=False,
        choices=['lstm',"transformer"],
        help="Choice your model to test.",
    )
    args = parser.parse_args()

    main(args)