import torch
import torch.nn as nn
import config
from torch import nn, optim
from model import Model
from torch.utils.data import DataLoader
from prepare_data import CustomDataset
from tqdm.auto import tqdm
import numpy as np
import wandb
from inference import predict
from utils import create_val

def train_fn(dataloader, model, args,optimizer,criterion):
    model.train()

    data_loader_tqdm = tqdm(enumerate(dataloader),desc = "Training",total = len(dataloader),leave=True)
    train_loss =list()
    for batch, (x, y) in data_loader_tqdm:
        optimizer.zero_grad()
        y_pred  = model(x)
        loss = criterion(y_pred.transpose(1, 2), y)


        loss.backward()
        optimizer.step()
        data_loader_tqdm.set_postfix({"Batch":batch,"Loss":loss.item()})
        train_loss.append(loss.item())
    training_loss =  sum(train_loss)/len(train_loss)
    return training_loss

def evaluate_fn(dataloader, model, args,criterion):
    model.eval()
    model.to("cuda")

    test_loss = list()
    data_loader_tqdm = tqdm(enumerate(dataloader),desc = "Validation",total = len(dataloader),leave=True)
    with torch.no_grad():
        for batch, (x, y) in data_loader_tqdm:
            y_pred = model(x)
            loss = criterion(y_pred.transpose(1, 2), y)

            test_loss.append(loss.item())
            data_loader_tqdm.set_postfix({"Batch":batch,"Loss":loss.item()})
    testing_loss = sum(test_loss)/len(test_loss)
    return testing_loss

def main(model,args):
    trains = CustomDataset(args,"train")
    tests = CustomDataset(args,"test")  

    train_dataloader = DataLoader(trains,batch_size = args["BATCH_SIZE"],shuffle = True,drop_last = True)
    test_dataloader = DataLoader(tests,batch_size = args["BATCH_SIZE"],drop_last = True)
    optimizer = optim.Adam(model.parameters(), args["LR"])
    model.to("cuda")
    criterion = nn.CrossEntropyLoss()
    word2index= trains.word2index
    index2word = trains.index2word
    tqdm_obj_epoch = tqdm(range(args["N_EPOCHS"]),total = args["N_EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf
    for epoch in tqdm_obj_epoch:
        training_loss = train_fn(train_dataloader,model,args,optimizer,criterion)
        validation_loss = evaluate_fn(test_dataloader,model,args,criterion)

        true_inp,true_out,all_preds = predict(word2index,index2word,model,params)
        text_table.add_data(true_inp,all_preds,true_out,epoch)
        print(f"Training loss : {training_loss}\tTesting loss : {validation_loss}")
        if validation_loss <val_loss:
            val_loss = validation_loss
            early_stopping = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "params" : args
                },
                params['CHECKPOINTS_DIR']
            )
        else:
            early_stopping += 1

        if early_stopping == args["PATIENCE"]:
            print(f"Model checkpoints saved to {args['CHECKPOINTS_DIR']}")
            print("Early stopping")
            break
        
        wandb.log({"epoch": epoch, "training_loss": training_loss, "val_loss": validation_loss})
    wandb.log({"training_samples" : text_table})

if __name__=="__main__":    
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    # train_custom_data = CustomDataset(params,"train")
    test_custom_data = CustomDataset(params,"test")
    # train_data_loader = DataLoader(train_custom_data,batch_size=params['BATCH_SIZE'])
    test_data_loader = DataLoader(test_custom_data,batch_size=params['BATCH_SIZE'])

    create_val(test_data_loader,test_custom_data,params)
    wandb.init(
        config=params,
        project="text-generation-lstm",
        entity="ravikumarmn",
        name=f'batch_size_{params["BATCH_SIZE"]}-emb_dim_{params["EMB_DIM"]}-hidden_size_{params["N_HIDDEN"]}-max_seq_len_{params["MAX_SEQ_LEN"]}',
        group="multiclass",
        notes = "predicting text words using lstm.",
        tags=['lstm'],
        mode=" "
    )
    text_table = wandb.Table(columns=["input","predicted","true_output","epoch"])

    n_vocab = len(test_custom_data.unique_words)
    model = Model(n_vocab,params)
    # train_fn(train_custom_data,model, params)
    # evaluate_fn(test_custom_data,model, params)
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"trainable_params : {trainable_params}")
    main(model,params)
    print("traning complete.")
    #1e7916410526ba74cb8e2cbaadcaf20a1fea2240


