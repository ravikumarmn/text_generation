import torch
import torch.nn as nn
import config
from torch import nn, optim
from models.lstm import Model
from models.transformer import TransformerModel
from torch.utils.data import DataLoader
from dataset import CustomDataset
from tqdm.auto import tqdm
import numpy as np
import wandb
from inference import predict_fn
import argparse
from os import listdir
from pathlib import Path

def train_fn(dataloader, model, optimizer,criterion):
    model.train()

    data_loader_tqdm = tqdm(enumerate(dataloader),desc = "Training",total = len(dataloader),leave=True)
    train_loss =list()
    for batch, (x, y) in data_loader_tqdm:
        optimizer.zero_grad()
        y_pred  = model(x,y)
        loss = criterion(y_pred.transpose(1, 2), y)


        loss.backward()
        optimizer.step()
        data_loader_tqdm.set_postfix({"Batch":batch,"Loss":loss.item()})
        train_loss.append(loss.item())
    training_loss =  sum(train_loss)/len(train_loss)
    return training_loss
    
@torch.no_grad()
def evaluate_fn(dataloader, model,criterion, device = 'cpu'):
    model.eval()
    model.to(device)

    test_loss = list()
    data_loader_tqdm = tqdm(enumerate(dataloader),desc = "Validation",total = len(dataloader),leave=True)
    for batch, (x, y) in data_loader_tqdm:
        y_pred = model(x,y)
        loss = criterion(y_pred.transpose(1, 2), y)

        test_loss.append(loss.item())
        data_loader_tqdm.set_postfix({"Batch":batch,"Loss":loss.item()})
    testing_loss = sum(test_loss)/len(test_loss)
    return testing_loss

def init_wandb(params,arg_params):
    wandb.init(
        config=params,
        project="text-generation-lstm",
        entity="ravikumarmn",
        name=f'batch_size_{params["BATCH_SIZE"]}-emb_dim_{params["EMB_DIM"]}-hidden_size_{params["N_HIDDEN"]}-traim_size{params["TRIM_SIZE"]}',
        group="multiclass",
        notes = "predicting text words using lstm.",
        tags=['lstm'],
        mode=arg_params.wandb_mode
    )
    text_table = wandb.Table(columns=["input","predicted","epoch"])
    return text_table

def remove_checkpoints(all_dir,filepath):
    lst_epochs = [int(x.split("_")[-1][:-3]) for x in all_dir]
    remove_idx = lst_epochs.index(min(lst_epochs))
    file = Path(str(filepath) +"/"+all_dir[remove_idx])
    file.unlink()

def main(arg_params):
    args =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    args['DEVICE'] = arg_params.device
    args['EXPERIMENT_NAME'] = arg_params.model
    
    print(f"\nTraining on : '{args['DEVICE']}' device\n")
    text_table = init_wandb(args,arg_params)

    custom_train = CustomDataset(args,"train")
    custom_test = CustomDataset(args,"test")  
    n_vocab = len(custom_train.words) +1
    if arg_params.model =="transformer":
        model = TransformerModel(n_vocab,args)
    if arg_params.model == "lstm":
        model = Model(n_vocab,args)

    model.to(args['DEVICE'])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"trainable_params : {trainable_params}")

    train_dataloader = DataLoader(custom_train,batch_size = args["BATCH_SIZE"],shuffle = True)
    test_dataloader = DataLoader(custom_test,batch_size = args["BATCH_SIZE"])

    optimizer = optim.Adam(model.parameters(), args["LR"])
    criterion = nn.CrossEntropyLoss()

    index2word = custom_train.index2word


    tqdm_obj_epoch = tqdm(range(args["N_EPOCHS"]),total = args["N_EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:

        training_loss = train_fn(train_dataloader,model,optimizer,criterion)
        validation_loss = evaluate_fn(test_dataloader,model,criterion,device=args['DEVICE'])
        true_inp,all_preds = predict_fn(index2word,model,args)

        text_table.add_data(true_inp,all_preds,epoch)

        print(f"Training loss : {training_loss}\tTesting loss : {validation_loss}")

        if validation_loss <val_loss:
            val_loss = validation_loss
            early_stopping = 0

            filepath = Path(args["CHECKPOINTS_DIR"]+args["EXPERIMENT_NAME"])
            try:
                filepath.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                print("Folder is already exist.")


            all_checkpoints = listdir(filepath)
            if len(all_checkpoints)>2:
                remove_checkpoints(all_checkpoints,filepath)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "params" : args,
                    "n_vocab":len(index2word)+1
                },args["CHECKPOINTS_DIR"]+args["EXPERIMENT_NAME"]+args["CHECKPOINT_NAME"]+f'_{epoch}.pt'
            )
        else:
            early_stopping += 1

        if early_stopping == args["PATIENCE"]:
            print(f'Model checkpoints saved to {args["CHECKPOINTS_DIR"]+args["EXPERIMENT_NAME"]}')
            print("Early stopping")
            break
        
        wandb.log({"epoch": epoch, "training_loss": training_loss, "val_loss": validation_loss})
    wandb.log({"training_samples" : text_table})
    print("traning complete.")

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_mode",
        choices=["online","disabled","offline"],
        default="disabled",
        help="Enter wandb.ai mode (online or disabled",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=['cpu',"cuda"],
        help="Choice your device to train the model.",
    )

    parser.add_argument(
        "--model",
        default="lstm",
        choices=['lstm',"transformer"],
        help="Choice your model to train.",
    )

    args = parser.parse_args()
    
    main(args)
    

    #1e7916410526ba74cb8e2cbaadcaf20a1fea2240


