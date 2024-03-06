import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import *
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer, fname = "baseLine-finetuning01"):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # task2: setup model's optimizer_scheduler if you have
    if args.scheduler == 'cosine':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, steps)

    lossList = []
    valLoss = []
    accList = []
    valAcc = []
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        acc = 0
        model.train()
        criterion = criterion.to(device)

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            if model.scheduler:
              model.scheduler.step()
            model.zero_grad()
            losses += loss.item() # average loss per batch
            acc += (logits.argmax(1) == labels).float().sum().item()
        
        lossList.append(losses/len(train_dataloader))
        accList.append(acc/len(datasets['train']))

        vls, vacc  = run_eval(args, model, datasets, tokenizer,  cr = criterion, split='validation')
        
        valLoss.append(vls)
        valAcc.append(vacc)
        
        print('train: epoch', epoch_count, '| losses:', losses, '| avg loss:', losses/len(train_dataloader))
    plot_losses(lossList, valLoss, fname)
    plot_acc(accList, valAcc, fname)
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    model.opt_setUp(opt,6,0.006) #set up the custom schedulers
    # task2: setup model's optimizer_scheduler if you have
    if args.scheduler == 'warm_up':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = model.warmup_scheduler
      model.scheduler.num_training_steps = steps
    elif args.scheduler == "SWA":
      model.scheduler = model.swa_scheduler
      model.avg_model = torch.optim.swa_utils.AveragedModel(model)
      
    # task3: write a training loop
    lossList = []
    valLoss = []
    accList = []
    valAcc = []
    for epoch_count in range(args.n_epochs):
        losses = 0
        acc = 0
        model.train()
        criterion = criterion.to(device)

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            #print("optimized-step")
            '''
            The Combined technique involves running a warmup for a certain number of steps before 
            running SWA for the rest of it; SWA complements the warmup as it makes use of the early exposure 
            of the model to the new data.
            '''
            # run the combined technique if the argument is Comb
            if args.scheduler == "Comb" and step > model.swa_start:
              if model.avg_model:
                model.avg_model.update_parameters(model)
              model.swa_scheduler.step()
            elif args.scheduler == "Comb":
              model.warmup_scheduler.step()
            else:
              if model.avg_model:
                model.avg_model.update_parameters(model)
              model.scheduler.step()
            #reset the gradients
            model.zero_grad()
            losses += loss.item() # average loss per batch
            acc += (logits.argmax(1) == labels).float().sum().item()
        
        lossList.append(losses/len(train_dataloader))
        accList.append(acc/len(datasets['train']))

        vls, vacc  = run_eval(args, model, datasets, tokenizer,  cr = criterion, split='validation')
        
        valLoss.append(vls)
        valAcc.append(vacc)
        
        print('train: epoch', epoch_count, '| losses:', losses, '| avg loss:', losses/len(train_dataloader))
    plot_losses(lossList, valLoss, fname)
    plot_acc(accList, valAcc, fname)  


def run_eval(args, model, datasets, tokenizer, cr = None, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    losses = 0
    acc = 0

    if cr:
      cr = cr.to('cpu')
    if split=="test":
      cr = nn.CrossEntropyLoss() 

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        if (args.task == 'supcon'):
            _, logits = model(inputs, labels)
            loss = cr(logits.to('cpu'), labels.to('cpu'))
            losses += loss.item()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
        else:
          logits = model(inputs, labels)
          if cr:
            losses += cr(logits.to('cpu'), labels.to('cpu')).item()
          tem = (logits.argmax(1) == labels).float().sum()
          acc += tem.item()

    print(f'{split} acc:', acc/len(datasets[split]), f'|total loss:', losses, f'|avg loss:', losses/len(dataloader), f'|dataset split {split} size:', len(datasets[split]))
    return losses/len(dataloader), acc/len(datasets[split])

def run_eval_aug(args, model, datasets, tokenizer, cr = None, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    losses = 0

    if not cr and args.task == 'supcon':
      from loss import SupConLoss
      cr = SupConLoss(temperature=args.temperature)
    if cr:
      cr = cr.to('cpu')

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        if (args.task == 'supcon'):
            f1, _ = model(inputs, labels)
            f2, _ = model (inputs, labels)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if (args.method == 'SupCon'):
                loss = cr(features, labels)
            elif (args.method == 'SimCLR'):
                loss = cr(features)
            else:
                raise ValueError('Must set a method to use SupCon: {} is not allowed'.format(args.method))
            # if step % 300 == 0:
            #   print("================== {} ==================".format(step))
            #   print ("input", inputs)
            #   print ("-----------------------------------")
            #   print ("features", features)
            #   print("====================================")
            losses += loss.item()
        else:
          raise ValueError('Only used with SupCon task')
    
    print(f'{split}', f'|total loss:', losses, f'|avg loss:', losses/len(dataloader), f'|dataset split {split} size:', len(datasets[split]))
    return losses/len(dataloader)

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == 'cosine':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, steps)
    elif args.scheduler == 'linear':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, steps, gamma=0.1)

    lossList = []
    valLoss = []
    accList = []
    valAcc = []
    
    # Trainging the augmentaion only
    for param in model.classify.parameters():
       param.requires_grad = False

    print (" ============ Training augmentation ============ ")
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        criterion = criterion.to(device)

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)

            f1, _ = model(inputs, labels)
            f2, _ = model(inputs, labels)

            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if (args.method == 'SupCon'):
                loss = criterion(features, labels)
            elif (args.method == 'SimCLR'):
                loss = criterion(features)
            else:
                raise ValueError('Must set a method to use SupCon: {} is not allowed'.format(args.method))
            # if step % 400 == 0:
            #   print("================== {} ==================".format(step))
            #   print ("input", inputs)
            #   print ("-----------------------------------")
            #   print ("features", features)
            #   print("====================================")
            loss.backward()
            model.optimizer.step()  # backprop to update the weights
            if model.scheduler:
               model.scheduler.step()
            
            model.zero_grad()
            losses += loss.item() # average loss per batch

        lossList.append(losses/len(train_dataloader))
        vls = run_eval_aug(args, model, datasets, tokenizer,  cr = criterion, split='validation')
        valLoss.append(vls)
        print('train: epoch', epoch_count, '| losses:', losses, '| avg loss:', losses/len(train_dataloader))
        if (epoch_count % 5 == 0):
          torch.save(model.state_dict(), f"./models/aug/{fname}-notDoneYet-{epoch_count}.pt")

    torch.save(model.state_dict(), f"./models/aug/{fname}-notDoneYet-final.pt")
    plot_losses(lossList, valLoss, fname + "augmentation")
    # save the model

    args.n_epochs = args.n_epochs_cla

    lossList = []
    valLoss = []
    model.dropout = nn.Dropout(p=0.2)
    print (" ============ Training classifier ============ ")
    # Training the classifier only
    for param in model.parameters():
       param.requires_grad = False # freeze the parameters

    for param in model.classify.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()  
    # resest the optimizer and scheduler

    model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_cla, eps=args.adam_epsilon)
    if args.scheduler == 'cosine':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, steps)
    elif args.scheduler == 'linear':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, steps, gamma=0.1)
    
    for epoch_count in range(args.n_epochs_cla):
        losses = 0
        acc = 0
        model.train()
        criterion = criterion.to(device)

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model, use_text=False)

            _, logits = model(inputs, labels)
            loss = criterion(logits, labels)

            loss.backward()
            model.optimizer.step()  # backprop to update the weights
            if model.scheduler:
               model.scheduler.step()
            
            model.zero_grad()
            losses += loss.item() # average loss per batch
            acc += (logits.argmax(1) == labels).float().sum().item()
        
        lossList.append(losses/len(train_dataloader))
        accList.append(acc/len(datasets['train']))

        vls, vacc  = run_eval(args, model, datasets, tokenizer,  cr = criterion, split='validation')
        
        valLoss.append(vls)
        valAcc.append(vacc)
        
        print('train: epoch', epoch_count, '| losses:', losses, '| avg loss:', losses/len(train_dataloader), '| acc:', acc/len(datasets['train']))
        if (epoch_count % 5 == 0):
          torch.save(model.state_dict(), f"./models/cla/{fname}-notDoneYet-{epoch_count}.pt")


    torch.save(model.state_dict(), f"./models/cla/{fname}-notDoneYet-final.pt")
    plot_losses(lossList, valLoss, fname + "classifier")
    plot_acc(accList, valAcc, fname + "classifier")

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  fname = get_name(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer, fname)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'supcon':
    from loss import SupConLoss
    cr = SupConLoss(temperature=args.temperature)
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    # load the model
    # model.load_state_dict(torch.load(f"./models/aug/bert_supcon_method-SupCon_lr-2e-05_bs-64_ep-20_dr-0.6_eps-1e-08_hdim-50_scheduler-cosine-notDoneYet-10.pt"))
    run_eval_aug(args, model, datasets, tokenizer, cr = cr, split='validation')
    supcon_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  
  dumpArgs(args, fname)
   
