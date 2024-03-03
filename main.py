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
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    model.opt_setUp(opt)
    # model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # task2: setup model's optimizer_scheduler if you have
    if args.scheduler == 'warm_up':
      steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = model.warmup_scheduler
      #model.scheduler.optimizer = model.optimizer
      model.scheduler.num_training_steps = steps
    elif args.scheduler == "SWA":
      #steps = args.n_epochs * len(list(enumerate(train_dataloader)))
      model.scheduler = model.swa_scheduler
      model.avg_model = torch.optim.swa_utils.AveragedModel(model)
      #model.scheduler.optimizer = model.optimizer

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
            if model.avg_model:
              model.avg_model.update_parameters(model)
            model.scheduler.step()
            #print("sched-step")
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
    loss = 0
    acc = 0
    if split=="test":
      cr = nn.CrossEntropyLoss() 
    if cr:
      cr = cr.to('cpu') 
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        if cr:
         loss += cr(logits.to('cpu'), labels.to('cpu')).item()

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

    print(f'{split} acc:', acc/len(datasets[split]), f'|total loss:', loss, f'|avg loss:', loss/len(dataloader), f'|dataset split {split} size:', len(datasets[split]))
    return loss/len(dataloader), acc/len(datasets[split])

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    
    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  fname = get_name(args)
#   fname = "add_scheduler_" + get_name(args)


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
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
  
  dumpArgs(args, fname)
   
