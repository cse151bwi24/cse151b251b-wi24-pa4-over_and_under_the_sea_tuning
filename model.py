import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    
    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)
    
    # print("---")
    # print(type(self.encoder.parameters()))
    # print(type(self.classify.parameters()))
    # print("---")
    
    # task1: add necessary class variables as you wish.
    # parameters = list(self.encoder.parameters()) + list(self.classify.parameters())
    self.optimizer = None
    self.scheduler = None
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    outputs = self.encoder(**inputs, output_hidden_states=True) # eval mode?
    last_hidden_states = outputs.hidden_states[-1]
    cls = last_hidden_states[:,0,:]
    
    return self.classify(cls)


class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(IntentModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)
    self.optimizer = None
    self.scheduler = None

    # # Setting up warm_up scheduler to just warm-- no decay
    self.warmup_scheduler = None
    # Setting up SWA
    #swa_model = torch.optim.swa_utils.AveragedModel()
    #self.swa_scheduler = None
    self.swa_scheduler = None
    self.avg_model = None
    self.swa_start = 0 # used for combined technique to begin SWA
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
  def opt_setUp (self, opt, warmup_steps,swa_lr):
    self.optimizer = opt
    self.warmup_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = warmup_steps)
    self.swa_start = warmup_steps # for the combined technique
    self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr = swa_lr)


class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    self.linear_head = nn.Sequential(
        nn.Linear(feat_dim,  384),
        nn.ReLU(),
        nn.Linear(384, 192),
    )
    
    self.method = args.method
 
  def forward(self, inputs, targets):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    # print ("Forwarding")
    # print("Using model", self.encoder)
    # print (inputs)

    outputs = self.encoder(**inputs, output_hidden_states=True) # eval mode?
    last_hidden_states = outputs.hidden_states[-1]
    cls = last_hidden_states[:,0,:]
    # print("--------------------cls-------------------")
    # print(cls)
    f1 = self.dropout(cls)
    f2 = self.dropout(cls)
    # print("--------------------drop-------------------")
    # print(f1, f2)
    # print("---------------normalized-------------------")
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    # print(f1, f2)
    f1p = self.linear_head(f1)
    f2p = self.linear_head(f2)
    # print("--------------------linear-------------------")
    # print (f1, f2)
    features = torch.cat([f1p.unsqueeze(1), f2p.unsqueeze(1)], dim=1)
    # print("===========com=============")
    # print (features.shape)
    logits = self.classify(self.dropout(cls))
    return features, logits

