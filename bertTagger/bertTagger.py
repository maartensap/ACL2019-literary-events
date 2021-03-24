#/usr/bin/env python

import sys, os
import argparse
import random
import json
import logging
import pandas as pd
from tqdm import tqdm, trange
from IPython import embed
from nltk.tokenize import word_tokenize

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
try:
  from pytorch_transformers import *
except:
  from transformers import *

from sklearn.metrics import roc_auc_score, accuracy_score, \
  precision_recall_fscore_support

def loadTokenizer(model="bert-base-cased"):
  special_tokens = dict(zip(
    'sep_token pad_token cls_token'.split(),
    "[SEP] [PAD] [CLS]".split()))
  if "roberta" in model:
    tokC = RobertaTokenizer
  elif "bert" in model:
    tokC = BertTokenizer
  elif "xlnet" in model:
    tokC = XLNetTokenizer
  tok = tokC.from_pretrained(model)
  tok.add_special_tokens(special_tokens)
  return tok, special_tokens

def loadModel(model,tokenizer,device="cpu"):
  if "roberta" in model:
    modelC = RobertaForTokenClassification
  elif "bert" in model:
    modelC = BertForTokenClassification
  elif "xlnet" in model:
    modelC = XLNetForTokenClassification
    
  model = modelC.from_pretrained(model)
  model.resize_token_embeddings(len(tokenizer))
  model.to(device)
  return model

def loadSavedModel(path, model, tokenizer, device="cpu"):
  output_model_file = os.path.join(path, "pytorch_model.bin")
  model_state_dict = torch.load(output_model_file)
  model.load_state_dict(model_state_dict)
  return model

class TaggingDataHandler:
  def __init__(self,path,tokCol,labelCol,debug,txtCol=None):
    # todo: if no tok col but txtCol, tokenize here first (for eval)
    self.df = pd.read_csv(path)
    if debug:
      self.df = self.df.sample(min(args.debug,len(self.df)))

    if txtCol:
      tqdm.pandas(ascii=True,desc=f"Tokenizing {txtCol}")
      self.df[tokCol] = self.df[txtCol].progress_apply(word_tokenize)
      
    self.tokCol = tokCol
    self.labelCol = labelCol

    if labelCol not in self.df:
      # randomly assign labels (for predition)
      self.df[labelCol] = self.df[tokCol].apply(
        lambda x: ["EVENT" if len(x)<4 else "O" for i in x])
      
    for c in [tokCol,labelCol]:
      try:
        self.df[c] = self.df[c].apply(eval)
      except:
        pass
      
    self.labelMap = {"O": 0, "EVENT": 1}
    
  def createModellingExamples(self,tok,max_len=512):
    df, labelCol, tokCol = self.df,self.labelCol, self.tokCol
    clf_id, sep_id, pad_id = tok.convert_tokens_to_ids(
      ["[CLS]","[SEP]","[PAD]"])

    nCols = []
    for c in [tokCol]:
      nCol = c+"_ids"
      tqdm.pandas(desc=f"Tokenizing {c}",ascii=True)
      print(df[c].head(3))
      # df[c] = df[c].fillna()
      df[nCol] = df[c].progress_apply(lambda x: list(map(tok.encode,x)))
      nCols.append(nCol)
      
    # realigning the labels
    df[labelCol+"_ids"] = df[labelCol].apply(
      lambda x: [[self.labelMap.get(i,0)] for i in x])
    df[labelCol+"_ids"] = df[[labelCol+"_ids",tokCol+"_ids"]].apply(
      lambda x: [l for ls,ids in zip(*x) for l in ls*len(ids)],axis=1)
    df["tok_index_map"] = df[tokCol+"_ids"].apply(
      lambda x: [i for i,ids in enumerate(x) for _ in ids])
    df[tokCol+"_ids"] = df[tokCol+"_ids"].apply(
      lambda x: [i for ids in x for i in ids])
    
    assert all(df["labels_ids"].apply(len) == df["toks_ids"].apply(len))
    max_len = min(max_len, df.toks_ids.apply(len).max()+2)
    
    df["text_ids"] = df[tokCol+"_ids"].apply(lambda x: x[:max_len-2])
    df["target_ids"] = df[labelCol+"_ids"].apply(lambda x: x[:max_len-2])
    
    df["text_ids"] = df["text_ids"].apply(
      lambda x: [clf_id] + x + [sep_id] + [pad_id] * (max_len-len(x)-2))
    df["target_ids"] = df["target_ids"].apply(
      lambda x: [0] + x + [0]* (max_len-len(x)-1))
    
    assert (df["text_ids"].apply(len) <= max_len).all()
    
    self.text_ids_col = "text_ids"
    self.target_ids_col = "target_ids"
    #embed();exit()
    
    return df
  
  def tensorize(self,device):
    # turn the data into tensors for feeding into batch creator
    text_input = torch.tensor(
      pd.np.stack(self.df[self.text_ids_col]),
      device=device, dtype=torch.long)
    target_input = torch.tensor(
      pd.np.stack(self.df[self.target_ids_col]),
      device=device, dtype=torch.long)
    
    attn_mask = torch.tensor(1 - (pd.np.stack(self.df[self.text_ids_col]) == 0),
                             device=device,dtype=torch.long)
    inputs = {"inp_"+self.text_ids_col: text_input}
    
    targets = {"target_ids": target_input, "attn": attn_mask}
        
    return inputs, targets

def aggregatePredBatches(targ_,pred_,mask_):
  targ = pd.np.concatenate([t.cpu().numpy() if t.is_cuda else t.numpy() for t in targ_])
  mask = pd.np.concatenate([t.cpu().numpy() if t.is_cuda else t.numpy() for t in mask_])
  score = pd.np.concatenate([p.cpu().numpy() if p.is_cuda else p.numpy() for p in pred_])[:,:,-1]
  # ends = mask.argmin(axis=1)
  return targ, mask, score

def prepPredictions(targ_,pred_,mask_,df):
  targ, mask, score = aggregatePredBatches(targ_,pred_,mask_)
  ends = mask.argmin(axis=1)
  
  # removing start and end token predictions
  targL = [ts[1:e-1] for ts,e in zip(targ,ends)]
  scoreL = [ss[1:e-1] for ss,e in zip(score,ends)]
  
  predL = [s.round().astype(int) for s in scoreL]
  from itertools import groupby
  from collections import Counter
  
  def aggPredToWord(x):
    x = list(x)
    if len(x) == 1:
      return x[0][1]
    votes = [i[1] for i in x]
    # majority
    return Counter(votes).most_common(1)[0][0]
  
  predA = [[aggPredToWord(g) for k,g in groupby(zip(i,s),key=lambda x: x[0])]
           for i,s in zip(df["tok_index_map"],predL)]
  targA = [[aggPredToWord(g) for k,g in groupby(zip(i,s),key=lambda x: x[0])]
           for i,s in zip(df["tok_index_map"],targL)]

  df["predEventLabel"] = predA
  df["targEventLabel"] = targA
  
  try:
    assert (df["predEventLabel"].apply(len) == df["targEventLabel"].apply(len)).all()
  except:
    embed();exit()
  
  df["nEvents"] = df["predEventLabel"].apply(pd.np.sum)
  df["pctEvents"] = df["nEvents"] / df["toks"].apply(len)
  return df

def computeClfScores(targ,pred):
  try:
    assert (pred.apply(len) == targ.apply(len)).all()
  except:
    embed();exit()
  targL = [t for ts in targ for t in ts]
  predL = [p for ps in pred for p in ps]
  pr,re,f1,_ = precision_recall_fscore_support(targL, predL, average="binary")
  outDict = dict(
    # acc = accuracy_score(targL,predL),
    f1 = f1, pre=pr, rec=re,
    supp=pd.np.mean(targL))
  return outDict

  
def computeClfScoresOld(targ_,pred_,mask_):
  # linearize
  targ, mask, score = aggregatePredBatches(targ_,pred_,mask_)
  # targ = pd.np.concatenate([t.cpu().numpy() if t.is_cuda else t.numpy() for t in targ_])
  # mask = pd.np.concatenate([t.cpu().numpy() if t.is_cuda else t.numpy() for t in mask_])
  # score = pd.np.concatenate([p.cpu().numpy() if p.is_cuda else p.numpy() for p in pred_])[:,:,-1]
  ends = mask.argmin(axis=1)
  
  targLin = pd.np.array([t for ts,e in zip(targ,ends) for t in ts[:e]])
  scoreLin = pd.np.array([s for ss,e in zip(score,ends) for s in ss[:e]])
  predLin = scoreLin.round()
  
  pr,re,f1,s = precision_recall_fscore_support(targLin, predLin, average="binary")
  try:
    auc = roc_auc_score(targLin,scoreLin)
  except ValueError:
    auc = pd.np.nan
    
  outDict = dict(
    auc = auc,
    acc = accuracy_score(targLin,predLin),
    pre=pr,rec=re,f1=f1,supp=targLin.sum()
  )
  return outDict

def do_eval(args,dataHandler,tok,model):
  logger.info("Evaluating the model")
  devData = dataHandler(args.eval_data,args.tokCol,
                        args.labelCol,args.debug,txtCol=args.txtCol)
    
  devdf = devData.createModellingExamples(tok,max_len=args.max_seq_len)

  # Tensorize
  inputs, targets = devData.tensorize(device)
  inp_keys = list(inputs.keys())
  tar_keys = list(targets.keys())
  
  dev_data = TensorDataset(*inputs.values(),*targets.values())
  dev_sampler = SequentialSampler(dev_data)
  dev_loader = DataLoader(dev_data, sampler=dev_sampler,
                          batch_size=args.eval_batch_size)

  # Running eval loop
  model.eval()

  targs, preds, masks = [], [], []
  eval_loss = 0
  nb_eval_steps, nb_eval_examples = 0, 0
  for batch in tqdm(dev_loader, desc="Evaluating",ascii=True):
    batch_d = dict(zip(inp_keys+tar_keys,batch))

    with torch.no_grad():
      loss, logits = model(batch_d["inp_text_ids"],
                           labels=batch_d["target_ids"],
                           attention_mask=batch_d["attn"])
      
      # prob of positive class
      probs = logits.softmax(dim=-1)
      targs.append(batch_d["target_ids"])
      masks.append(batch_d["attn"])
      preds.append(probs)

    eval_loss += loss.item()
  
    nb_eval_examples += batch_d["inp_text_ids"].size(0)
    nb_eval_steps += 1

  eval_loss = eval_loss / nb_eval_steps
  
  df = prepPredictions(targs,preds,masks,devdf)
  metrics = computeClfScores(df["targEventLabel"],df["predEventLabel"])
  oldMetrics = computeClfScoresOld(targs,preds,masks)

  metrics["eval_loss"] = eval_loss
  
  if args.pred_output_file:  
    df.to_pickle(args.pred_output_file)
    logger.info(f"Exported predictions to {args.pred_output_file}")
  return metrics

def do_train(args,dataHandler,tok,model):
  logger.info("Training the model")
  # Tokenize and encode dataset

  trnData = dataHandler(args.train_data,args.tokCol,
                        args.labelCol,args.debug)
      
  trndf = trnData.createModellingExamples(tok,max_len=args.max_seq_len)
  logger.info(f"Training examples: {len(trndf)}")

  # Tensorize
  inputs, targets = trnData.tensorize(device)
  inp_keys = list(inputs.keys())
  tar_keys = list(targets.keys())
  
  trn_data = TensorDataset(*inputs.values(),*targets.values())
  trn_sampler = RandomSampler(trn_data)
  trn_loader = DataLoader(trn_data, sampler=trn_sampler,
                          batch_size=args.train_batch_size)
  
  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(trn_loader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(trn_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    
  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = t_total)
  # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
  output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
      
  # Do training loop
  nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
  model.train()
  for _ in trange(int(args.num_train_epochs), desc="Epoch",ascii=True):
    tr_loss = 0
    nb_tr_steps = 0
    cum_acc = 0
    tqdm_bar = tqdm(trn_loader, desc="Training",ascii=True)
    for step, batch in enumerate(tqdm_bar):
      batch_d = dict(zip(inp_keys+tar_keys,batch))

      loss, logits = model(batch_d["inp_text_ids"],
                           labels=batch_d["target_ids"],
                           attention_mask=batch_d["attn"])
      
      acc = ((logits.argmax(dim=-1) == batch_d["target_ids"]).type(
        torch.long) * batch_d["attn"]).sum().item() / batch_d["attn"].sum().item()
      
      cum_acc += acc
      
      loss.backward()
      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      model.zero_grad()
      
      tr_loss += loss.item()
      exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
      nb_tr_steps += 1
      tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e} acc:{:.2f}".format(exp_average_loss, scheduler.get_lr()[0],cum_acc/nb_tr_steps)
    
    # Saving model at every epoch
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)

  train_loss = tr_loss/nb_tr_steps
  
  # Save model
  model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
  torch.save(model_to_save.state_dict(), output_model_file)
  logger.info(f"Saved model to {output_model_file}")
  return {"train_loss": train_loss,
          "num_train_steps": nb_tr_steps,
          "num_train_epochs": args.num_train_epochs}

def main(args):
  global device
  global logger
  if not args.do_train and not args.do_eval and  not args.do_gen:
    raise ValueError("At least one of `do_train`, `do_gen`, "
                     " or `do_eval` must be True.")
  
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  if args.do_train and args.log_file:
    logging.basicConfig(filename=os.path.join(args.output_dir,args.log_file),
                        level=logging.INFO)
    logger = logging.getLogger(__name__)    
    logger.addHandler(logging.StreamHandler())
  else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
  logger.info(args)
  
  # Set random seeds
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  logger.info("device: {}, n_gpu {}".format(device, n_gpu))  
  
  tokenizer, special_tokens = loadTokenizer(args.bert_model)
  bert_model = loadModel(args.bert_model, tokenizer, device)
  dataHandler = TaggingDataHandler
  
  if not args.do_train:
    bert_model = loadSavedModel(args.output_dir,bert_model,tokenizer,device)
  
  out_dict = {}
  if args.do_train:
    out_dict = do_train(args,dataHandler,tokenizer,bert_model)
    
  if args.do_eval:
    r = do_eval(args,dataHandler,tokenizer,bert_model)
    out_dict.update(r)
        
  print(out_dict)
  logger.info(out_dict)

if __name__ =="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--do_train",action="store_true")
  parser.add_argument("--do_eval",action="store_true")
  parser.add_argument("--train_data")
  parser.add_argument("--eval_data")

  parser.add_argument("--pred_output_file")
  parser.add_argument("--tokCol",default="toks")
  parser.add_argument("--txtCol")
  parser.add_argument("--labelCol",default="labels")

  parser.add_argument("--output_dir")
  parser.add_argument("--log_file",default="log.txt")
  
  parser.add_argument("--debug",default=0,type=int)
  parser.add_argument('--bert_model', default="bert-base-uncased",
                      help="Bert, RoBERTa, XLNet") 

  parser.add_argument('--max_seq_len', type=int, default=128)
  
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--n_positions', type=int, default=256)
  parser.add_argument('--num_train_epochs', type=int, default=3)
  parser.add_argument('--train_batch_size', type=int, default=4)
  parser.add_argument('--eval_batch_size', type=int, default=4)
  parser.add_argument('--max_grad_norm', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=2e-5)
  parser.add_argument('--warmup_proportion', type=float, default=0.002)
  parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lm_coef', type=float, default=0.9)
  parser.add_argument('--n_valid', type=int, default=374)
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_steps", default=-1, type=int,
                      help="If > 0: set total number of training steps to perform."
                      "Override num_train_epochs.")
  parser.add_argument("--warmup_steps", default=0, type=int,
                      help="Linear warmup over warmup_steps.")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before "
                      "performing a backward/update pass.")

  args = parser.parse_args()
  
  main(args)
