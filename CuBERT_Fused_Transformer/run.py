import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
import gc
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from csv import reader

def data_process(file,maxl):
  list_of_lists=[]
  with open(file, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
      row=list(map(int,row))
      row.append(1)
      if(len(row)<=maxl):
        for i in range(maxl - len(row)):
          row.append(0)
        list_of_lists.append(row)
      else:
        row=row[:maxl]
        row[maxl-1]=1
        list_of_lists.append(row)
  
  
  return list_of_lists
  
def data_process_assert(file,maxl):
  list_of_lists=[]
  with open(file, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
      row=list(map(int,row))
      row.append(1)
      row=row[6:]
      if(len(row)<=maxl):
        for i in range(maxl - len(row)):
          row.append(0)
        list_of_lists.append(row)
      else:
        row=row[:maxl]
        row[maxl-1]=1
        list_of_lists.append(row)
  
  
  return list_of_lists

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("processing data")
src_train = torch.tensor(data_process("../train_test.txt",350)).to(device=device)
trg_train = torch.tensor(data_process_assert("../train_assert.txt",35)).to(device=device)

src_test = torch.tensor(data_process("../test_test.txt",350)).to(device=device)
trg_test = torch.tensor(data_process_assert("/home/achendrac/google-research/cubert/data/test_assert.txt",35)).to(device=device)

src_valid = torch.tensor(data_process("../Eval_test.txt",350)).to(device=device)
trg_valid = torch.tensor(data_process_assert("../Eval_assert.txt",35)).to(device=device)
print("processing data ends")

train=[]
test=[]
valid=[]

print("processing dataloader")
for i in range(len(src_train)):
  t=(src_train.data[i],trg_train.data[i])
  train.append(t)
for i in range(len(src_test)):
  t=(src_test.data[i],trg_test.data[i])
  test.append(t)
for i in range(len(src_valid)):
  t = (src_valid.data[i],trg_valid.data[i])
  valid.append(t)

train_iter = DataLoader(train, batch_size=16,
                      shuffle=True)
valid_iter = DataLoader(valid, batch_size=16,
                      shuffle=True)
test_iter = DataLoader(test, batch_size=16,
                     shuffle=True)
print("processing dataloader ends")

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device=device)
    def forward(self, x):
        return self.embed(x).to(device=device)

from torch.autograd import Variable
import math
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

def create_masks(src,trg):
  input_pad = 0
  target_pad = 0
  input_msk = (src != input_pad).unsqueeze(1)

  target_msk = (trg != target_pad).unsqueeze(1)
  size = trg.size(1) # get seq_len for matrix

  nopeak_mask = np.triu(np.ones((1, size, size), dtype='float'),k=1).astype('uint8')
  nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(device=device)
  target_msk = target_msk & nopeak_mask

  return input_msk,target_msk

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

import torch.nn.functional as F
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
      mask = mask.unsqueeze(1)
      scores = scores.masked_fill(mask == 0, -1e9)
    scores = torch.nn.functional.softmax(scores, dim = -1)    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

import copy
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model)
        self.attn2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask, out_cubert):
        x2 = self.norm_1(x)
        #y2 = self.norm_1(out_cubert)
        y = self.attn2(x, out_cubert, out_cubert, mask)
        x = x + self.dropout_1(self.attn1(x2,x2,x2,mask))
        x2 = self.norm_2(x + y)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.attn_3 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
    def forward(self, x, e_outputs, src_mask, trg_mask, cubert_output):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        
        y = self.attn_3(x,cubert_output,cubert_output,src_mask)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        
        x2 = self.norm_3(x + y)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,400)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        with torch.no_grad():
          out_cubert = model2(src)
          torch.cuda.empty_cache()
        for i in range(N):
            x = self.layers[i](x, mask, out_cubert[0])
        return out_cubert[0].to(device=device), self.norm(x + out_cubert[0])
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,34)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask, cubert_output):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, cubert_output)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        cubert_output , e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, cubert_output)
        output = self.out(d_output)
        return output

d_model = 1024
heads = 8
N = 2
epochs = 9
src_vocab = 50297
trg_vocab = 50297


model = Transformer(src_vocab, trg_vocab, d_model, N, heads).to(device=device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

import time

def train_model(epochs, print_every=100):
    step1 = 0
    step2 = 0
    
    start = time.time()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            model.train()

            src = batch[0].to(device=device)
            trg = batch[1].to(device=device)
            
            trg_input = trg[:, :-1]
            
            # the words we are trying to predict
            
            targets = trg[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),targets, ignore_index=0)
            loss.backward()
            optim.step()
            writer.add_scalar("Training loss", loss.data, global_step=step1)
            step1 += 1
            total_loss += loss.data
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("Training time = %dm, epoch %d, iter = %d, loss = %.3f,%ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()
        
        for i, batch in enumerate(valid_iter):
            model.eval()

            src = batch[0].to(device=device)
            trg = batch[1].to(device=device)
            
            trg_input = trg[:, :-1]
            
            # the words we are trying to predict
            
            targets = trg[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),targets, ignore_index=0)
            loss.backward()
            optim.step()
            writer.add_scalar("Validation loss", loss.data, global_step=step2)
            step2 += 1
            total_loss += loss.data
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("Validation time = %dm, epoch %d, iter = %d, loss = %.3f,%ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()



PRE_TRAINED_MODEL_CUBERT = "../cubert/pretrained_cubert"

print("loading cubert")
from transformers import BertModel
model2 = BertModel.from_pretrained(PRE_TRAINED_MODEL_CUBERT).to(device=device)
model2.eval()
print("loading cubert ends")

writer = SummaryWriter(f"../runs_Cubert_fused3_m/loss_plot")
train_model(epochs)
torch.save(model,"../model_Cubert_fused3_m")


