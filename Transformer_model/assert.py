# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This modules demonstrates how to convert code to subtokenized sentences."""

import itertools
import json
from typing import List, Text

from absl import app
from absl import flags
from absl import logging
from tensor2tensor.data_generators import text_encoder
from cubert import cubert_tokenizer
from cubert import tokenizer_registry
from cubert import unified_tokenizer
import csv 

FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary_filepath', None,
                    'Path to the subword vocabulary.')

flags.DEFINE_string('input_filepath', None,
                    'Path to the Python source code file.')

flags.DEFINE_string('saved_model', None,
                    'Path to the saved model to be used for assert generation')

flags.DEFINE_enum_class(
    'tokenizer',
    default=tokenizer_registry.TokenizerEnum.JAVA,
    enum_class=tokenizer_registry.TokenizerEnum,
    help='The tokenizer to use.')


def code_to_cubert_sentences(
    code,
    initial_tokenizer,
    subword_tokenizer,
):
  """Tokenizes code into a list of CuBERT sentences.

  Args:
    code: The source code to tokenize. This must be a parseable unit of code,
      meaning it represents an AST (or a complete subtree of an AST). For
      example, there should be no unmatched parentheses, and `if` and other
      blocks of code must have bodies.
    initial_tokenizer: The first tokenizer that creates sentences, probably a
      cubert_tokenizer.CuBertTokenizer.
    subword_tokenizer: A second tokenizer that splits tokens of the
      `initial_tokenizer` into subtokens.

  Returns:
    A list of sentences.
  """
  tokens = initial_tokenizer.tokenize(code)[:-1]  # type: List[Text]
  logging.vlog(5, 'Code >>>%s<<< is tokenized into >>>%s<<<.', code, tokens)

  # This will split the list into sublists of non-NEWLINE tokens (key is
  # False) and NEWLINE tokens (key is True).
  groups_by_endtoken = itertools.groupby(
      tokens, key=lambda x: x == unified_tokenizer.NEWLINE)
  # This will keep only the sublists that aren't just [NEWLINE]*, i.e., those
  # that have key False. We call these raw_sentences, because they're not
  # terminated.
  raw_sentences = [list(v) for k, v in groups_by_endtoken if not k
                  ]  # type: List[List[Text]]

  # Now we append a NEWLINE token after all sentences. Note that our tokenizer
  # drops any trailing \n's before tokenizing, but for the purpose of forming
  # properly terminated sentences, we always end sentences in a NEWLINE token.
  sentences = [s + [unified_tokenizer.NEWLINE] for s in raw_sentences
              ]  # type: List[List[Text]]
  logging.vlog(5, 'Tokens are split into sentences: >>>%s<<<.',
               sentences)

  # Now we have to encode tokens using the subword text encoder, expanding the
  # sentences.
  subtokenized_sentences = []  # type: List[List[Text]]
  for sentence in sentences:
    encoded_tokens = [subword_tokenizer.encode_without_tokenizing(t)
                      for t in sentence]  # type: List[List[int]]
    logging.vlog(5, 'Sentence encoded into >>>%s<<<.', encoded_tokens)
    flattened_encodings = sum(encoded_tokens, [])  # type: List[int]
    logging.vlog(5, 'Flattened into >>>%s<<<.', flattened_encodings)
    decoded_tokens = subword_tokenizer.decode_list(
        flattened_encodings)  # type: List[Text]
    logging.vlog(5, 'Sentence re-decoded into >>>%s<<<.', decoded_tokens)

    subtokenized_sentences.append(decoded_tokens)
  logging.vlog(5, 'Sentences are further subtokenized: >>>%s<<<.',
               subtokenized_sentences)
  return subtokenized_sentences

#___________________________________________________________________________________________________________________


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
    # scores = F.softmax(scores, dim=-1)
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
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
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
        self.ff = FeedForward(d_model).cuda()
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
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
        # with torch.no_grad():
          # x = model2(src)
          # print(x)
          # torch.cuda.empty_cache()
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,34)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

d_model = 1024
heads = 8
N = 3

src_vocab = 50297
trg_vocab = 50297


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Transformer(src_vocab, trg_vocab, d_model, N, heads).to(device=device)

dic={}

with open(FLAGS.vocabulary_filepath,'r') as fp: 
  lines = fp.read().splitlines()
k=0
for line in lines:
  dic[k]=line
  k=k+1

def translate(model, src, max_len = 34, custom_string=False):
    
    model.eval()
    src_mask = (src != 0).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)
    
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0]=torch.LongTensor([65]).to(device)

    for i in range(1, max_len):    
             
        trg_mask = np.triu(np.ones((1, i, i), dtype='float'),k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
        
        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
        e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        outputs[i] = ix[0][0]
    for i in range(len(outputs)):
      if (outputs[i]==1 or outputs[i]==15):
        print()
        break
      t = dic[outputs[i].item()]

      t = t[1:len(t)-1]

      if (t.endswith('^_')):
        print(t[:len(t)-2]+'',end='')
      elif (t.endswith('_')):
        print(t[:len(t)-1]+ ' ',end='')
      else:
        print(t,end='')

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

model = torch.load(FLAGS.saved_model).to(device=device)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # The value of the `TokenizerEnum` is a `CuBertTokenizer` subclass.
  tokenizer = FLAGS.tokenizer.value()
  subword_tokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocabulary_filepath)
  
  dic={}
  with open(FLAGS.vocabulary_filepath,'r') as fp: 
    lines = fp.read().splitlines()
  k=0
  for line in lines:
    line=line[1:len(line)-1]
    dic[line]=k
    k=k+1
  max_len = 0
  UKN=5
  with open(FLAGS.input_filepath, 'r') as input_file:
    code = input_file.readlines()
  k=1
  li_final=[]
  for lines in code:
    subtokenized_sentences = code_to_cubert_sentences(
    	code = lines,
    	initial_tokenizer = tokenizer,
    	subword_tokenizer=subword_tokenizer)
    li=[]
    li2=[]
    try:

      for i in range(0,len(subtokenized_sentences[0])):
        if(subtokenized_sentences[0][i]!=' ' and subtokenized_sentences[0][i]!='_'):
          li.append(dic.get(subtokenized_sentences[0][i],UKN))
          li2.append(subtokenized_sentences[0][i])
    except:
      pass
    li_final.append(li)

  t=li_final[0]
  t.append(1)
  if(len(t)<=350):
    for i in range(350 - len(t)):
      t.append(0)       
  else:    
    t=t[:350]
    t[350-1]=1
         
  t=torch.LongTensor(t).unsqueeze(0).to(device=device)
  translate(model,t)

if __name__ == '__main__':
  flags.mark_flag_as_required('vocabulary_filepath')
  flags.mark_flag_as_required('input_filepath')
  flags.mark_flag_as_required('saved_model')
  app.run(main)
