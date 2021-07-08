import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import json
import pickle as pkl
import random
import time
import math
import numpy as np
import logging
import argparse
from nlgeval import compute_metrics
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch_utils import *
from encoderRNN import *
from attention import *
from bridge import *
from decoder import *
from contextRNN import *
from kb_encoder import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device {} ".format(device))

model_type = 'hred_seqlstm'
data_path = "data/dataset/v2/dialogue_data/context_2_20/"
save_path = 'models/' + model_type + '/'
model_folder = save_path + model_type + '_' + model_name + '/'
model_save_path = model_folder + model_type + '_' + model_name + '.pt'
save_pred_file = model_folder + model_type + '_' + model_name + '.txt'
save_attn_c1 = model_folder + model_type + '_' + model_name + '_c1.pkl'
save_attn_c2 = model_folder + model_type + '_' + model_name + '_c2.pkl'
save_metrics_file = model_folder + model_type + '_' + model_name + '_metrics.txt'
#annoy_path = "data/raw_catalog/image_annoy_index/

print(model_folder)
print(model_save_path)
print(save_pred_file)
print(save_attn_c1)

train_data = pkl.load(open(data_path + "train.pkl", 'rb'))

print("Sample training data:")
print(train_data[0])

valid_data = pkl.load(open(data_path + "valid.pkl", 'rb'))

print("Sample validation data:")
print(valid_data[0])

test_data = pkl.load(open(data_path + "test.pkl", 'rb'))

print("Sample Testing data:")
print(test_data[0])

vocab = pkl.load(open(data_path + "vocab.pkl", "rb"))[1]
print("Vocabulary Size: {} ".format(len(vocab)))


kb_vec = None
use_kb = False
celeb_vec = None
kb_size = None
celeb_vec_size = None
kb_len = None
celeb_len = None

def convert_to_gpu(data):
    data = data.to(device)
    return data

def convert_to_tensor(data):
    data = torch.LongTensor(data)
    return data

def convert_states_to_torch(data):
    torch_data = convert_to_gpu(convert_to_tensor(data))
    return torch_data


def get_image_representation(image_filename, image_rep_size, annoyIndex, annoyPkl):
    image_filename = image_filename.strip()	
    if image_filename=="":
        return [0.]*image_rep_size
    #FOR ANNOY BASED INDEX
    try:	
        len_images +=1
        return annoyIndex.get_item_vector(annoyPkl[image_filename])
        # Eg: 1838414, 3294309, 3469177
    except:
        return [0.]*image_rep_size

def get_batch_mmd_data(batch_data, sos_id, eos_id, pad_id):
    batch_data = np.array(batch_data)
    batch_size = batch_data.shape[0]
    text_enc_input = np.array(batch_data[:,0].tolist())
    text_enc_in_len = np.array(batch_data[:,1].tolist())
    dec_out_seq = np.array(batch_data[:,4].tolist())
    dec_seq_length = np.array(batch_data[:,5].tolist())
    sos_to_target = np.reshape(np.asarray([sos_id]*batch_size), (batch_size, 1))
    dec_text_input = np.concatenate((sos_to_target, dec_out_seq[:,:-1]), axis=1) 
    dec_text_input[dec_text_input==eos_id]=pad_id
    text_enc_input = text_enc_input.transpose(1,0,2) # (context,batch_size,max_len)

    text_enc_in_len = text_enc_in_len.transpose(1,0) # (context,batch_size)
    text_enc_input = convert_to_gpu(convert_to_tensor(text_enc_input))

    dec_out_seq = convert_to_gpu(convert_to_tensor(dec_out_seq))
    dec_text_input = convert_to_gpu(convert_to_tensor(dec_text_input))
    text_enc_in_len = convert_to_gpu(convert_to_tensor(text_enc_in_len))
    dec_seq_length = convert_to_gpu(convert_to_tensor(dec_seq_length))
    return text_enc_input, text_enc_in_len, dec_text_input, dec_out_seq, dec_seq_length

src_vocab_size = len(vocab)
tgt_vocab_size = len(vocab) 
src_emb_dim = 512 
tgt_emb_dim = 512
enc_hidden_size = 512
dec_hidden_size = 512
context_hidden_size = 512 
image_in_size = 20480
bidirectional_enc = True 
bidirectional_context = False
context_size = 2
num_enc_layers = 1 
num_dec_layers = 1
num_context_layers = 1
dropout_enc = 0.5 
dropout_dec = 0.5
dropout_context = 0.5
max_decode_len = 20
non_linearity = 'tanh' 
enc_type = 'GRU'
dec_type = 'GRU'
context_type ='GRU'
use_attention = True 
decode_function = 'softmax'
pad_id = 0
sos_id = 2
eos_id = 3
image_rep_size = 4096
tie_embedding = True
activation_bridge = 'Tanh'
num_states = 10
attentiondim = 200

batch_size = 256
learning_rate = 0.0001
weight_decay = 0
clip_grad = 1

batch_id = 0
batch_data = train_data[batch_id*batch_size:(batch_id+1)*batch_size]
text_enc_input, text_enc_in_len, dec_text_input, dec_out_seq, dec_seq_length= get_batch_mmd_data(batch_data, 
                    sos_id,
                    eos_id, pad_id)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.w = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0, scale=0.01,
                                                                size=(hidden_dim, attention_dim)))).float().to(device)
        self.b = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0,scale=0.01,
                                                                size=attention_dim))).float().to(device)
        self.u = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0,scale=0.01,
                                                                size=(attention_dim, 1)))).float().to(device)
        
    def forward(self, x):
        maxlen = x.size(1)
        xlstm = x.view(-1, self.hidden_dim)
        v = F.relu(torch.mm(xlstm, self.w) + self.b)
        vu = torch.mm(v, self.u).view(-1, maxlen)
        alpha = F.softmax(vu, dim=1).unsqueeze(2)
        out = torch.bmm(x.transpose(1, 2),alpha).squeeze()
        return alpha, out


class HRED(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
                enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
                image_in_size, bidirectional_enc=True, bidirectional_context=False, 
                num_enc_layers=1, num_dec_layers=1, num_context_layers=1, 
                dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
                non_linearity='tanh', enc_type='GRU', dec_type='GRU', context_type='GRU', 
                use_attention=True, decode_function='softmax', sos_id=2, eos_id=3, 
                tie_embedding=True, activation_bridge='Tanh', num_states=None,
                use_kb=False, kb_size=None, celeb_vec_size=None):
        super(HRED, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_emb_dim = src_emb_dim #512
        self.tgt_emb_dim = tgt_emb_dim #512
        self.batch_size = batch_size 
        self.bidirectional_enc = bidirectional_enc #True
        self.bidirectional_context = bidirectional_context #False
        self.num_enc_layers = num_enc_layers #1
        self.num_dec_layers = num_dec_layers #1
        self.num_context_layers = num_context_layers #1
        self.dropout_enc = dropout_enc #dropout prob for encoder
        self.dropout_dec = dropout_dec #dropout prob for decoder
        self.dropout_context = dropout_context #dropout prob for context
        self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.context_type = context_type
        self.sos_id = sos_id # start token
        self.eos_id = eos_id # end token
        self.decode_function = decode_function # @TODO: softmax or log softmax 
        self.max_decode_len = max_decode_len # max timesteps for decoder
        self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
        self.num_directions = 2 if bidirectional_enc else 1
        self.enc_hidden_size = enc_hidden_size // self.num_directions
        self.num_directions = 2 if bidirectional_context else 1
        self.context_hidden_size = context_hidden_size // self.num_directions
        self.dec_hidden_size = dec_hidden_size
        self.use_attention = use_attention
        self.attentiondim = attentiondim
        self.use_kb = use_kb 
        self.kb_size = kb_size 
        self.celeb_vec_size = celeb_vec_size
        self.kb_emb_size = self.tgt_emb_dim
        self.kb_hidden_size = self.dec_hidden_size
        
        #print(self.kb_size)
        #print(self.kb_emb_size)
        
        # Initialize encoder
        self.encoder = EncoderRNN(self.src_vocab_size, self.src_emb_dim, self.enc_hidden_size, 
                        self.enc_type, self.num_enc_layers, batch_first=True, dropout=self.dropout_enc, 
                        bidirectional=self.bidirectional_enc)
       
        self.activation_bridge = activation_bridge
        
        self.attention = Attention(enc_hidden_size, self.attention_size)
        #self.attention_c1 = Attention(enc_hidden_size, self.attention_size)
        #self.attention_c2 = Attention(enc_hidden_size, self.attention_size)
        self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size)
        # Initialize context encoder
        self.context_input_size = enc_hidden_size #self.image_out_size + enc_hidden_size # image+text
        self.context_encoder = ContextRNN(self.context_input_size, self.context_hidden_size, 
                                self.context_type, self.num_context_layers, batch_first=True,
                                dropout=self.dropout_context, bidirectional=self.bidirectional_context)
        # Initialize RNN decoder
        self.decoder = DecoderRNN(self.tgt_vocab_size, self.tgt_emb_dim, self.dec_hidden_size, 
                        self.dec_type, self.num_dec_layers, self.max_decode_len,  
                        self.dropout_dec, batch_first=True, use_attention=self.use_attention, 
                        attn_size = self.attention_size, sos_id=self.sos_id, eos_id=self.eos_id,
                        use_input_feed=True,
                        use_kb=self.use_kb, kb_size=self.kb_hidden_size, celeb_vec_size=self.kb_hidden_size)

        if tie_embedding:
            self.decoder.embedding = self.encoder.embedding
        # Initialize parameters
        self.init_params()

    def forward(self, text_enc_input, text_enc_in_len=None, dec_text_input=None,
                dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
                use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None, epoch=None):
        # text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
        # text_enc_in_len == (turn, batch) # np.array
        assert (text_enc_input.size(0)==context_size), "Context size not equal to first dimension"
        # Define variables to store outputs
        batch_size = text_enc_input.size(1)
        # https://github.com/pytorch/pytorch/issues/5552
        context_enc_input_in_place = torch.zeros(batch_size, context_size, self.context_input_size)
        
        context_enc_input = context_enc_input_in_place.clone()
        context_enc_input = context_enc_input.to(device) # Port to cuda
        for turn in range(0, context_size):
            # pytorch multiple packedsequence input ordering
            text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input

            if(turn == 0):
                encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
                c1_att_wgts_c1, c1_enc_rep_c1 = self.attention(encoder_outputs)
            else:
                encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
                c2_att_wgts_c2, c2_enc_rep_c2 = self.attention(encoder_outputs)
            
        inter_layer = torch.cat([c1_enc_rep_c1.unsqueeze(1), c2_enc_rep_c2.unsqueeze(1)], 1)
        text_outputs = F.relu(inter_layer)
            
        context_enc_input = text_outputs # (batch, 2, features)
        
        context_enc_outputs, context_enc_hidden = self.context_encoder(context_enc_input)
        context_projected_hidden = self.bridge(context_enc_hidden, 
                                bidirectional_encoder=self.bidirectional_context)#.unsqueeze(0) 
                                # (B,D) => (Layer,B,D)
        
        
        if not decode:
            decoder_outputs = self.decoder(dec_text_input,
                                           init_h=context_projected_hidden,
                                           encoder_outputs = encoder_outputs,
                                           input_valid_length = text_enc_in_len[turn],
                                           context_enc_outputs = context_enc_outputs, 
                                           decode=decode)
            if(epoch is not None):
                return decoder_outputs, c1_att_wgts_c1.detach().cpu().numpy(), c2_att_wgts_c2.detach().cpu().numpy()
            return decoder_outputs
        else:
            prediction = self.decoder(init_h=context_projected_hidden,
                                encoder_outputs = encoder_outputs,
                                input_valid_length = text_enc_in_len[turn],
                                context_enc_outputs = context_enc_outputs,
                                decode=decode)
            if(epoch is not None):
                return prediction, c1_att_wgts_c1.detach().cpu().numpy(), c2_att_wgts_c2.detach().cpu().numpy()
            return prediction

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data.uniform_(-initrange, initrange)

model = HRED(src_vocab_size=len(vocab), #config['model']['src_vocab_size'],
            tgt_vocab_size=len(vocab), #config['model']['tgt_vocab_size'],
            src_emb_dim=src_emb_dim,
            tgt_emb_dim=tgt_emb_dim,
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            context_hidden_size=context_hidden_size,
            batch_size=batch_size,
            image_in_size=image_in_size,
            bidirectional_enc=bidirectional_enc,
            bidirectional_context=bidirectional_context,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            num_context_layers=num_context_layers,
            dropout_enc=dropout_enc,
            dropout_dec=dropout_dec,
            dropout_context=dropout_context,
            max_decode_len=max_decode_len,
            non_linearity=non_linearity,
            enc_type=enc_type,
            dec_type=dec_type,
            context_type=context_type,
            use_attention=use_attention,
            decode_function=decode_function,
            num_states=num_states,
            use_kb=use_kb, kb_size=kb_size, celeb_vec_size=celeb_vec_size
            )
model = model.to(device)

output = model(text_enc_input, text_enc_in_len, 
               dec_text_input, dec_out_seq, context_size=context_size, 
               teacher_forcing_ratio=0.9, kb_vec=kb_vec, 
               celeb_vec=celeb_vec, kb_len=kb_len, celeb_len=celeb_len)
print(output.size())

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = model.to(device)

losses = []
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

total_attn_wgts_c1 = []
total_attn_wgts_c2 = []
total_samples = len(train_data)
num_train_batch = int(math.ceil(float(total_samples)/float(batch_size)))
num_epochs = 5

print("Training Batches {} and for Epochs {} ".format(num_train_batch, num_epochs))

for epoch in range(num_epochs):
    total_loss = 0. 
    n_total_words = 0.
    epoch_start = time.time()
    model.train()
    for batch_id in range(num_train_batch):
        batch_start = time.time()
        batch_data = train_data[batch_id*batch_size:(batch_id+1)*batch_size]
    
        text_enc_input, text_enc_in_len, dec_text_input, dec_out_seq, dec_seq_length= get_batch_mmd_data(batch_data, 
                    sos_id,
                    eos_id, pad_id)
        
        optimizer.zero_grad()
        
        
        if(epoch + 1 == num_epochs):
            dec_output_prob, attn_wgts_c1, attn_wgts_c2 = model(text_enc_input, text_enc_in_len, 
                            dec_text_input, dec_out_seq, context_size=context_size, 
                            teacher_forcing_ratio=1, epoch=epoch)
            total_attn_wgts_c1.extend(attn_wgts_c1)
            total_attn_wgts_c2.extend(attn_wgts_c2)
        else:
            dec_output_prob = model(text_enc_input, text_enc_in_len, 
                            dec_text_input, dec_out_seq, context_size=context_size, 
                            teacher_forcing_ratio=1)
        # loss = loss_criterion(dec_output_prob, dec_out_seq)
        loss = loss_criterion(dec_output_prob.contiguous().view(-1, tgt_vocab_size), #config['model']['tgt_vocab_size']),
            dec_out_seq.view(-1))

        # target_toks = dec_out_seq.ne(config['data']['pad_id']).long().sum().data[0]
        n_words = dec_seq_length.float().sum().item()
        n_total_words += n_words
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        # exp_lr_scheduler.step()
        batch_elapsed = (time.time() - batch_start)/60
        batch_loss = loss.item()/n_words # @TODO
        if (batch_id+1) % 1000 == 0:
            print('Batch Loss: Epoch [%d], Batch [%d], Loss: %.6f, Perplexity: %5.5f, Batch Time:%5.4f'
                  %(epoch+1, batch_id+1, batch_loss, np.exp(batch_loss), batch_elapsed)) 
        total_loss += loss.item()
        losses.append(batch_loss)
    epoch_loss = total_loss / n_total_words

    epoch_elapsed = time.time() - epoch_start
    print('Epoch Loss: Epoch [%d], Loss: %.6f, Perplexity: %5.5f, Epoch Time:%5.4f'
          %(epoch+1, epoch_loss, np.exp(epoch_loss), epoch_elapsed))

torch.save(model.state_dict(), model_save_path)
pkl.dump(total_attn_wgts_c1, open(save_attn_c1, 'wb'))
pkl.dump(total_attn_wgts_c2, open(save_attn_c2, 'wb'))

#t1 =  pkl.load(open(save_attn_c1, 'rb'))
#t2 =  pkl.load(open(save_attn_c2, 'rb'))

model = HRED(src_vocab_size=len(vocab), #config['model']['src_vocab_size'],
            tgt_vocab_size=len(vocab), #config['model']['tgt_vocab_size'],
            src_emb_dim=src_emb_dim,
            tgt_emb_dim=tgt_emb_dim,
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            context_hidden_size=context_hidden_size,
            batch_size=batch_size,
            image_in_size=image_in_size,
            bidirectional_enc=bidirectional_enc,
            bidirectional_context=bidirectional_context,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            num_context_layers=num_context_layers,
            dropout_enc=dropout_enc,
            dropout_dec=dropout_dec,
            dropout_context=dropout_context,
            max_decode_len=max_decode_len,
            non_linearity=non_linearity,
            enc_type=enc_type,
            dec_type=dec_type,
            context_type=context_type,
            use_attention=use_attention,
            decode_function=decode_function,
            num_states=num_states
            )
model = model.to(device)

model.load_state_dict(torch.load(model_save_path))
model.eval()

total_samples = len(test_data)
num_test_batch = int(math.ceil(float(total_samples)/float(batch_size)))
sentences=[]
epoch = None

for batch_id in range(num_test_batch):
    batch_start = time.time()
    batch_data = test_data[batch_id*batch_size:(batch_id+1)*batch_size]
    

    text_enc_input, text_enc_in_len, dec_text_input, dec_out_seq, dec_seq_length= get_batch_mmd_data(batch_data, 
                    sos_id,
                    eos_id, pad_id)
    dec_output_prob = model(text_enc_input, text_enc_in_len, 
                            dec_text_input, dec_out_seq, context_size=context_size, 
                            teacher_forcing_ratio=0, decode=True)
    dec_output_seq = dec_output_prob[:,0,:].data.cpu().numpy()
    
    for sequence in dec_output_seq:
        words = []
        for word_id in sequence:
            if word_id == eos_id:
                break
            word = vocab[word_id]
            words.append(word)
        sentence = ' '.join(words)
        sentences.append(sentence)
with open(save_pred_file, 'w') as out_file:
    for item in sentences:
        out_file.write("{}\n".format(item))

truth_sen = []
with open(data_path+'test_tokenized.txt', 'r', encoding='latin1') as f:
    for line in f.readlines():
        truth_sen.append(line[:-1])
        
pred_sen = []
with open(save_pred_file, 'r', encoding='latin1') as f:
    for line in f.readlines():
        pred_sen.append(line[:-1])

from nlgeval import NLGEval

omit_metrics = ['METEOR','CIDEr']
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=omit_metrics)
metrics_dict = nlgeval.compute_metrics([truth_sen], pred_sen)
print(metrics_dict)
