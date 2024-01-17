import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from torch.nn import functional as F


class Cross_Attention(nn.Module):
    def __init__(self, emb_size, dropout):
        super(Cross_Attention, self).__init__()

        self.emb_size = emb_size


    def forward(self, x, attention_scores):         
        out = torch.mul(x, attention_scores)
        return out


class Intra_Attention(nn.Module):
    def __init__(self, emb_size, head, dropout):
        super(Intra_Attention, self).__init__()

        self.num_attention_heads = head
        self.attention_head_size = emb_size // head
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(emb_size, self.all_head_size)
        self.key = nn.Linear(emb_size, self.all_head_size)
        self.value = nn.Linear(emb_size, self.all_head_size)

        self.dense1 = nn.Linear(emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        mixed_query_layer = self.query(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]
        out = context_layer + self.dense1(context_layer)
  
        return self.norm(out)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding = self.encoding.cuda()
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]
    

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.att_dp = config.atten_dropout
        self.emb_size = config.emb_size
        self.length = config.seqlength
        self.dim_t, self.dim_v, self.dim_a = config.dim_t, config.dim_v, config.dim_a
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.L1Loss()
        self.n_head = config.n_head

        ####text encoder and decoder
        self.encoder_te = nn.Linear(self.dim_t, self.emb_size)
        self.encoder_tm = nn.Linear(self.dim_t, self.emb_size)
        self.decoder_t = nn.Sequential(nn.Linear(self.emb_size * 2, self.emb_size), nn.Linear(self.emb_size, self.dim_t))

        ####visual encoder and decoder
        self.encoder_ve = nn.Linear(self.dim_v, self.emb_size)
        self.encoder_vm = nn.Linear(self.dim_v, self.emb_size)
        self.decoder_v = nn.Sequential(nn.Linear(self.emb_size * 2, self.emb_size), nn.Linear(self.emb_size, self.dim_v))

         ####acoustic encoder and decoder
        self.encoder_ae = nn.Linear(self.dim_a, self.emb_size)
        self.encoder_am = nn.Linear(self.dim_a, self.emb_size)
        self.decoder_a = nn.Sequential(nn.Linear(self.emb_size * 2, self.emb_size), nn.Linear(self.emb_size, self.dim_a))

        ###modality classifier  0: text, 1: visual,  2: acoustic
        self.classifier_m = nn.Linear(self.emb_size, 3)
        self.intra_t = Intra_Attention(self.emb_size, self.n_head, self.att_dp)
        self.intra_a = Intra_Attention(self.emb_size, self.n_head, self.att_dp)
        self.intra_v = Intra_Attention(self.emb_size, self.n_head, self.att_dp)

        self.intra_t.cuda()
        self.intra_a.cuda()
        self.intra_v.cuda()

    def weight(self, length):
        w = torch.ones(size=(length, length))
        v = torch.arange(1, length + 1) / (length / 2)
        temp = torch.arange(0, length + 1) / (length / 2)

        for i in range(length):
            w[i] = v.clone()
            temp[0] = v[-1].clone()
            temp[1:] = v.clone()
            v = temp[:length].clone()
        return w.transpose(1, 0)

    def forward(self, text, visual, acoustic):

        self.batch = visual.shape[0]
 
        ###disentangle
        x_te, x_ve, x_ae = self.encoder_te(text), self.encoder_ve(visual), self.encoder_ae(acoustic) ##[l,b, emb_size]
        x_tm, x_vm, x_am = self.encoder_tm(text), self.encoder_vm(visual), self.encoder_am(acoustic) ##[l,b, emb_size]
        ####modality classifier
        prediction1 = rearrange(self.classifier_m(x_tm), 'h b d -> (h b) d')
        prediction2 = rearrange(self.classifier_m(x_vm), 'h b d -> (h b) d')
        prediction3 = rearrange(self.classifier_m(x_am), 'h b d -> (h b) d')
        #####decoder
        x_t_, x_v_, x_a_ = self.decoder_t(torch.cat([x_te, x_tm], -1)), self.decoder_v(torch.cat([x_ve, x_vm], -1)), self.decoder_a(torch.cat([x_ae, x_am], -1))
        ###loss
        t_truth = torch.zeros(self.length * self.batch).long().cuda()
        v_truth = torch.ones(self.length * self.batch).long().cuda()
        a_truth = 2 * torch.ones(self.length * self.batch).long().cuda()
        loss_m = (self.criterion1(prediction1, t_truth) + self.criterion1(prediction2, v_truth) + self.criterion1(prediction3, a_truth)) /3
        loss_d = (self.criterion2(x_t_, text) + self.criterion2(x_v_, visual) + self.criterion2(x_a_, acoustic)) / 3- \
                    (self.criterion2(x_te, x_tm) + self.criterion2(x_ve, x_vm) + self.criterion2(x_ae, x_am)) / 3
                    
        ###psr
        x_te = self.intra_t(x_te)
        x_ae = self.intra_a(x_ae)
        x_ve = self.intra_v(x_ve)

        x_t_norm, x_a_norm, x_v_norm = F.normalize(x_te, dim=2), F.normalize(x_ae, dim=2), F.normalize(x_ve, dim=2)
        corr_t, corr_a, corr_v = torch.matmul(x_t_norm, x_t_norm.permute(0, 2, 1)), torch.matmul(x_a_norm, x_a_norm.permute(0, 2,
                                                                                                                  1)), torch.matmul(x_v_norm, x_v_norm.permute(0, 2, 1))
        
        corr_t, corr_a, corr_v = (corr_t+1)/2, (corr_a+1)/2, (corr_v+1)/2  #180

        w = self.weight(self.length).unsqueeze(0).cuda()
        w = w.repeat(self.batch, 1, 1)
        corr_t, corr_a, corr_v = torch.mul(corr_t, w), torch.mul(corr_a, w), torch.mul(corr_v, w)
        corr_t, corr_a, corr_v = torch.triu(corr_t, diagonal=1), torch.triu(corr_v, diagonal=1), torch.triu(corr_v, diagonal=1)
        corrt, corra, corrv = torch.sum(corr_t, (1, 2)), torch.sum(corr_a, (1, 2)), torch.sum(corr_v, (1, 2))
        corrt_mean, corra_mean, corrv_mean = corrt / (self.length * (self.length - 1) / 2), corra / (
                    self.length * (self.length - 1) / 2), corrv / (self.length * (self.length - 1) / 2)
        corr_loss = (corrt_mean + corra_mean + corrv_mean) / 3

        return torch.cat((x_te, x_ae, x_ve), 1), loss_m, loss_d, corr_loss


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config
        self.emb_size = config.emb_size
        self.length = config.seqlength
        self.dropout = config.dropout
        self.att_dp = config.atten_dropout
        self.n_head = config.n_head
        num_layer = 1

        self.proj_t = nn.Linear(self.emb_size, self.emb_size)
        self.proj_a = nn.Linear(self.emb_size, self.emb_size)
        self.proj_v = nn.Linear(self.emb_size, self.emb_size)


        self.attention_t = nn.Sequential(nn.Linear(self.emb_size * 3, 1), nn.Sigmoid())
        self.attention_a = nn.Sequential(nn.Linear(self.emb_size * 3, 1), nn.Sigmoid())
        self.attention_v = nn.Sequential(nn.Linear(self.emb_size * 3, 1), nn.Sigmoid())

        self.cross = Cross_Attention(self.emb_size, self.att_dp)

        self.cross.cuda()

        layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=self.n_head, dropout=self.att_dp)
        self.intra_modality_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.intra_modality_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.intra_modality_v = nn.TransformerEncoder(layer, num_layers=num_layer)

        

        # self.classifier_e = nn.Sequential(nn.Linear(self.emb_size * 3, self.emb_size * 2),
        #                                     nn.Dropout(self.dropout),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm1d(self.emb_size * 2),
        #                                     nn.Linear(self.emb_size * 2, self.emb_size),
        #                                     nn.Dropout(self.dropout),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm1d(self.emb_size),
        #                                     nn.Linear(self.emb_size, 8))


        self.classifier_e = nn.Sequential( 
                                           nn.Linear(self.emb_size, self.emb_size),
                                           nn.Dropout(self.dropout),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(self.emb_size),
                                           nn.Linear(self.emb_size, 8))

    

    def forward(self, inputs):
        x_t, x_a, x_v = inputs[:, :self.length], inputs[:, self.length:self.length * 2], inputs[:, self.length * 2:]
        self.batch = inputs.shape[0]
        
        action_t = self.attention_t(torch.cat((x_t, self.proj_t(x_t - x_a), self.proj_t(x_t - x_v)), -1))
        action_a = self.attention_a(torch.cat((x_a, self.proj_a(x_a - x_t), self.proj_a(x_a - x_v)), -1))
        action_v = self.attention_v(torch.cat((x_v, self.proj_v(x_v - x_t), self.proj_v(x_v - x_a)), -1))

        action = torch.cat((action_t, action_a, action_v), 1)
        return action


    def fused(self, state, action):    
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length * 2:]     
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:, self.length * 2:]
        self.batch = state_t.shape[0]

        action_t = action_t.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 
        action_a = action_a.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 
        action_v = action_v.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 
        x_t_o = self.cross(state_t, action_t)     
        x_a_o = self.cross(state_a, action_a)
        x_v_o = self.cross(state_v, action_v)

        state_new = torch.cat((x_t_o, x_a_o, x_v_o), 1)
        return state_new

    def temporal(self, inputs):
        x_t, x_a, x_v = inputs[:, :self.length], inputs[:, self.length:self.length * 2], inputs[:, self.length * 2:]
        emb_t, emb_a, emb_v = x_t.permute(1, 0, 2), x_a.permute(1, 0, 2), x_v.permute(1, 0, 2)

        emb_t = self.intra_modality_t(emb_t)
        emb_a = self.intra_modality_a(emb_a)
        emb_v = self.intra_modality_v(emb_v)

        emb_t, emb_a, emb_v = emb_t.permute(1, 0, 2), emb_a.permute(1, 0, 2), emb_v.permute(1, 0, 2)

        ####classification
        emb_t_mean, emb_a_mean, emb_v_mean = torch.mean(emb_t, 1).squeeze(), torch.mean(emb_a, 1).squeeze(), torch.mean(emb_v, 1).squeeze()
        emo_predictions = self.classifier_e((emb_t_mean + emb_a_mean + emb_v_mean)/3)
        # emo_predictions = self.classifier_e(torch.cat((emb_t_mean, emb_a_mean, emb_v_mean), -1))
        return emo_predictions

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.emb_size = config.emb_size
        self.length = config.seqlength
        self.dropout = config.dropout
        self.att_dp = config.atten_dropout
        self.n_head = config.n_head 
        num_layer = 1


        layer = nn.TransformerEncoderLayer(d_model=self.emb_size + self.n_head, nhead=self.n_head, dropout=self.att_dp)
        self.intra_modality_state_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.intra_modality_state_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.intra_modality_state_v = nn.TransformerEncoder(layer, num_layers=num_layer)
        

        self.proj = nn.Sequential(
            nn.Linear((self.emb_size + self.n_head) * 3, self.emb_size * 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_size * 2),
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.Linear(self.emb_size, 1))

    def forward(self, state, action):
        b = state.shape[0]
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:,
                                                                                                   self.length * 2:]
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:,
                                                                                                        self.length * 2:]
        
        action_t, action_a, action_v = action_t.repeat(1, 1, self.n_head), action_a.repeat(1, 1, self.n_head), action_v.repeat(1, 1, self.n_head)
        
        text, acoustic, visual = torch.cat((state_t, action_t), -1), torch.cat((state_a, action_a), -1), torch.cat((state_v, action_v), -1)

        text, visual, acoustic = text.permute(1, 0, 2), visual.permute(1, 0, 2), acoustic.permute(1,0,2)
        

        text = self.intra_modality_state_t(text)
        visual = self.intra_modality_state_v(visual)
        acoustic = self.intra_modality_state_a(acoustic)

        text, acoustic, visual = text.permute(1, 0, 2).mean(1), acoustic.permute(1, 0, 2).mean(1), visual.permute(1, 0, 2).mean(1)

        q = self.proj(torch.cat((text, acoustic, visual), -1))
        return q.squeeze()
