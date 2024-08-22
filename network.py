import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from torch.nn import functional as F


class DPSR(nn.Module):
    def __init__(self, emb_size, head, dropout):
        super(DPSR, self).__init__()

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
    

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.att_dp = config.atten_dropout
        self.emb_size = config.emb_size
        self.length = config.seqlength
        self.dim_t, self.dim_v, self.dim_a = config.dim_t, config.dim_v, config.dim_a
        self.n_head = config.n_head

        ####text encoder and decoder
        self.encoder_t = nn.Linear(self.dim_t, self.emb_size)

        ####visual encoder and decoder
        self.encoder_v = nn.Linear(self.dim_v, self.emb_size)

         ####acoustic encoder and decoder
        self.encoder_a = nn.Linear(self.dim_a, self.emb_size)

        self.dpsr_t = DPSR(self.emb_size, self.n_head, self.att_dp)
        self.dpsr_a = DPSR(self.emb_size, self.n_head, self.att_dp)
        self.dpsr_v = DPSR(self.emb_size, self.n_head, self.att_dp)

        self.dpsr_t.cuda()
        self.dpsr_a.cuda()
        self.dpsr_v.cuda()

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
        x_t, x_v, x_a = self.encoder_t(text), self.encoder_v(visual), self.encoder_a(acoustic) ##[l,b, emb_size]
                    
        ###dpsr
        x_t = self.dpsr_t(x_t)
        x_a = self.dpsr_a(x_a)
        x_v = self.dpsr_v(x_v)
        ###dpsr loss
        x_t_norm, x_a_norm, x_v_norm = F.normalize(x_t, dim=2), F.normalize(x_a, dim=2), F.normalize(x_v, dim=2)
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

        return torch.cat((x_t, x_a, x_v), 1), corr_loss


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


        self.action_t = nn.Sequential(nn.Linear(self.emb_size * 2, 1), nn.Sigmoid())
        self.action_a = nn.Sequential(nn.Linear(self.emb_size * 2, 1), nn.Sigmoid())
        self.action_v = nn.Sequential(nn.Linear(self.emb_size * 2, 1), nn.Sigmoid())

        layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=self.n_head, dropout=self.att_dp)
        self.temporal_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_v = nn.TransformerEncoder(layer, num_layers=num_layer)

        

        # self.classifier_e = nn.Sequential(nn.Linear(self.emb_size * 3, self.emb_size*3),
        #                                     nn.Dropout(self.dropout),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm1d(self.emb_size*3),
        #                                     # nn.Linear(self.emb_size, self.emb_size),
        #                                     # nn.Dropout(self.dropout),
        #                                     # nn.ReLU(),
        #                                     # nn.BatchNorm1d(self.emb_size),
        #                                     nn.Linear(self.emb_size*3, 8))


        self.classifier_e = nn.Sequential( 
                                           nn.Linear(self.emb_size, self.emb_size),
                                           nn.Dropout(self.dropout),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(self.emb_size),
                                           nn.Linear(self.emb_size, 8))

    def generate_advise(self, text, audio, vision): 
        sim_ta, sim_tv, sim_at, sim_av, sim_vt, sim_va = self.similarity_matrix(text, audio, vision)
        diff_ta, diff_tv, diff_at, diff_av, diff_vt, diff_va = -sim_ta, -sim_tv, -sim_at, -sim_av, -sim_vt, -sim_va
        ###text-center
        sim_tt = torch.ones_like(sim_ta)
        tt, ta, tv, ta_neg, tv_neg = torch.sum(sim_tt, dim= [1, 2]), torch.sum(sim_ta, dim= [1, 2]), torch.sum(sim_tv, dim= [1, 2]), torch.sum(diff_ta, dim= [1, 2]), torch.sum(diff_tv, dim= [1, 2])
        scale_t = torch.nn.Softmax(1)(torch.cat((tt.unsqueeze(1), ta.unsqueeze(1), tv.unsqueeze(1), ta_neg.unsqueeze(1), tv_neg.unsqueeze(1)), 1))
        advise_ta = torch.matmul(nn.Softmax(-1)(sim_ta), audio)
        advise_tv = torch.matmul(nn.Softmax(-1)(sim_tv), vision)        
        advise_ta_neg = torch.matmul(nn.Softmax(-1)(diff_ta), audio)
        advise_tv_neg = torch.matmul(nn.Softmax(-1)(diff_tv), vision)
        advise_t = text * repeat(scale_t[:, 0], 'b -> b t c', t=self.length, c= self.emb_size) + \
                    advise_ta * repeat(scale_t[:, 1],'b -> b t c', t=self.length, c= self.emb_size) + \
                    advise_tv * repeat(scale_t[:, 2], 'b -> b t c', t=self.length, c= self.emb_size) + \
                    advise_ta_neg * repeat(scale_t[:, 3], 'b -> b t c', t=self.length, c= self.emb_size) +\
                    advise_tv_neg * repeat(scale_t[:, 4], 'b -> b t c', t=self.length, c= self.emb_size)   
        ###audio-center
        sim_aa = torch.ones_like(sim_ta)
        aa, at, av, at_neg, av_neg = torch.sum(sim_aa, dim= [1, 2]), torch.sum(sim_at, dim= [1, 2]), torch.sum(sim_av, dim= [1, 2]), torch.sum(diff_at, dim= [1, 2]), torch.sum(diff_av, dim= [1, 2]) 
        scale_a = torch.nn.Softmax(1)(torch.cat((aa.unsqueeze(1), at.unsqueeze(1), av.unsqueeze(1), at_neg.unsqueeze(1), av_neg.unsqueeze(1)), 1))
        advise_at = torch.matmul(nn.Softmax(-1)(sim_at), text)
        advise_av = torch.matmul(nn.Softmax(-1)(sim_av), vision)        
        advise_at_neg = torch.matmul(nn.Softmax(-1)(diff_at), text)
        advise_av_neg = torch.matmul(nn.Softmax(-1)(diff_av), vision)
        advise_a = audio * repeat(scale_a[:, 0], 'b -> b t c', t=self.length, c= self.emb_size) +\
              advise_at * repeat(scale_a[:, 1], 'b -> b t c', t=self.length, c= self.emb_size) + \
              advise_av * repeat(scale_a[:, 2], 'b -> b t c', t=self.length, c= self.emb_size) + \
              advise_at_neg * repeat(scale_a[:, 3], 'b -> b t c', t=self.length, c= self.emb_size) +\
              advise_av_neg * repeat(scale_a[:, 4], 'b -> b t c', t=self.length, c= self.emb_size) 
        ###vision-center
        sim_vv = torch.ones_like(sim_ta)
        vv, vt, va, vt_neg, va_neg = torch.sum(sim_vv, dim= [1, 2]), torch.sum(sim_vt, dim= [1, 2]), torch.sum(sim_va, dim= [1, 2]), torch.sum(diff_vt, dim= [1, 2]), torch.sum(diff_va, dim= [1, 2]) 
        scale_v = torch.nn.Softmax(1)(torch.cat((vv.unsqueeze(1), vt.unsqueeze(1), va.unsqueeze(1), vt_neg.unsqueeze(1), va_neg.unsqueeze(1)), 1))
        advise_vt = torch.matmul(nn.Softmax(-1)(sim_vt), text)
        advise_va = torch.matmul(nn.Softmax(-1)(sim_va), audio)        
        advise_vt_neg = torch.matmul(nn.Softmax(-1)(diff_vt), text)
        advise_va_neg = torch.matmul(nn.Softmax(-1)(diff_va), audio)
        advise_v = vision * repeat(scale_v[:, 0], 'b -> b t c', t=self.length, c= self.emb_size) + \
            advise_vt * repeat(scale_v[:, 1], 'b -> b t c', t=self.length, c= self.emb_size) + \
            advise_va * repeat(scale_v[:, 2], 'b -> b t c', t=self.length, c= self.emb_size) + \
            advise_vt_neg * repeat(scale_v[:, 3], 'b -> b t c', t=self.length, c= self.emb_size) + \
            advise_va_neg * repeat(scale_v[:, 4], 'b -> b t c', t=self.length, c= self.emb_size)
        return advise_t, advise_a, advise_v     


    def similarity_matrix(self, emb_t, emb_a, emb_v):
        emb_t_norm = F.normalize(emb_t, p=2, dim = 2)
        emb_a_norm = F.normalize(emb_a, p=2, dim = 2)
        emb_v_norm = F.normalize(emb_v, p=2, dim = 2)

        ###text TA, TV
        similarity_ta = torch.matmul(emb_t_norm, emb_a_norm.permute(0, 2, 1))
        similarity_tv = torch.matmul(emb_t_norm, emb_v_norm.permute(0, 2, 1))

        ###audio, AT, AV
        similarity_at = torch.matmul(emb_a_norm,emb_t_norm.permute(0, 2, 1))
        similarity_av = torch.matmul(emb_a_norm, emb_v_norm.permute(0, 2, 1))

        ###vision VT, VA
        similarity_vt = torch.matmul(emb_v_norm, emb_t_norm.permute(0, 2, 1))
        similarity_va = torch.matmul(emb_v_norm, emb_a_norm.permute(0, 2, 1))

        return similarity_ta, similarity_tv, similarity_at, similarity_av, similarity_vt, similarity_va


    def forward(self, inputs):
        x_t, x_a, x_v = inputs[:, :self.length], inputs[:, self.length:self.length * 2], inputs[:, self.length * 2:]
        self.batch = inputs.shape[0]

        x_t_, x_a_, x_v_ = self.proj_t(x_t), self.proj_a(x_a), self.proj_v(x_v)
        advise_t, advise_a, advise_v = self.generate_advise(x_t_, x_a_, x_v_)
        
        action_t = self.action_t(torch.cat((x_t, advise_t), -1))
        action_a = self.action_a(torch.cat((x_a, advise_a), -1))
        action_v = self.action_v(torch.cat((x_v, advise_v), -1))

        action = torch.cat((action_t, action_a, action_v), 1)
        return action


    def fused(self, state, action):    
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length * 2:]     
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:, self.length * 2:]
        self.batch = state_t.shape[0]

        action_t = action_t.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 
        action_a = action_a.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 
        action_v = action_v.reshape(self.batch, self.length, 1).repeat(1, 1, self.emb_size) 

        x_t_o = torch.mul(state_t, action_t)
        x_a_o = torch.mul(state_a, action_a)
        x_v_o = torch.mul(state_v, action_v)

        state_new = torch.cat((x_t_o, x_a_o, x_v_o), 1)
        return state_new

    def temporal(self, inputs):
        x_t, x_a, x_v = inputs[:, :self.length], inputs[:, self.length:self.length * 2], inputs[:, self.length * 2:]
        emb_t, emb_a, emb_v = x_t.permute(1, 0, 2), x_a.permute(1, 0, 2), x_v.permute(1, 0, 2)

        emb_t = self.temporal_t(emb_t)
        emb_a = self.temporal_a(emb_a)
        emb_v = self.temporal_v(emb_v)

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
        self.critic_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_v = nn.TransformerEncoder(layer, num_layers=num_layer)
        

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
        

        text = self.critic_t(text)
        visual = self.critic_v(visual)
        acoustic = self.critic_a(acoustic)

        text, acoustic, visual = text.permute(1, 0, 2).mean(1), acoustic.permute(1, 0, 2).mean(1), visual.permute(1, 0, 2).mean(1)

        q = self.proj(torch.cat((text, acoustic, visual), -1))
        return q.squeeze()
